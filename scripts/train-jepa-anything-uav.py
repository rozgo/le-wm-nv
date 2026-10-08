"""Experimental JEPA-Anything latent training; export to the native UAV runtime.

This is an OPF adaptation with our TCN/GRU backbone, not an author UAV model.
The separate native importer must verify recursive Python/Candle parity before
the frozen physics prober can be trained. No existing model is overwritten.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import h5py
import hdf5plugin  # noqa: F401 -- registers dataset compression filters
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file


class Block(nn.Module):
    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, 3, dilation=dilation)
        self.residual = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.padding = 2 * dilation

    def forward(self, x):
        h = F.gelu(self.conv1(F.pad(x, (self.padding, 0))), approximate="tanh")
        h = F.gelu(self.conv2(F.pad(h, (self.padding, 0))), approximate="tanh")
        return F.gelu(h + (self.residual(x) if self.residual is not None else x), approximate="tanh")


class TCN(nn.Module):
    def __init__(self, cin, channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i, cout in enumerate(channels):
            self.blocks.append(Block(cin, cout, 2 ** i))
            cin = cout

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        return x[:, :, -1]


class Model(nn.Module):
    def __init__(self, opf_class):
        super().__init__()
        self.state_encoder = TCN(18, [8, 8, 16])
        self.state_projection = nn.Linear(16, 24)
        self.action_encoder = TCN(4, [4, 4, 8])
        self.predictor = nn.GRUCell(8, 24)
        self.factor_head = nn.Linear(24, 24)
        self.opf = opf_class(24, 4, 6, learnable=True, orthogonality_mode="qr_retraction")

    def encode(self, states):
        return self.state_projection(self.state_encoder(states))

    def rollout(self, states, actions):
        context = self.encode(states[:, :10])
        windows = actions.unfold(1, 10, 1)[:, :20].transpose(-1, -2).contiguous()
        encoded = self.action_encoder(windows.reshape(-1, 10, 4)).reshape(-1, 20, 8)
        basis = self.opf.analysis_basis().reshape(24, 24)
        h, factors, latents = context, [], []
        for i in range(20):
            f = self.factor_head(self.predictor(encoded[:, i], h))
            h = f @ basis
            factors.append(f.reshape(-1, 4, 6))
            latents.append(h)
        return context, torch.stack(factors, 1), torch.stack(latents, 1)


def fingerprint(root):
    result = hashlib.sha256()
    for name in ("metadata.json", "data.h5", "domains.json"):
        result.update(name.encode())
        with (root / name).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                result.update(chunk)
    return result.hexdigest()


def dataset(root, reference):
    audit = json.loads((root / "audit.json").read_text())
    expected = reference["provenance"]["dataset_artifact_sha256"]
    assert audit["passed"] and audit["audit_version"] >= 2
    assert fingerprint(root) == expected == audit["artifact_sha256"]
    with h5py.File(root / "data.h5") as f:
        state, action = f["state"][:], f["action"][:]
        episodes, domains, steps = [f[key][:].reshape(-1) for key in ("episode_idx", "domain_idx", "step_idx")]
    assert json.loads((root / "metadata.json").read_text())["sample_rate_hz"] == 20
    valid = np.flatnonzero((episodes[:-29] == episodes[29:]) & (steps[29:] == steps[:-29] + 29))
    unique = np.unique(domains)
    assert len(unique) == 100
    train = valid[np.isin(domains[valid], unique[:80])]
    validation = valid[np.isin(domains[valid], unique[80:90])]
    assert sorted(np.unique(episodes[train]).tolist()) == reference["provenance"]["training_episodes"]
    norm = reference["contract"]["normalization"]
    for array, key in ((state, "state"), (action, "action")):
        array -= np.asarray(norm[key]["mean"], np.float32)
        array /= np.asarray(norm[key]["std"], np.float32)
    return torch.tensor(state, device="cuda"), torch.tensor(action, device="cuda"), train, validation


def export(model, output):
    tensors = {}
    for key, value in model.state_dict().items():
        if key.startswith("opf."):
            continue
        if key.startswith("predictor."):
            key += "_l0"
        tensors[key] = value.detach().cpu().contiguous()
    tensors["opf_basis"] = model.opf.analysis_basis().detach().reshape(24, 24).cpu().contiguous()
    save_file(tensors, str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory exists; use a new directory or --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.core_dir / "jepa-anything-core/src"))
    from jepa_anything_core import OrthogonalFactorProjection, jepa_anything_objective

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = json.loads((args.reference_package / "checkpoint.json").read_text())
    states, actions, train, validation = dataset(args.dataset_dir, reference)
    offsets = torch.arange(30, device="cuda")

    def batch(rows):
        index = torch.tensor(rows, device="cuda")[:, None] + offsets
        return states[index], actions[index]

    model = Model(OrthogonalFactorProjection).cuda()
    teacher_encoder = copy.deepcopy(model.state_encoder).requires_grad_(False)
    teacher_projection = copy.deepcopy(model.state_projection).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=1e-5)

    def loss(s, a):
        context, predicted, _ = model.rollout(s, a)
        with torch.no_grad():
            windows = s.unfold(1, 10, 1)[:, 1:21].transpose(-1, -2).contiguous()
            target = teacher_projection(teacher_encoder(windows.reshape(-1, 10, 18)))
        # Teacher features are detached; the analysis basis still receives the
        # factor regression/activity gradients, as in the upstream objective.
        return jepa_anything_objective(predicted.reshape(-1, 4, 6), model.opf(target),
            model.opf.analysis_basis(), context, orthogonality_weight=.02,
            factor_activity_weight=.10, encoder_variance_weight=.02)

    start_step, best, best_step, elapsed_prior = 0, float("inf"), 0, 0
    snapshot = args.output_dir / "training-state.pt"
    if args.resume:
        saved = torch.load(snapshot, map_location="cuda", weights_only=True)
        assert saved["seed"] == args.seed and saved["batch_size"] == args.batch_size
        model.load_state_dict(saved["model"])
        teacher_encoder.load_state_dict(saved["teacher_encoder"])
        teacher_projection.load_state_dict(saved["teacher_projection"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step, best, best_step, elapsed_prior = [saved[k] for k in ("step", "best", "best_step", "elapsed_seconds")]
    validation_rng = np.random.default_rng(31415)
    validation_rows = validation_rng.permutation(validation)[:8 * args.batch_size]
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    started = time.monotonic()
    permutation, cached_epoch = None, None
    with (args.output_dir / "metrics.jsonl").open("a" if args.resume else "x") as metrics:
        for index in range(start_step, args.steps):
            epoch, part = divmod(index, batches_per_epoch)
            if epoch != cached_epoch:
                permutation = np.random.default_rng(args.seed + epoch).permutation(train)
                cached_epoch = epoch
            selected = permutation[part * args.batch_size:(part + 1) * args.batch_size]
            if len(selected) < 2:
                raise ValueError("singleton training tail")
            step = index + 1
            lr = .005 * step / 200 if step <= 200 else .0001 + .5 * (.005 - .0001) * (1 + math.cos(math.pi * min(step - 200, 800) / 800))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            terms = loss(*batch(selected))
            terms.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            model.opf.after_optimizer_step(step)
            with torch.no_grad():
                for target_module, online_module in ((teacher_encoder, model.state_encoder), (teacher_projection, model.state_projection)):
                    for target, online in zip(target_module.parameters(), online_module.parameters()):
                        target.lerp_(online, .004)
            if step % 50 == 0 or step == 1:
                record = {"step": step, "loss": terms.total.item(), "prediction": terms.prediction.item(),
                    "factor_activity": terms.factor_activity.item(), "encoder_variance": terms.encoder_variance.item(),
                    "grad_norm": norm.item(), "lr": lr, "elapsed_seconds": elapsed_prior + time.monotonic() - started}
                metrics.write(json.dumps(record) + "\n"); metrics.flush()
                print(json.dumps(record), flush=True)
            if step % 134 == 0 or step == args.steps:
                with torch.no_grad():
                    val = float(np.mean([loss(*batch(validation_rows[i:i + args.batch_size])).prediction.item()
                        for i in range(0, len(validation_rows), args.batch_size)]))
                assert math.isfinite(val)
                if val < best:
                    best, best_step = val, step
                    torch.save(model.state_dict(), args.output_dir / "best-state.pt")
                    export(model, args.output_dir / "latent.safetensors")
                record = {"validation_step": step, "validation_prediction": val, "best": best, "best_step": best_step}
                metrics.write(json.dumps(record) + "\n"); metrics.flush(); print(json.dumps(record), flush=True)
                torch.save({"model": model.state_dict(), "teacher_encoder": teacher_encoder.state_dict(),
                    "teacher_projection": teacher_projection.state_dict(), "optimizer": optimizer.state_dict(),
                    "seed": args.seed, "batch_size": args.batch_size, "step": step, "best": best,
                    "best_step": best_step, "elapsed_seconds": elapsed_prior + time.monotonic() - started}, snapshot)

    elapsed = elapsed_prior + time.monotonic() - started
    model.load_state_dict(torch.load(args.output_dir / "best-state.pt", map_location="cuda", weights_only=True))
    with torch.no_grad():
        s, a = batch(validation_rows[:4])
        context, _, predicted = model.rollout(s, a)
        fixture = {"batch": 4, "time": 30, "states": s.flatten().tolist(), "actions": a.flatten().tolist(), "predicted": predicted.flatten().tolist()}
        geometry = (model.opf.analysis_basis().reshape(24, 24) @ model.opf.analysis_basis().reshape(24, 24).T - torch.eye(24, device="cuda")).abs().max().item()
    (args.output_dir / "parity-fixture.json").write_text(json.dumps(fixture))
    metadata = copy.deepcopy(reference["provenance"])
    metadata.update({"model_family": "jepa_anything_opf_uav_adaptation", "seed": args.seed,
        "created_at_unix": int(time.time()), "model_config": {**reference["contract"]["model"], "opf_factors": 4},
        "latent_steps": args.steps, "batch_size": args.batch_size, "elapsed_seconds": elapsed,
        "best_step": best_step, "best_validation_prediction": best, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(), "orthogonality_max_abs": geometry,
        "upstream_core_commit": "c6e6c88f3ef75a4ce7acd660d6fa5779d995512c",
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loss_config": {"factor_prediction": 1, "orthogonality": .02, "factor_activity": .10, "encoder_variance": .02, "min_std": .1, "ema": .996, "qr_retraction_every": 1},
        "experimental_scope": "single-seed complete-system comparison; same data/backbone widths; extra OPF head/basis and different latent training objective"})
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"completed": True, "elapsed_seconds": elapsed, "best_step": best_step, "best_validation": best, "trainable_parameters": metadata["trainable_parameters"]}), flush=True)


if __name__ == "__main__":
    main()

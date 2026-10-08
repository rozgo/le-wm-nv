"""Train matched representations, then independent physics/learned readouts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .models import WorldModel, Readout, predict
from .experiment import validate_data, digest


def load_data(path):
    data = np.load(path)
    return {k: torch.tensor(data[k], device="cuda") for k in ("states", "actions")}


def batch(data, rng, size, horizon=15):
    episodes = torch.tensor(rng.integers(len(data["states"]), size=size), device="cuda")
    starts = torch.tensor(rng.integers(data["states"].shape[1]-10-horizon+1, size=size), device="cuda")
    indices = starts[:, None]+torch.arange(10+horizon, device="cuda")
    return data["states"][episodes[:, None], indices], data["actions"][episodes[:, None], indices[:, :-1]]


def load_model(path):
    ckpt = torch.load(path, map_location="cuda", weights_only=True)
    model = WorldModel(ckpt["variant"], **ckpt["normalization"]).cuda()
    model.load_state_dict(ckpt["state"]); model.eval()
    return model, ckpt


def load_pair(path, kind):
    model, _ = load_model(path/"latent.pt")
    readout = Readout(kind).cuda()
    readout.load_state_dict(torch.load(path/f"{kind}.pt", map_location="cuda", weights_only=True)["state"])
    return model.eval(), readout.eval()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--variant", choices=["standard", "opf"], required=True); p.add_argument("--seed", type=int, required=True)
    args = p.parse_args(); protocol = json.loads(args.protocol.read_text())
    manifest = validate_data(args.data, protocol)
    contract = {"airframe": "x", "heading_modes": protocol.get("heading_modes", ["fixed"]),
                "protocol_sha256": digest(args.protocol), "data_manifest_sha256": digest(args.data/"manifest.json")}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    norm = manifest["normalization"]
    train, val = load_data(args.data/"train.npz"), load_data(args.data/"validation.npz")
    model = WorldModel(args.variant, **norm).cuda()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed); validation = [batch(val, np.random.default_rng(600+i), 512) for i in range(4)]
    best, started, records = float("inf"), time.monotonic(), []
    for step in range(1, protocol["latent_steps"]+1):
        optimizer.zero_grad(set_to_none=True)
        loss, pred = model.loss(*batch(train, rng, protocol["batch_size"]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); model.update_teacher()
        if step%200 == 0:
            with torch.no_grad(): metric = np.mean([model.loss(*b)[1].item() for b in validation])
            if metric < best:
                best = metric
                torch.save({"state": model.state_dict(), "normalization": norm, "variant": args.variant, "seed": args.seed, "step": step, "contract": contract}, args.output/"latent.pt")
            row = {"stage": "latent", "step": step, "loss": loss.item(), "validation": float(metric), "elapsed": time.monotonic()-started}
            records.append(row); print(json.dumps(row), flush=True)
    latent_seconds = time.monotonic()-started
    model, latent = load_model(args.output/"latent.pt")
    model.requires_grad_(False)
    # Identical sampled windows for every model arm, cached only after freezing.
    cache_rng = np.random.default_rng(args.seed+1000)
    caches = []
    with torch.no_grad():
        for _ in range(32):
            s, u = batch(train, cache_rng, 512)
            z, _ = model.rollout(s[:, :10], u[:, :9], u[:, 9:])
            caches.append((z, s[:, 9], u[:, 9:], s[:, 10:]))
        cache = [torch.cat([c[j] for c in caches]) for j in range(4)]
        validation_cache = []
        for s, u in validation:
            z, _ = model.rollout(s[:, :10], u[:, :9], u[:, 9:])
            validation_cache.append((z, s[:, 9], u[:, 9:], s[:, 10:]))
    readout_summary = {}
    for kind in protocol["readouts"]:
        torch.manual_seed(args.seed+2000)
        readout = Readout(kind).cuda()
        optimizer = torch.optim.AdamW(readout.parameters(), lr=.001, weight_decay=1e-5)
        best, started = float("inf"), time.monotonic()
        rng = np.random.default_rng(args.seed+3000)
        for step in range(1, protocol["readout_steps"]+1):
            index = torch.tensor(rng.integers(len(cache[0]), size=protocol["batch_size"]), device="cuda")
            z, initial, u, target = [c[index] for c in cache]
            optimizer.zero_grad(set_to_none=True)
            pred = readout(z, initial, u, model.sm, model.ss)
            loss = ((pred-target)/model.ss).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(readout.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if step%300 == 0:
                with torch.no_grad():
                    metric = np.mean([((readout(z, initial, u, model.sm, model.ss)-target)/model.ss).square().mean().item()
                                      for z, initial, u, target in validation_cache])
                if metric < best:
                    best = metric
                    torch.save({"state": readout.state_dict(), "step": step, "kind": kind, "validation": float(metric), "contract": contract}, args.output/f"{kind}.pt")
                row = {"stage": kind, "step": step, "loss": loss.item(), "validation": float(metric), "elapsed": time.monotonic()-started}
                records.append(row); print(json.dumps(row), flush=True)
        readout_summary[kind] = {"best_validation_normalized_mse": best, "seconds": time.monotonic()-started,
                                 "parameters": sum(p.numel() for p in readout.parameters())}
    gram = model.basis@model.basis.T
    summary = {"variant": args.variant, "seed": args.seed, "contract": contract, "latent_parameters": count, "latent_best_step": latent["step"],
               "latent_seconds": latent_seconds, "readouts": readout_summary, "basis_gram_max_error": (gram-torch.eye(24, device="cuda")).abs().max().item(),
               "data_manifest_sha256": hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest(), "metrics": records}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

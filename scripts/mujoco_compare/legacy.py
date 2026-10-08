"""Unchanged SkyJEPA checkpoint transferred into the new MuJoCo plant.

This is a transfer reference, not a capacity/data-matched training arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from .models import Readout
from .evaluate import ranking, flight


class Block(nn.Module):
    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, 3, dilation=dilation)
        self.residual = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.padding = dilation*2

    def forward(self, x):
        h = F.gelu(self.conv1(F.pad(x, (self.padding, 0))), approximate="tanh")
        h = F.gelu(self.conv2(F.pad(h, (self.padding, 0))), approximate="tanh")
        return F.gelu(h+(x if self.residual is None else self.residual(x)), approximate="tanh")


class TCN(nn.Module):
    def __init__(self, cin, channels):
        super().__init__(); self.blocks = nn.ModuleList()
        for i, cout in enumerate(channels):
            self.blocks.append(Block(cin, cout, 2**i)); cin = cout

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks: x = block(x)
        return x[:, :, -1]


class LegacySky(nn.Module):
    def __init__(self, package):
        super().__init__()
        self.state_encoder = TCN(18, [8, 8, 16]); self.state_projection = nn.Linear(16, 24)
        self.action_encoder = TCN(4, [4, 4, 8]); self.predictor = nn.GRUCell(8, 24)
        weights = load_file(str(package/"latent.safetensors"))
        weights = {k.removesuffix("_l0") if k.startswith("predictor.") else k: v for k, v in weights.items()}
        self.load_state_dict(weights, strict=True)
        norm = json.loads((package/"checkpoint.json").read_text())["contract"]["normalization"]
        for key, group, field in (("sm", "state", "mean"), ("ss", "state", "std"), ("am", "action", "mean"), ("ast", "action", "std")):
            self.register_buffer(key, torch.tensor(norm[group][field], dtype=torch.float32))

    def rollout(self, states, history_actions, actions):
        batch, horizon, _ = actions.shape
        z = self.state_projection(self.state_encoder((states-self.sm)/self.ss))
        all_actions = (torch.cat((history_actions, actions), 1)-self.am)/self.ast
        windows = all_actions.unfold(1, 10, 1).transpose(-1, -2).contiguous()
        encoded = self.action_encoder(windows.reshape(-1, 10, 4)).reshape(batch, horizon, 8)
        output = []
        for step in range(horizon):
            z = self.predictor(encoded[:, step], z); output.append(z)
        return torch.stack(output, 1), None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    model = LegacySky(args.checkpoint).cuda().eval()
    readout = Readout("physics")
    readout.net = nn.Sequential(nn.Linear(24, 32), nn.GELU(approximate="tanh"), nn.Linear(32, 32), nn.GELU(approximate="tanh"), nn.Linear(32, 15))
    weights = load_file(str(args.checkpoint/"prober.safetensors"))
    for prefix, idx in (("input", 0), ("hidden.0", 2), ("output", 4)):
        readout.net[idx].load_state_dict({k: weights[f"{prefix}.{k}"] for k in ("weight", "bias")})
    readout.cuda().eval()
    ranks = ranking(model, readout, np.load(args.run/"ranking-ground-truth.npz"))
    flights = []
    for split, seed_base in (("test", 72000), ("combination", 73000)):
        for i in range(8):
            for task in ("circle", "figure8"):
                result = flight(model, readout, seed_base+i, split, task, 74000+i)
                flights.append({k: v for k, v in result.items() if k not in ("samples", "predictions")})
                print(json.dumps({"legacy": "skyjepa-transfer", "split": split, "domain": i, "kind": task, "rmse": result["rmse"]}), flush=True)
    (args.run/"evaluation"/"skyjepa-transfer.json").write_text(json.dumps({"label": "SkyJEPA seed 7 / unchanged transfer", "ranking": ranks, "flights": flights,
       "scope": "Trained on the previous simulator and 0.17 m arms. New MuJoCo drone has 0.24 m arms. Not a matched training comparison.",
       "checkpoint_sha256": {f: hashlib.sha256((args.checkpoint/f).read_bytes()).hexdigest() for f in ("latent.safetensors", "prober.safetensors")}}, indent=2))


if __name__ == "__main__": main()

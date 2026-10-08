"""Summarize paired seeds without selecting a favorable flight or condition."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); a = p.parse_args()
    results = json.loads((a.run/"results.json").read_text())
    experiments = [json.loads(f.read_text()) for f in (a.run/"evaluation").glob("*.json")
                   if re.fullmatch(r"(standard|opf)-(physics|learned)-\d+\.json", f.name)]
    seeds = sorted({e["seed"] for e in experiments})
    paired = []
    for kind in ("physics", "learned"):
        for split in ("test", "combination"):
            for seed in seeds:
                row = {"readout": kind, "split": split, "seed": seed}
                for variant in ("standard", "opf"):
                    e = next(x for x in experiments if x["variant"] == variant and x["readout"] == kind and x["seed"] == seed)
                    row[variant] = {
                        "regret": float(np.mean([r["normalized_regret"] for r in e["ranking"] if r["split"] == split])),
                        "flight_rmse": float(np.mean([r["rmse"] for r in e["flights"] if r["split"] == split])),
                        "heading_rmse_deg": float(np.mean([r.get("heading_rmse_deg", 0.) for r in e["flights"] if r["split"] == split])),
                        "contacts": sum(r["contact_steps"] for r in e["flights"] if r["split"] == split)}
                row["opf_minus_standard"] = {k: row["opf"][k]-row["standard"][k] for k in ("regret", "flight_rmse")}
                paired.append(row)
    results["paired_training_seeds"] = paired
    legacy = a.run/"evaluation"/"skyjepa-transfer.json"
    if legacy.exists():
        data = json.loads(legacy.read_text())
        results["skyjepa_transfer"] = [{"split": split,
             "flight_rmse": float(np.mean([r["rmse"] for r in data["flights"] if r["split"] == split])),
             "regret": float(np.mean([r["normalized_regret"] for r in data["ranking"] if r["split"] == split])),
             "contacts": sum(r["contact_steps"] for r in data["flights"] if r["split"] == split)} for split in ("test", "combination")]
    sources = list((a.run/"evaluation").glob("*.json"))+list(a.run.glob("*/summary.json"))
    results["source_sha256"] = {str(f.relative_to(a.run)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    (a.run/"results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"aggregate": results["aggregate"], "paired": paired}, indent=2))


if __name__ == "__main__": main()

"""Run the fixed single-seed UAV comparison and record synchronized video traces.

Requires finished training and frozen executables in ROOT/bin. Every result is
bound to executable/checkpoint hashes. No publication or upload occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(root, binary, checkpoint, arguments, output):
    executable = root / "bin" / binary
    command = [str(executable), "--checkpoint-dir", str(checkpoint), *map(str, arguments), "--output", str(output)]
    identity = {"argv": command, "executable_sha256": sha(executable),
        "checkpoint_sha256": sha(checkpoint / "checkpoint.json")}
    sidecar = output.with_suffix(".invocation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        assert json.loads(sidecar.read_text()) == identity
        print("reuse", output.name, flush=True)
        return
    if sidecar.exists():
        assert json.loads(sidecar.read_text()) == identity
    else:
        sidecar.write_text(json.dumps(identity, indent=2))
    print("run", output.name, flush=True)
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("done", output.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    args = parser.parse_args()
    root, reference = args.artifact_root.resolve(), args.reference_root.resolve()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            executable = (p / "exe").resolve(strict=True).name
        except (OSError, RuntimeError):
            continue
        if executable in ("lewm-train-skyjepa", "cargo", "rustc", "nvcc"):
            raise RuntimeError(f"wait for {executable} PID {p.name} before timing")
    checkpoints = {"skyjepa": reference / "seed-7/prober", "opf": root / "opf-prober",
        "nominal": reference / "seed-7/prober"}
    for population, distribution in (("in-range", "training-ranges"), ("shifted", "extended-mass-and-motor-lag")):
        for name, checkpoint in checkpoints.items():
            run(root, "lewm-bench-skyjepa", checkpoint, ["--controller", "nominal-physics-mppi" if name == "nominal" else "trained-mppi",
                "--warm-start", "fresh-prior", "--trim-multiplier", 1, "--domain-distribution", distribution,
                "--domain-seed", 9182026, "--random-domains", 20, "--samples", 512, "--horizon", 15,
                "--planner-seed", 7, "--duration-seconds", 8, "--radius-m", 2, "--period-seconds", 8, "--allow-fail"],
                root / "control" / f"{name}-{population}.json")
    for scenario in ("circle", "figure-eight"):
        for name in ("skyjepa", "opf"):
            run(root, "lewm-sim-skyjepa", checkpoints[name], ["--reference", scenario, "--randomize-domain",
                "--domain-seed", 9182027, "--samples", 512, "--horizon", 15, "--planner-seed", 7,
                "--warm-start", "fresh-prior", "--control-steps", 400], root / "traces" / f"{name}-{scenario}.json")
    for name in ("skyjepa", "opf"):
        run(root, "lewm-eval-skyjepa", checkpoints[name], ["--dataset-dir", reference / "data-pilot",
            "--split", "test", "--rollout-steps", 60, "--batch-size", 512], root / "open-loop" / f"{name}.json")


if __name__ == "__main__":
    main()

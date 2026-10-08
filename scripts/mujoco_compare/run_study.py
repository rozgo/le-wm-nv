"""Run a frozen, new study in an isolated directory; never reuse old weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .environment import ROOT
from .experiment import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-workers", type=int, default=2)
    args = p.parse_args()
    args.run, args.protocol = args.run.resolve(), args.protocol.resolve()
    args.run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.run/"protocol.json")
    protocol = json.loads(args.protocol.read_text())
    paths = list((ROOT/"scripts/mujoco_compare").glob("*.py"))+list((ROOT/"assets/mujoco").glob("*.xml"))+[args.protocol]
    source_manifest = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = args.run/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
        source_manifest[str(relative)] = digest(path)
    (args.run/"source-manifest.json").write_text(json.dumps(source_manifest, indent=2))
    env = dict(os.environ, PYTHONPATH=str(ROOT/"scripts"), MUJOCO_GL="egl")
    started = time.time()

    def stage(module, argv, log):
        command = [sys.executable, "-m", f"mujoco_compare.{module}", *map(str, argv)]
        print(json.dumps({"event": "start", "log": log, "elapsed_seconds": time.time()-started}), flush=True)
        with (args.run/log).open("w") as output:
            subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"event": "complete", "log": log, "elapsed_seconds": time.time()-started}), flush=True)

    try:
        stage("data", ["--output", args.run/"data", "--protocol", args.protocol], "data.log")
        jobs = [(variant, seed) for seed in protocol["seeds"] for variant in protocol["variants"]]
        def train(job):
            variant, seed = job
            stage("train", ["--data", args.run/"data", "--output", args.run/f"{variant}-{seed}",
                            "--protocol", args.protocol, "--variant", variant, "--seed", seed], f"{variant}-{seed}.log")
        with ThreadPoolExecutor(max_workers=args.training_workers) as pool:
            list(pool.map(train, jobs))
        stage("evaluate", ["--run", args.run, "--protocol", args.protocol, "--stage", "ground-truth"], "ground-truth.log")
        # Serial evaluation makes each measured planning interval comparable.
        stage("evaluate", ["--run", args.run, "--protocol", args.protocol, "--stage", "models"], "evaluation.log")
        stage("summarize", ["--run", args.run], "summary.log")
        (args.run/"complete.json").write_text(json.dumps({"elapsed_seconds": time.time()-started,
            "training_workers": args.training_workers, "evaluation_workers": 1, "protocol_sha256": digest(args.protocol)}, indent=2))
    except BaseException as error:
        (args.run/"failed.json").write_text(json.dumps({"elapsed_seconds": time.time()-started, "error": str(error)}, indent=2))
        raise


if __name__ == "__main__": main()

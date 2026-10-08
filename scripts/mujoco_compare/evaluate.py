"""Paired action-ranking and flight evaluations on untouched MuJoCo domains."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .environment import Plant, sample_domain, reference, geometric, prior, DT
from .models import predict, costs
from .train import load_pair
from .experiment import control_tasks, trace_name, validate_data, digest, require_x_artifact
from .reference_plan import ReferencePlan


def candidates(base, rng, count):
    noise = rng.normal(0, .18, (count, len(base), 4))
    for j in range(1, len(base)): noise[:, j] = .65*noise[:, j-1]+.76*noise[:, j]
    values = np.clip(base[None]+noise, 0, 12).astype(np.float32)
    values[0] = base
    return values


def numpy_cost(states, targets, actions, base):
    weights = np.array([400.]*3+[40.]*3+[20.]*12)
    return (((states-targets)**2*weights).sum(-1)+.05*((actions-base)**2).sum(-1)).mean(-1)


def build_ranking(data_dir, protocol, output):
    validate_data(data_dir, protocol)
    config = protocol["candidate_ranking"]
    horizon = config["horizon"]
    entries = []
    for split in ("test", "combination"):
        dataset = np.load(data_dir/f"{split}.npz")
        for seed in np.unique(dataset["domains"]):
            plant = Plant(sample_domain(int(seed), split))
            eps = np.flatnonzero(dataset["domains"] == seed)
            for k in range(protocol["candidate_ranking"]["contexts_per_domain"]):
                ep = eps[config.get("episode_indices", list(range(1, 1+config["contexts_per_domain"])))[k]]
                step = config.get("context_steps", [40+30*j for j in range(config["contexts_per_domain"])])[k]
                history = dataset["states"][ep, step-9:step+1]
                past = dataset["actions"][ep, step-9:step]
                base = dataset["actions"][ep, step:step+horizon]
                goals = dataset["states"][ep, step+1:step+1+horizon].copy()
                goals[:, :3] += np.array([.25, -.15, .05])
                actions = candidates(base, np.random.default_rng(int(seed)*10+k), config["candidates"])
                actual = []
                for sequence in actions:
                    plant.restore(dataset["snapshots"][ep, step])
                    actual.append([plant.step(u)[0] for u in sequence])
                actual = np.asarray(actual)
                entries.append({"split": split, "seed": int(seed), "context": k, "airframe": "x",
                                "heading": str(dataset["headings"][ep]) if "headings" in dataset else "fixed", "history": history,
                                "past": past, "actions": actions, "goals": goals, "base": base,
                                "actual": actual, "costs": numpy_cost(actual, goals, actions, base)})
            print(json.dumps({"ranking_ground_truth": int(seed), "split": split}), flush=True)
    np.savez_compressed(output, **{key: np.asarray([e[key] for e in entries]) for key in entries[0]})


@torch.no_grad()
def ranking(model, readout, dataset):
    rows = []
    for i in range(len(dataset["seed"])):
        count = len(dataset["actions"][i])
        ts = lambda x: torch.as_tensor(x, device="cuda", dtype=torch.float32)
        actions = ts(dataset["actions"][i])
        pred = predict(model, readout, ts(dataset["history"][i])[None].expand(count, -1, -1),
                       ts(dataset["past"][i])[None].expand(count, -1, -1), actions)
        scores = costs(pred, ts(dataset["goals"][i]), actions, ts(dataset["base"][i])).cpu().numpy()
        truth = dataset["costs"][i]
        selected = int(np.argmin(scores))
        regret = (truth[selected]-truth.min()) / max(np.percentile(truth, 90)-truth.min(), 1e-6)
        correlation = np.corrcoef(scores.argsort().argsort(), truth.argsort().argsort())[0, 1]
        error = pred.cpu().numpy()-dataset["actual"][i]
        rows.append({"split": str(dataset["split"][i]), "domain": int(dataset["seed"][i]), "context": int(dataset["context"][i]),
                     "heading": str(dataset["heading"][i]) if "heading" in dataset else "fixed",
                     "normalized_regret": float(regret), "rank_correlation": float(correlation),
                     "position_rmse": float(np.sqrt(np.mean(np.sum(error[:, :, :3]**2, -1)))),
                     "endpoint_position_rmse": float(np.sqrt(np.mean(np.sum(error[:, -1, :3]**2, -1)))),
                     "selected_candidate": selected, "oracle_candidate": int(np.argmin(truth))})
    return rows


@torch.no_grad()
def flight(model, readout, domain_seed, split, kind, planner_seed, samples=128, seconds=12, heading="fixed"):
    plant = Plant(sample_domain(domain_seed, split))
    initial = reference(0, kind, heading=heading)[0]; plant.reset(initial)
    states = [initial.copy() for _ in range(10)]
    past_actions = [np.full(4, plant.trim) for _ in range(9)]
    rng = np.random.default_rng(planner_seed)
    recorded, predictions, errors, latencies, contacts = [], [], [], [], 0
    attitude_errors, heading_errors = [], []
    plan = ReferencePlan(round(seconds/DT), plant.trim, kind, heading)
    for step in range(round(seconds/DT)):
        t = step*DT
        base, goals = plan.at(plant.state(), step)
        if model is None:
            action, predicted, elapsed = base[0], np.empty((0, 18)), 0.
        else:
            choices = candidates(base, rng, samples)
            ts = lambda x: torch.as_tensor(np.asarray(x), device="cuda", dtype=torch.float32)
            h = ts(states[-10:])[None].expand(samples, -1, -1)
            u = ts(past_actions[-9:])[None].expand(samples, -1, -1)
            actions = ts(choices)
            torch.cuda.synchronize(); started = time.perf_counter()
            pred = predict(model, readout, h, u, actions)
            score = costs(pred, ts(goals), actions, ts(base))
            # Fixed, scale-normalized MPPI weights; identical for all arms.
            weights = torch.softmax(-(score-score.min())/(score.std()+1e-5), 0)
            selected = (actions*weights[:, None, None]).sum(0)
            predicted = predict(model, readout, h[:1], u[:1], selected[None])[0].cpu().numpy()
            action = selected[0].cpu().numpy()
            elapsed = 1000*(time.perf_counter()-started)
        state, contact = plant.step(action)
        target = plan.states[step+1]
        states.append(state); past_actions.append(action)
        errors.append(np.linalg.norm(state[:3]-target[:3])); latencies.append(elapsed); contacts += int(contact)
        delta = target[6:15].reshape(3, 3).T @ state[6:15].reshape(3, 3)
        attitude_errors.append(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))))
        yaw = np.arctan2(state[9], state[6])-np.arctan2(target[9], target[6])
        heading_errors.append(float(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw)))))
        recorded.append({"time": t+DT, "state": state.tolist(), "target": target.tolist(), "action": action.tolist(),
                         "snapshot": plant.snapshot().tolist(), "error": float(errors[-1]), "latency_ms": elapsed,
                         "heading_error_deg": heading_errors[-1], "attitude_error_deg": attitude_errors[-1], "contact": contact})
        predictions.append(predicted.tolist())
    return {"domain_seed": domain_seed, "split": split, "domain": plant.domain.json(), "kind": kind,
            "airframe": "x", "heading": heading,
            "attitude_rmse_deg": float(np.sqrt(np.mean(np.square(attitude_errors)))),
            "heading_rmse_deg": float(np.sqrt(np.mean(np.square(heading_errors)))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))), "max_error": float(np.max(errors)), "contact_steps": contacts,
            "p95_ms": float(np.percentile(latencies[10:], 95)), "missed_50ms": int(np.sum(np.asarray(latencies[10:]) > 50)),
            "samples": recorded, "predictions": predictions}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--stage", choices=["ground-truth", "models"], required=True)
    args = p.parse_args(); protocol = json.loads(args.protocol.read_text()); torch.set_num_threads(4)
    validate_data(args.run/"data", protocol)
    protocol_sha = digest(args.protocol)
    tasks = control_tasks(protocol)
    if args.stage == "ground-truth":
        build_ranking(args.run/"data", protocol, args.run/"ranking-ground-truth.npz"); return
    evaluation = args.run/"evaluation"; evaluation.mkdir(exist_ok=True)
    ground_truth = np.load(args.run/"ranking-ground-truth.npz")
    if "airframe" not in ground_truth or not np.all(ground_truth["airframe"] == "x"):
        raise ValueError("Counterfactual ground truth must use the X-frame drone")
    summaries = []
    for seed in protocol["seeds"]:
        for variant in protocol["variants"]:
            for kind in protocol["readouts"]:
                label = f"{variant}-{kind}-{seed}"
                result_path = evaluation/f"{label}.json"
                if result_path.exists():
                    existing = json.loads(result_path.read_text())
                    if existing.get("protocol_sha256", protocol_sha) != protocol_sha: raise ValueError("Evaluation protocol changed")
                    summaries.append(existing); continue
                training = json.loads((args.run/f"{variant}-{seed}"/"summary.json").read_text())
                require_x_artifact(training.get("contract", {}))
                if training.get("contract", {}).get("protocol_sha256", protocol_sha) != protocol_sha: raise ValueError("Checkpoint protocol differs")
                model, readout = load_pair(args.run/f"{variant}-{seed}", kind)
                ranks = ranking(model, readout, ground_truth)
                flights = []
                for split in ("test", "combination"):
                    for i in range(protocol["domains"][split]):
                        for task in tasks:
                            trial = flight(model, readout, protocol["domain_seeds"][split]+i, split, task["kind"],
                                           protocol["control"]["planner_seed"]+i, protocol["control"]["candidates"],
                                           protocol["control"]["duration_seconds"], task["heading"])
                            trace = evaluation/trace_name(label, split, i, task["kind"], task["heading"], protocol["version"])
                            trace.write_text(json.dumps(trial))
                            flights.append({k: v for k, v in trial.items() if k not in ("samples", "predictions")})
                            print(json.dumps({"model": label, "split": split, "domain": i, "task": task, "rmse": trial["rmse"]}), flush=True)
                result = {"label": label, "seed": seed, "variant": variant, "readout": kind, "ranking": ranks, "flights": flights,
                          "airframe": "x", "protocol_sha256": protocol_sha,
                          "checkpoint_sha256": hashlib.sha256((args.run/f"{variant}-{seed}"/"latent.pt").read_bytes()).hexdigest()}
                result_path.write_text(json.dumps(result, indent=2)); summaries.append(result)
    baseline_path = evaluation/"geometric.json"
    if not baseline_path.exists():
        baseline = []
        for split in ("test", "combination"):
            for i in range(protocol["domains"][split]):
                for task in tasks:
                    trial = flight(None, None, protocol["domain_seeds"][split]+i, split, task["kind"], 0,
                                   seconds=protocol["control"]["duration_seconds"], heading=task["heading"])
                    baseline.append({k: v for k, v in trial.items() if k not in ("samples", "predictions")})
        baseline_path.write_text(json.dumps(baseline, indent=2))
    aggregate = []
    for variant in protocol["variants"]:
        for kind in protocol["readouts"]:
            for split in ("test", "combination"):
                selected = [s for s in summaries if s["variant"] == variant and s["readout"] == kind]
                ranks = [r for s in selected for r in s["ranking"] if r["split"] == split]
                flights = [r for s in selected for r in s["flights"] if r["split"] == split]
                aggregate.append({"variant": variant, "readout": kind, "split": split, "seeds": len(selected),
                                  "regret": float(np.mean([r["normalized_regret"] for r in ranks])),
                                  "rank_correlation": float(np.mean([r["rank_correlation"] for r in ranks])),
                                  "prediction_rmse": float(np.mean([r["position_rmse"] for r in ranks])),
                                  "flight_rmse": float(np.mean([r["rmse"] for r in flights])),
                                  "attitude_rmse_deg": float(np.mean([r["attitude_rmse_deg"] for r in flights])),
                                  "heading_rmse_deg": float(np.mean([r["heading_rmse_deg"] for r in flights])),
                                  "contacts": sum(r["contact_steps"] for r in flights), "flights": len(flights),
                                  "mean_p95_ms": float(np.mean([r["p95_ms"] for r in flights]))})
    by_heading = []
    for row in aggregate:
        selected = [s for s in summaries if s["variant"] == row["variant"] and s["readout"] == row["readout"]]
        for heading in protocol.get("heading_modes", ["fixed"]):
            flights = [f for s in selected for f in s["flights"] if f["split"] == row["split"] and f["heading"] == heading]
            ranks = [r for s in selected for r in s["ranking"] if r["split"] == row["split"] and r["heading"] == heading]
            by_heading.append({"variant": row["variant"], "readout": row["readout"], "split": row["split"], "heading": heading,
                               "flight_rmse": float(np.mean([f["rmse"] for f in flights])),
                               "heading_rmse_deg": float(np.mean([f["heading_rmse_deg"] for f in flights])),
                               "attitude_rmse_deg": float(np.mean([f["attitude_rmse_deg"] for f in flights])),
                               "regret": float(np.mean([r["normalized_regret"] for r in ranks])),
                               "flights": len(flights), "contact_steps": sum(f["contact_steps"] for f in flights)})
    (args.run/"results.json").write_text(json.dumps({"airframe": "x", "protocol_sha256": protocol_sha, "aggregate": aggregate,
                                                   "by_heading": by_heading, "geometric": json.loads(baseline_path.read_text())}, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__": main()

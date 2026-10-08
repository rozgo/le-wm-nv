# SkyJEPA and JEPA-Anything UAV comparison

This experiment keeps SkyJEPA intact and adds a separate JEPA-Anything OPF
adaptation for comparison. Both learned models use the same rotor simulator,
geometric command prior, metric MPPI planner and physics-prober design. The
video replays measured trajectories side by side with a shared clock, camera
and scale. It is a simulation comparison, not a real-flight demonstration.

**Measured result, September 18, 2026.** SkyJEPA was more accurate and faster
than this first OPF adaptation in both control conditions. All three
comparators passed 126/126 tracking cases, with no ground contact or 50 ms
control-deadline misses. Nominal-physics MPPI remained the strongest control
reference. The [machine-readable results](../benchmarks/jepa-anything/results.json)
retain exact values, hashes, prediction horizons and video-flight errors.

| Model | Unseen-drone mean RMSE | Shifted mean RMSE | Planning p95, unseen / shifted |
| --- | ---: | ---: | ---: |
| SkyJEPA | 0.1976 m | 0.2059 m | 5.48 / 5.71 ms |
| JEPA-Anything OPF adaptation | 0.2359 m | 0.2329 m | 6.31 / 6.21 ms |
| Nominal-physics MPPI | 0.1705 m | 0.1665 m | 0.72 / 0.73 ms |

Open-loop evaluation covered all 26,400 valid 60-step windows from the 200
held-out-domain episodes, with zero training-domain overlap. The following
are position-vector RMSE values on that exact population:

| Horizon | SkyJEPA | OPF adaptation | Constant velocity |
| --- | ---: | ---: | ---: |
| 0.25 s | 0.0305 m | 0.0813 m | 0.0537 m |
| 0.75 s | 0.2860 m | 0.4144 m | 0.4525 m |
| 1.00 s | 0.5107 m | 0.6689 m | 0.7668 m |
| 3.00 s | 8.1649 m | 8.8209 m | 3.3228 m |

OPF therefore did not improve the long-horizon weakness in this trial.
Validation selected OPF latent step 134 from the 1,000 executed updates and
prober step 4,995 from 5,000 updates. Its latent stage took 24.33 seconds in
PyTorch and its native prober stage took 275.40 seconds; the framework and
objective differences prevent interpreting this as a training-speed victory.

The finished 48-second, 1920×1080, 30 fps MP4 is saved locally under
`reports/jepa-comparison-20260918/video/skyjepa-vs-jepa-anything.mp4`, with raw
reports and a video provenance manifest alongside it. It shows both fixed-seed
flights and the aggregate results, including the OPF model's weaker outcome.

The fixed [protocol](../benchmarks/jepa-anything/protocol.json) uses training
seed 7, the existing audited 1,600 training episodes, 200 validation episodes
and 200 test episodes, separated by physical domain. It reuses training-only
normalization. OPF receives 1,000 latent updates followed by 5,000 native
prober updates, with batch size 2,048. Checkpoints are selected on validation
data. Existing SkyJEPA seed-7 weights are retained as the reference.

The control evaluation uses fresh domain seed `9182026`, 20 randomized drones
plus a nominal anchor, and hover, circle and figure-eight trajectories. Each
condition therefore contains 63 flights. The shifted condition changes mass
and motor lag for its 60 randomized flights and retains three nominal anchors.
Every comparator uses 512 MPPI candidates, a 15-step horizon, planner seed 7
and a fresh geometric warm start. Nominal-physics MPPI supplies a third
reference. Training and build processes finish before timing begins; other
workloads on the shared GPU are recorded in the native benchmark reports.

Video seed `9182027` was fixed before evaluating either model. Both circle and
figure-eight clips run for 20 simulated seconds at normal speed. Footage is
not selected by which model wins. Position comes from the recorded simulator
states, interpolated from 20 Hz to 30 fps; attitude uses the recorded rotation.
The drone glyph is enlarged for legibility. The last eight seconds show
aggregate control results taken directly from the benchmark JSON files.

**What the OPF adaptation changes.** It keeps the TCN state/action encoder
widths, the 24-wide recurrent state and the 20-step training rollout. A new
linear output predicts four six-dimensional factor groups from a shared GRU
transition. A learned orthonormal basis synthesizes the complete next latent,
which becomes the recurrent state for the following transition. Training uses
an EMA target encoder, factor regression, coordinate activity and encoder
variance terms from the pinned upstream core, with QR retraction after each
optimizer step. Inference loads the frozen basis once and performs no QR.

This is our UAV adaptation of the released OPF method. The authors did not
supply a drone checkpoint. It has 7,104 latent parameters versus SkyJEPA's
5,928, a 19.84% increase. The different objective, extra parameters and
single training seed limit causal and generalization claims. This experiment
compares complete implementations; it does not isolate the effect of OPF.
Experimental latent training uses PyTorch, while prober training, inference,
planning and evaluation use Rust/Candle CUDA. Training times must not be read
as a framework-independent comparison of the two research methods.

**Compatibility checks.** The native importer verified the trained model's
20-step Python/Candle rollout with maximum absolute error `8.9406967e-8`.
Its basis Gram-matrix error was `4.7683716e-7`. The targeted native test run
passed 54 tests, including checkpoint integrity, backward compatibility,
training contracts and existing model/control math. Original checkpoints
retain version 2 and their previous serialization; OPF packages use version 3
so an older runtime cannot silently interpret them as SkyJEPA weights.

**Reproduction.** Keep the original `skyjepa-remediation-v3` artifacts and
create a new run directory. Pin
[JEPA-Anything](https://github.com/Gen-Verse/JEPA-Anything/tree/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c)
to `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`. Build the five binaries listed
below, then copy them into the run's `bin/` directory to keep them fixed.

```bash
cargo build --release --locked \
  --bin lewm-import-opf --bin lewm-train-skyjepa \
  --bin lewm-bench-skyjepa --bin lewm-sim-skyjepa --bin lewm-eval-skyjepa

# REF: existing remediation artifacts; RUN: new comparison directory;
# CORE: checkout of the pinned upstream JEPA-Anything repository.
uv run --locked scripts/train-jepa-anything-uav.py \
  --dataset-dir "$REF/data-pilot" --reference-package "$REF/seed-7/latent" \
  --core-dir "$CORE" --output-dir "$RUN/opf-latent-python"

"$RUN/bin/lewm-import-opf" \
  --reference-package "$REF/seed-7/latent" \
  --weights "$RUN/opf-latent-python/latent.safetensors" \
  --metadata "$RUN/opf-latent-python/metadata.json" \
  --fixture "$RUN/opf-latent-python/parity-fixture.json" \
  --output-dir "$RUN/opf-latent"

"$RUN/bin/lewm-train-skyjepa" --stage prober \
  --dataset-dir "$REF/data-pilot" --latent-checkpoint "$RUN/opf-latent" \
  --output-dir "$RUN/opf-prober" --split-by domains --seed 7 --batch-size 2048 \
  --prober-max-steps 5000 --warmup-steps 500 --cosine-steps 4500 \
  --max-lr 0.005 --min-lr 0.0001 --log-every 100 --save-every 1000

uv run --locked scripts/evaluate-jepa-comparison.py "$RUN" "$REF"
uv run --locked scripts/render-jepa-comparison.py "$RUN" \
  --output-dir "$RUN/video" --preview-only
# Inspect the previews, then omit --preview-only to encode the MP4.
```

The GPU run lives at
`/home/rozgo/.stable_worldmodel/le-wm-nv-data/jepa-comparison-20260918`.
Its `source-manifest.json` records source hashes; invocation sidecars bind
reports to frozen executable and checkpoint hashes. `video-manifest.json`
binds the finished video to the reports used to draw it. Original GPU-server
files, unrelated jobs and RustDesk are preserved.

# JEPA-Anything assessment — 2026-09-18

**Recommendation: run a bounded comparison as an experimental model variant.**
JEPA-Anything is technically usable on our GPU and offers a relevant research
direction for longer predictions. The available evidence does not establish
that it would outperform SkyJEPA or LeWM on our workloads. Keep the current
models as measured baselines and require a UAV experiment before adoption.

Reviewed our source at `01d9e58`, upstream GitHub at
[`c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`](https://github.com/Gen-Verse/JEPA-Anything/tree/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c),
the September 17 paper, and the Hugging Face release at
[`9fbcdb06e80a136a783b839c11ff5d5b561db853`](https://huggingface.co/datasets/Gen-Verse/jepa-anything/tree/9fbcdb06e80a136a783b839c11ff5d5b561db853).

**What “Anything” means.** Orthogonal Predictive Factorization (OPF) divides a
latent state into complementary coordinate groups, predicts the groups, and
synthesizes the next latent state. Observation adapters, encoders, data and
weights remain task dependent. This gives us a reusable modeling approach;
it does not provide a pretrained model that already understands our drone or
arbitrary new vehicles. The released library contains projection, objectives,
baselines and diagnostics. Its factors acquire physical meaning only if
experiments establish that meaning.
[Project explanation](https://github.com/Gen-Verse/JEPA-Anything/blob/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c/README.md),
[core interface](https://github.com/Gen-Verse/JEPA-Anything/blob/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c/jepa-anything-core/README.md).

**The release is more substantial than GitHub alone suggests.** GitHub's
checkpoint manifest contains only an untrained synthetic example. However,
the separate Hugging Face **dataset repository** contains 36 checkpoint files,
shared inference loaders and domain documentation. The collection labels the
release as a full project plus domain checkpoints. Three locomotion files
contain both `standard` and `jepa_anything` branches. The domain release is
useful for inspection and inference; I did not find a locomotion training loop
or a complete CEM evaluation runner in its file tree. The reusable core is
version 0.3.0, marked alpha, with Python >=3.10 and PyTorch >=2.1; its license
is Apache-2.0.
[GitHub manifest](https://github.com/Gen-Verse/JEPA-Anything/blob/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c/checkpoints/manifest.json),
[domain artifacts](https://huggingface.co/datasets/Gen-Verse/jepa-anything/tree/9fbcdb06e80a136a783b839c11ff5d5b561db853/08_Locomotion),
[package metadata](https://github.com/Gen-Verse/JEPA-Anything/blob/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c/jepa-anything-core/pyproject.toml).

The Mac's saved Hugging Face login was valid. The initial access failure came
from querying a model endpoint: this release lives under `/datasets/`.
Once the repository type was corrected, its public artifacts were downloadable
without copying credentials to the GPU machine.

**Published evidence is promising and task dependent.** The paper reports
improvements across ten matched dynamics tasks. Its separate five-seed
continuous-control comparison improves mean CEM return on Walker2d and
HalfCheetah, while Hopper favors standard JEPA. Long-rollout studies include
50-step Burgers forecasting and 100-step molecular prediction. These are
different systems, metrics and time scales from our quadrotor. I found no
direct SkyJEPA, LeWM or UAV comparison in the paper. These author-reported
results motivate an experiment; this investigation did not reproduce them.
[Paper, especially sections 3.3.2–3.3.5](https://arxiv.org/html/2609.20800v1).

**Fit with our current system.** The most useful comparison is with SkyJEPA's
latent predictor. LeWM already supports image and vector encoders in this
repository, so observation generality alone is insufficient reason to switch.

| Concern | Current le-wm-nv | JEPA-Anything implication |
| --- | --- | --- |
| Drone inputs | State18, four commanded rotor forces, temporal histories | Reuse these data contracts; released locomotion weights have different inputs and actions. |
| Dynamics | TCN encoders and a recurrent 24-wide GRU latent | Build and train a factorized transition with an explicit recurrent-state contract. |
| Training | SkyJEPA latent MSE plus SIGReg; targets receive gradients | Reference JEPA uses EMA targets and OPF objectives; isolate these changes experimentally. |
| Metric predictions | Learned physics prober and SO(3) integration | Preserve the interface, but retrain the prober when the representation changes. |
| Control | Geometric warm start, metric MPPI, 512 candidates, 15 steps | Keep planner settings fixed to isolate model quality. |
| Deployment | Rust/Candle CUDA, 20 Hz controller | Port a successful variant; the PyTorch release is not a native checkpoint replacement. |

Local sources: [SkyJEPA guide](skyjepa.md),
[recurrent model](../src/models/skyjepa/model.rs),
[training objective](../src/models/skyjepa/training.rs),
[fixed evaluation protocol](../benchmarks/skyjepa/protocol.json).
Upstream source:
[EMA baseline implementation](https://github.com/Gen-Verse/JEPA-Anything/blob/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c/jepa-anything-core/src/jepa_anything_core/baselines.py).

Our current weakness makes this worth testing: external in-range three-second
position RMSE is 10.40–12.28 m across SkyJEPA seeds, versus 3.30 m for constant
velocity. Short feedback-controlled flight is much stronger. Exact-trim
tracking averages 0.201–0.206 m with learned MPPI, but nominal-physics MPPI
still leads at 0.161 m and about 3.2 ms p95, versus approximately 8.8–8.9 ms
for learned MPPI. A new model must therefore improve useful prediction or
control under the existing runtime budget, rather than only improve latent
training loss. These are our recorded results, not new measurements.
[Current results and limits](skyjepa.md).

**What was actually checked on the RTX 4090.** Using the existing
PyTorch `2.12.0+cu130` environment on `rozgo-sim-pc`:

- The GitHub core completed three tiny CUDA optimizer updates at `d=24`,
  `K=4`, `r=6`, with finite gradients for the encoder, predictor and basis.
  Analysis followed by pseudoinverse synthesis reconstructed random states
  with maximum absolute error `3.4571e-6`.
- Downloaded the three pinned locomotion checkpoints, loaded their dictionaries
  with `weights_only=True`, and instantiated the released dynamics class.
  Both branches of each model produced finite outputs through 20 recursive
  steps for four zero-state/zero-action inputs.
- The synthetic structural recipe, checkpoint manifest verifier and reference
  task-design validator passed locally. The design validator reported ten
  passes, zero warnings and zero errors. The full upstream pytest suite was
  not run.

| Released model | State / action width | Latent width | Standard / OPF inference parameters | CUDA smoke |
| --- | ---: | ---: | ---: | --- |
| Hopper | 11 / 3 | 32 | 237,387 / 238,443 | Both branches finite |
| Walker2d | 17 / 6 | 32 | 241,233 / 242,289 | Both branches finite |
| HalfCheetah | 17 / 6 | 32 | 241,233 / 242,289 | Both branches finite |

Parameter counts above cover reconstructed inference `nn.Parameter` objects;
the loader stores the learned basis as a buffer. They are not full training
parameter counts and should not be used to audit the paper's capacity matching.
These checks establish execution compatibility, not physical validity,
benchmark scores, training memory requirements or controller latency.
The compact sizes make a small UAV trial plausible on our 24 GB GPU, but its
actual training budget remains to be measured. Existing jobs were preserved.

**Implementation details that matter.** The current core learns an analysis
basis and uses its pseudoinverse for synthesis. The released locomotion loader
instead constructs an orthonormal basis with QR inside the forward call.
It also decodes physical state and re-encodes it on each recursive call,
whereas SkyJEPA advances a latent recurrent state. Pin the intended variant
and verify numerical parity before a port. For fixed deployment weights,
precompute the synthesis matrix once; repeated QR or pseudoinverse work
should not enter our planning loop. This caching proposal is an engineering
inference, not a measured optimization.
[Core geometry](https://github.com/Gen-Verse/JEPA-Anything/blob/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c/jepa-anything-core/src/jepa_anything_core/opf.py),
[released inference implementation](https://huggingface.co/datasets/Gen-Verse/jepa-anything/blob/9fbcdb06e80a136a783b839c11ff5d5b561db853/model/dynamics.py).

**Proposed experiment, not yet executed.**

1. Reuse the audited UAV data, domain-disjoint splits, training-only
   normalization, action histories and fixed state/action semantics. Retain
   physical coupling: every predictive branch may consume the full available
   context. Do not pre-label factors as position, attitude or motor dynamics.
2. Start with `d=24`, four factors of six coordinates as a proposed compact
   variant. Keep SkyJEPA unchanged as the production baseline. Compare a
   monolithic reference and an unconstrained multi-head ablation with the OPF
   candidate under the same target-encoder scheme. A separate SIGReg-based
   hybrid can test how much benefit comes from factorization versus switching
   to EMA/variance objectives. Match and report total trainable parameters,
   rollout compute, data, updates and wall time.
3. Use one seed for a bounded development pilot, then seeds 7, 17 and 29 if
   validation improves. Retrain the physics prober for each representation.
   Evaluate metric errors at 0.25, 0.75, 1 and 3 seconds, including the mass
   and motor-lag shift; retain constant-velocity and nominal-physics baselines.
4. Only after predictive improvement, run the same closed-loop comparisons
   against the geometric controller, current SkyJEPA and nominal-physics MPPI.
   Retain the 512-candidate/15-step budget, 10 ms p95 target, 50 ms deadline,
   ground-contact checks and separate tracking/timing reports. Use validation
   for choices. The existing test suite is a regression benchmark; reserve a
   fresh untouched domain sample for final confirmation after model selection.

Advance only with consistent metric improvement across seeds and acceptable
latency. A reduction in three-second error that sacrifices short-horizon
control is insufficient. Improving on nominal-physics MPPI remains a stronger
practical target than improving on SkyJEPA alone.

The investigation added this report only to the working repository. Upstream
source and small checkpoint downloads are in temporary review directories;
no production model, dependency, controller or training run was changed.

Follow-up: the user requested a comparison and video. The separate
[UAV comparison report](jepa-anything-comparison.md) records that completed
experiment; the existing SkyJEPA model remains intact.

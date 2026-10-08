# JEPA Gym comparison — September 18, 2026

JEPA Gym is operational in MuJoCo, with a charcoal engineering scene,
interactive telemetry, heading control and recorded-model replay. The matched
three-seed comparison is complete. OPF improves mean position prediction with
the learned readout, but this experiment does not establish an overall
action-selection or flight-control advantage.

The [environment and reproduction guide](mujoco-engineering-gym.md) describes
the model, training, controller and data contracts. Exact values and source
hashes are in [results.json](../benchmarks/mujoco-jepa/results.json).

## What the comparison actually measures

These are local hybrid controllers in the LE-WM research pipeline. The earlier
video label "Standard JEPA" was too broad; the corrected labels are:

| Display name | What it is |
| --- | --- |
| LE-WM / JEPA baseline | New MLP history encoder, GRU predictor and EMA JEPA objective, trained locally on the MuJoCo data. |
| LE-WM / OPF variant | The same local architecture with Orthogonal Predictive Factorization adapted from JEPA-Anything. |
| Original SkyJEPA, transferred | The existing TCN/GRU checkpoint with its physics prober, evaluated separately on MuJoCo. |

Neither matched arm is an official upstream pretrained model. They share the
LE-WM-style state/action contract, MPPI controller, geometric action prior and
readout design. The physics readout combines learned outputs with supplied
equations; the learned readout predicts state directly, but still uses explicit
state observations and the shared geometric action prior. This is an ablation
of OPF within our pipeline, not a standalone JEPA-versus-JEPA-Anything product
comparison. The machine-readable `standard` variant identifier remains unchanged
to preserve checkpoint and trace provenance.

## Held-out heavy-and-slow combinations

Each row averages three training seeds over the same eight held-out domains.
The ranking test uses 32 saved contexts and 32 candidate action sequences per
seed. Each row contains 48 closed-loop flights. Lower numbers are better.

| Readout | Model | Action regret | Position prediction RMSE | Flight tracking RMSE |
| --- | --- | ---: | ---: | ---: |
| Physics | LE-WM / JEPA baseline | 0.1526 | 0.0942 m | 0.0541 m |
| Physics | LE-WM / OPF variant | 0.1560 | 0.1033 m | 0.0518 m |
| Learned | LE-WM / JEPA baseline | 0.2884 | 0.5420 m | 0.0757 m |
| Learned | LE-WM / OPF variant | 0.2907 | 0.4954 m | 0.0743 m |

With the learned readout, OPF reduces mean prediction error by 8.6% and mean
tracking error by 1.8%, while action regret is 0.8% higher. With the physics
readout, its mean tracking error is 4.1% lower, but prediction error and action
regret are worse. These small tracking differences are not a broad superiority
claim. The supplied physics remains valuable: both learned readouts have
substantially larger physical prediction errors than their physics counterparts.

The first seed's action-selection improvement does not persist consistently:

| Seed | JEPA baseline learned-readout regret | OPF variant learned-readout regret |
| --- | ---: | ---: |
| 7 | 0.3048 | 0.2543 |
| 17 | 0.3501 | 0.2315 |
| 29 | 0.2102 | 0.3862 |

## Ordinary held-out domains

| Readout | Model | Action regret | Position prediction RMSE | Flight tracking RMSE |
| --- | --- | ---: | ---: | ---: |
| Physics | LE-WM / JEPA baseline | 0.0584 | 0.0722 m | 0.0531 m |
| Physics | LE-WM / OPF variant | 0.0851 | 0.0850 m | 0.0542 m |
| Learned | LE-WM / JEPA baseline | 0.3162 | 0.5543 m | 0.0813 m |
| Learned | LE-WM / OPF variant | 0.2911 | 0.4996 m | 0.0855 m |

OPF's learned readout reduces mean prediction error by 9.9% and action regret
by 7.9%, while mean tracking error is 5.1% higher. Better reconstruction alone
is insufficient to guarantee better closed-loop decisions.

All 384 matched learned-model flights complete without ground contact. Mean
per-flight prediction/scoring p95 is approximately 14 ms with the physics
readout and 3.2 ms with the learned readout. These Python/CUDA timings exclude
the shared action-prior construction and simulation, and are not native-runtime
or complete control-loop timing claims.

## Practical references

| Reference | Ordinary-domain tracking RMSE | Combination-domain tracking RMSE |
| --- | ---: | ---: |
| Geometric controller | 0.0501 m | 0.0457 m |
| Unchanged SkyJEPA seed 7, transferred | 0.0564 m | 0.0521 m |

The geometric controller remains the strongest mean tracking reference here.
The unchanged SkyJEPA checkpoint is evaluated through a Python weight adapter
on the new MuJoCo plant; it was trained on the earlier simulator, data and
rotor geometry. It is a transfer reference, not part of the capacity-matched
JEPA/OPF experiment. Its 32 flights and the geometric reference's 32 flights
also have zero ground contact.

## Videos and heading support

The local deliverables are under `reports/mujoco-jepa-20260918`:

This completed study and its videos use the original **plus-frame** quadrotor.
The live gym subsequently changed to an **X frame** with diagonal arms, revised
thrust positions and a matching motor mixer. New X-frame inspection images and
the geometric heading demo are under `reports/mujoco-xframe-20260918`; they are
not learned-model evaluation results. The comparison videos retain the measured
plus-frame layout.

- `video/jepa-gym-comparison.mp4`: 34 seconds, 1080p30. Synchronized recorded
  flights, target/actual/predicted paths, motor commands, error charts and
  aggregate results. The fixed-heading clips use domain 73000, seed 7 and
  figure-eight motion, selected before outcomes were measured.
- `heading/jepa-gym-heading-demo.mp4`: 10 seconds, 1080p30. Face-the-path and
  independent hover-yaw demonstrations, explicitly labeled geometric control.

Heading control was added after training. It is available in the live viewer
and data-generation API; the current JEPA/OPF checkpoints were not retrained
for varying heading. The nominal 20-second tangent-heading figure-eight check
has 0.0588 m position RMSE and 3.45 degrees attitude RMSE, with no contact.
The hover yaw-sweep check covers 514 degrees with no contact. These are
controller feature checks, not evidence of learned yaw generalization.
See [heading checks](../benchmarks/mujoco-jepa/heading-checks.json).

Nine physics/geometry tests and two model tests passed. Mixed-heading data generation
was smoke-tested in one training and one held-out combination domain. The
interactive Mac viewer was visually checked with live heading telemetry, and
both MP4s passed full decoding and manifest/source-hash verification. The
comparison video also passed encoded-frame inspection.

The full GPU artifacts remain at
`/home/rozgo/.stable_worldmodel/le-wm-nv-data/mujoco-jepa-20260918`.
The `source/` snapshot preserves the exact fixed-heading study implementation;
the current workspace adds heading controls and corrects the landing skids to
follow the nose. Adjacent parked propellers are perpendicular and their visual
animation counter-rotates consistently with motor reaction torque. The rotor
sites, mixer, body mass and inertia retain the measured study's values. Updated
video manifests identify the corrected rendering asset separately from the
frozen study source; recorded comparison trajectories and results are unchanged.
The landing footprint and visual propeller collision flags did change, so the
archived asset is still required to reproduce the original contact geometry.
Three additional 12-second nominal checks (fixed, tangent and sweep heading)
produce exactly the same airborne states with the revised and frozen assets.
A five-second unpowered drop settles on the skids and landing struts, with no
body-shell ground contact. The four-view inspection image and numeric audit
are under `reports/mujoco-jepa-20260918/asset-review`.
Local
reports contain the model weights, aggregate reports, selected video traces,
source snapshot and video manifests. All original models and the original
GPU repository are preserved.

The next scientific step is a fresh, explicitly yaw-varying matched experiment
with more training seeds. The present evidence supports continued investigation
of OPF's learned readout, while retaining the physics and geometric references.

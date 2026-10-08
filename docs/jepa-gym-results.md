# JEPA Gym comparison — 18 September 2026

**One X-frame drone, two matched local JEPA models.** The active gym has one
asset and one motor mixer; there is no frame selector. Both models use the
same drone, data, controller and test conditions.

The five-seed result does **not demonstrate better flight control from OPF**.
On held-out higher-mass/slower-motor combinations, OPF slightly reduced mean
action regret, but increased prediction and tracking errors. These are local
LE-WM models with an OPF adaptation inspired by JEPA-Anything, not evaluations
of official upstream pretrained checkpoints.

[Watch the 42-second comparison](media/jepa-gym-comparison.mp4).
The video shows both readouts during turning flight, then independent yaw in
place. Its domain (83000) and training seed (7) were fixed before testing.
The closing table includes all five seeds, rather than only the illustrated
flights. MuJoCo poses and prediction paths come from recorded runs; propeller
animation is illustrative.

## Held-out results

Lower is better in every metric below. Tracking and heading values are means
of per-flight RMSEs, not one pooled RMSE. Prediction values average per-context
position RMSE across 32 candidate action sequences and a 0.75-second horizon.

| Readout | Local model | Action regret | Prediction RMSE | Tracking RMSE | Heading RMSE |
|---|---|---:|---:|---:|---:|
| Physics | JEPA baseline | 0.2408 | 0.1189 m | 0.0313 m | 0.9712° |
| Physics | OPF variant | 0.2353 | 0.1427 m | 0.0348 m | 0.9700° |
| Learned | JEPA baseline | 0.5277 | 0.5015 m | 0.0667 m | 0.9980° |
| Learned | OPF variant | 0.5236 | 0.5986 m | 0.0760 m | 0.9903° |

OPF reduced mean action regret by 2.3% with the physics readout and 0.8% with
the learned readout. The improvement occurred in only three of five and two
of five paired training seeds, respectively. This is weak evidence of an
action-selection benefit. Tracking error increased by 11.2% and 13.8%; OPF
improved tracking in only one of five physics seeds and no learned seeds.
Prediction error increased by 20.0% and 19.4%.

Ordinary test conditions also favored the baseline for tracking: 0.0410 versus
0.0470 m with physics, and 0.0914 versus 0.1011 m with the learned readout.
All 960 learned-model flights completed with **zero contact steps**. The 48
geometric-controller reference flights also had zero contacts and achieved
lower tracking error: 0.0322 m on ordinary conditions and 0.0225 m on the
held-out combinations. Neither learned controller improved on that reference
in aggregate.

Action regret measures the true cost of the model's best-ranked candidate
relative to the best available candidate, normalized by the gap to the 90th
percentile cost. Closed-loop MPPI instead applies a weighted average of
candidates, so a ranking improvement need not translate into better tracking.
Five training seeds do not establish a general conclusion about JEPA-Anything.

## What was held constant

Both representations have 40,928 trainable parameters and use a state-history
encoder, recurrent latent predictor and EMA target encoder. OPF adds four
factor groups and an orthonormal synthesis basis. The shared recurrent trunk
means this is an adaptation, not the paper's exact independent-head design.
Each representation receives both a physics readout and a learned state
readout. The latter removes the supplied dynamics integrator but still uses
a decoder, MPPI and the shared geometric action prior; it does not demonstrate
complete discovery of dynamics without prior structure.

The frozen [protocol](../benchmarks/mujoco-xframe/protocol.json) used seeds
7, 17, 29, 43 and 59, 864 data episodes across 72 physical conditions, and
validation-only checkpoint selection. Of those episodes, 576 were training
data. Closed-loop tests covered 16 unseen conditions, with fixed-heading
circles, tangent-heading figure eights and hover with independent yaw.
Counterfactual evaluation used 96 saved contexts and 3,072 MuJoCo rollouts.
No test-result tuning was performed.

The mean per-flight p95 prediction/scoring interval was about 14 ms for the
physics readout and 3.3 ms for the learned readout on the RTX 4090. This excludes
prior construction, input preparation, simulation and rendering; it is not a
complete control-loop timing measurement.

## Artifacts and checks

[Full results](../benchmarks/mujoco-xframe/results.json) include per-heading
breakdowns and paired seed results. The [gym guide](mujoco-engineering-gym.md)
describes the model, drone and reproduction commands. Local reports contain
the weights, summaries, six video traces and source snapshots; all 960 raw
flight traces and datasets remain on the GPU server at
`/home/rozgo/.stable_worldmodel/le-wm-nv-data/mujoco-xstudy-20260918`.

The single-frame cleanup reproduced sampled data episodes and recorded
trajectories exactly. Geometry, dynamics, model and contract checks passed
(16 tests). Training and evaluation source hashes are archived separately from
the rendering source, and the video manifest identifies its six source traces,
results, renderer and canonical X-frame scene.
The exported video passed a complete decode check: 1,260 frames at 1920×1080,
30 fps, lasting 42 seconds. Encoded frames were visually checked across every
comparison segment and the results screen.

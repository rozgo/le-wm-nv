# JEPA Gym

JEPA Gym compares two local JEPA models using **one X-frame drone** in MuJoCo.
The drone, environment, observations, controller and experimental conditions
are shared. There is no airframe selector or airframe comparison.

The [completed five-seed comparison](jepa-gym-results.md) includes the video,
measurements and interpretation.

## Drone and gym

The charcoal gym includes a floor grid, landing pads, reference hoops,
lane markings and live telemetry. The drone has diagonal carbon arms,
motor housings, a forward camera and landing skids aligned with its nose.
The original asset is `assets/mujoco/charcoal_drone.xml`.

Body +X points forward, +Y left and +Z up. Motor order is front-left,
rear-left, rear-right, front-right. Each motor is 0.24 m from the center,
45 degrees from the body axes; the nose points between the front motors.
The physical thrust sites and motor mixer use those same positions.

MuJoCo simulates rigid-body motion and ground contact at 400 Hz, with 20 Hz
control. Each motor has first-order lag, thrust gain and reaction torque.
An explicit linear drag force models air resistance. Training aids have no
collision response. Propeller animation is illustrative rather than measured
RPM; animated blades and translucent discs do not affect collisions.

## Interactive viewer

On this Mac, the prepared isolated environment launches with:

```bash
scripts/open-jepa-gym.command
scripts/open-jepa-gym.command --trajectory hover --heading sweep --camera closeup
```

The default flight is a figure eight with heading following the path.
`--heading fixed`, `tangent` and `sweep` specify heading behavior. Sweep commands
0.45 rad/s independent yaw. Space pauses; R resets. MuJoCo mouse controls orbit,
pan and zoom. `--camera closeup` follows the drone closely. `--mass` and
`--motor-lag` vary physical conditions of this same drone.

The live viewer identifies its geometric controller. To inspect an actual
recorded learned-model flight, including predicted paths, use
`--trace path/to/flight.json`. Current artifacts identify the X-frame geometry;
older experiments must use their archived source to avoid misrepresenting
recorded motor commands.

On other machines install `scripts/mujoco_compare/requirements.txt` in an
isolated environment and run `PYTHONPATH=scripts python -m mujoco_compare.viewer`.
Use MuJoCo's `mjpython` launcher on macOS. The viewer does not import PyTorch.

## Model comparison

- **LE-WM / JEPA baseline:** locally trained MLP history encoder, GRU latent
  predictor and EMA target encoder.
- **LE-WM / OPF variant:** the matched architecture with Orthogonal Predictive
  Factorization adapted from JEPA-Anything.

These are custom hybrid controllers, not official upstream pretrained models.
They share the LE-WM state/action contract, MPPI controller and geometric action
prior. This experiment measures what OPF adds inside that pipeline. The existing
SkyJEPA checkpoint is preserved separately and is not either matched model.

Both representations have 40,928 trainable parameters, excluding the frozen
EMA encoder. Ten state18 observations and nine motor4 commands encode a
24-dimensional latent. Predictions cover 15 steps, or 0.75 seconds. The
baseline uses a dense synthesis matrix. OPF uses four groups of six coordinates,
factor-coordinate targets and an orthonormal synthesis matrix maintained by QR.
Its factor outputs share one recurrent trunk. This tests the factorization and
its training constraints together, not a paper-exact independent-head model.

Each representation is frozen before fitting two separate readouts:

- **Physics:** learned acceleration residuals and angular-action map followed
  by supplied thrust, gravity and SO(3) integration equations.
- **Learned:** direct normalized state prediction without that physics
  integrator. This remains a decoder and retains the shared action prior.

## Fixed protocol

`benchmarks/mujoco-xframe/protocol.json` was fixed before training. It specifies
five seeds (7, 17, 29, 43, 59), 4,000 representation updates and 6,000 updates
per readout, using matched initialization seeds, batches and optimizer settings.
Validation selects checkpoints; test results never select hyperparameters.

Data contains 864 ten-second episodes across 72 physical configurations:
48 training, eight validation, eight ordinary test and eight held-out
heavy-and-slow combinations. Each configuration contributes 12 episodes with
balanced fixed, tangent and sweep heading modes. Mass and motor lag vary in
training, but the joint heavy-and-slow region is held out. Normalization uses
training data only. All configurations use the same X-frame geometry.

The primary metric is normalized candidate-action regret on held-out
combinations. Six contexts per test configuration cover all three heading modes.
From each exact saved MuJoCo state, including motor activation, 32 candidate
sequences establish counterfactual ground truth. Lower regret is better:
`(selected true cost - best true cost) / (90th-percentile true cost - best true cost)`.

Closed-loop tests use 128 candidates and a 15-step horizon. Every model/readout/
seed performs the same 48 twelve-second flights: fixed-heading circles,
tangent-heading figure eights and independent hover yaw in 16 test configurations.
The complete comparison contains 960 learned-model flights plus 48 geometric
reference flights. Trials start on the moving reference state. Position,
attitude, heading error, contacts and planning latency are recorded.

Planning times measure CUDA prediction/scoring/selected-rollout computation and
CPU result transfer. They exclude prior construction, input preparation,
simulation and rendering, and are not complete control-loop deadline claims.
Reference-dependent work is cached; fresh state feedback still determines the
first prior action. Cached and direct prior calculations were checked for parity.

## Reproduce

Set `PYTHONPATH=scripts` and choose a new output directory:

```bash
python -m unittest mujoco_compare.test_environment mujoco_compare.test_models mujoco_compare.test_experiment
python -m mujoco_compare.run_study --run /path/to/new-run --protocol "$PWD/benchmarks/mujoco-xframe/protocol.json"
MUJOCO_GL=egl python -m mujoco_compare.render --run /path/to/new-run
```

The runner saves source hashes, creates data, trains the two variants, evaluates
counterfactual actions and closed-loop flights, then summarizes paired seeds.
Training runs two jobs concurrently; evaluation is serial on the GPU.
It refuses to overwrite an existing run directory.

The video uses recorded MuJoCo poses and recorded model predictions. Domain
83000 and training seed 7 were selected before evaluation. It shows both
readouts on tangent-heading flight and a close-up learned-readout hover-yaw
comparison, followed by aggregate results. The renderer needs ffmpeg and uses
MuJoCo EGL. Decorative propeller animation does not represent inferred dynamics.

GPU artifacts for this study are under
`/home/rozgo/.stable_worldmodel/le-wm-nv-data/mujoco-xstudy-20260918`.
Earlier measurements and their source snapshots remain in their separate report
directories. The native Rust/Candle implementation and original GPU repository
are preserved.

References: [MuJoCo MJCF](https://mujoco.readthedocs.io/en/stable/XMLreference.html),
[JEPA-Anything assessment](jepa-anything-assessment.md).

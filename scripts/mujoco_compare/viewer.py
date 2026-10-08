"""Interactive engineering gym; run with mjpython on macOS."""
import argparse
import json
from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np
from .environment import Plant, Domain, reference, geometric, animate_propellers, DT
from .experiment import require_x_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", choices=["hover", "circle", "figure8"], default="figure8")
    parser.add_argument("--mass", type=float, default=1.3)
    parser.add_argument("--motor-lag", type=float, default=.04)
    parser.add_argument("--heading", choices=["fixed", "tangent", "sweep"], default="tangent")
    parser.add_argument("--camera", choices=["overview", "closeup"], default="overview")
    parser.add_argument("--trace", type=Path, help="Replay a recorded comparison flight, including predicted paths")
    parser.add_argument("--duration", type=float, help="Close after this many wall-clock seconds")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text()) if args.trace else None
    domain = Domain(**trace["domain"]) if trace else Domain(mass=args.mass, lag=args.motor_lag)
    if trace: require_x_artifact(trace)
    plant = Plant(domain)
    plant.reset(reference(0, args.trajectory, heading=args.heading)[0])
    flags = {"pause": False, "reset": False}

    def key(code):
        if code == 32: flags["pause"] = not flags["pause"]
        if code in (82, 114): flags["reset"] = True

    with mujoco.viewer.launch_passive(plant.model, plant.data, key_callback=key,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
        viewer.opt.geomgroup[:] = 1
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 132, -29, 7
        if args.camera == "closeup":
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 40, -25, 1.4
        viewer.cam.lookat[:] = [0, 0, 1.1]
        path = ([np.asarray(r["target"][:3]) for r in trace["samples"]] if trace else
                [reference(t, args.trajectory, heading=args.heading)[0][:3] for t in np.linspace(0, 10, 150)])
        trail, errors, step, predictions = [], [], 0, []
        opened = time.monotonic()
        while viewer.is_running():
            if args.duration and time.monotonic()-opened > args.duration: break
            started = time.monotonic()
            if flags["reset"]:
                plant.reset(reference(0, args.trajectory, heading=args.heading)[0]); step, trail, errors = 0, [], []
                flags["reset"] = False
            if not flags["pause"]:
                if trace:
                    index = step%len(trace["samples"])
                    if index == 0: trail = []
                    record = trace["samples"][index]
                    plant.restore(np.asarray(record["snapshot"]))
                    state, contact, error = plant.state(), plant.data.ncon > 0, record["error"]
                    predictions = np.asarray(trace["predictions"][index])[:, :3]
                    model_name = "LE-WM / OPF variant" if args.trace.name.startswith("opf-") else "LE-WM / JEPA baseline" if args.trace.name.startswith("standard-") else "Recorded model"
                    readout_name = "learned readout" if "-learned-" in args.trace.name else "physics readout"
                    mode = f"{model_name} / {readout_name} / replay"
                else:
                    target, acc = reference(step*DT, args.trajectory, heading=args.heading)
                    action = geometric(plant.state(), target, acc, plant.trim)
                    state, contact = plant.step(action)
                    error = np.linalg.norm(state[:3]-reference((step+1)*DT, args.trajectory, heading=args.heading)[0][:3])
                    mode = f"Geometric controller / {args.heading} heading"
                step += 1; trail.append(state[:3]); trail = trail[-120:]; errors.append(error)
                yaw = np.degrees(np.arctan2(state[9], state[6]))
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"JEPA GYM\n{mode}\n\nTime\nTracking error\nAltitude\nHeading\nMass\nMotor lag\nGround contact\n\nSPACE pause / R reset",
                    f"\n\n\n{plant.data.time:.2f} s\n{error:.3f} m\n{state[2]:.2f} m\n{yaw:.1f} deg\n{domain.mass:.2f} kg\n{domain.lag*1000:.0f} ms\n{contact}")])
            with viewer.lock():
                animate_propellers(plant, plant.data.time)
                if args.camera == "closeup": viewer.cam.lookat[:] = plant.data.qpos[:3]
                scene = viewer.user_scn; scene.ngeom = 0
                # Avoid importing the EGL renderer in a desktop viewer process.
                for points, rgba, width in ((path, [.5, .6, .7, .4], .009), (trail, [.15, .85, .7, 1.], .013), (predictions, [.97, .64, .22, 1.], .018)):
                    for a, b in zip(points[:-1], points[1:]):
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a.astype(np.float64), b.astype(np.float64)); scene.ngeom += 1
            viewer.sync()
            time.sleep(max(0, DT-(time.monotonic()-started)))


if __name__ == "__main__": main()

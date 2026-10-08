"""Engineering-gym video from recorded MuJoCo states and model predictions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .environment import Plant, Domain, sample_domain, reference, animate_propellers, SCENE
from .experiment import trace_name, require_x_artifact

BG = (13, 18, 24)
INK = (223, 233, 238)
MUTED = (132, 153, 167)
MINT = (72, 218, 183)
BLUE = (123, 177, 244)
AMBER = (243, 188, 91)


def font(size, mono=False):
    root = Path("/usr/share/fonts/truetype/dejavu")
    path = root/("DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf")
    if not path.exists():
        path = Path("/System/Library/Fonts/Menlo.ttc" if mono else "/System/Library/Fonts/Supplemental/Arial.ttf")
    return ImageFont.truetype(str(path), size)


def line(scene, start, end, color, width=.01, arrow=False):
    if np.linalg.norm(np.asarray(end)-start) < 1e-6 or scene.ngeom >= scene.maxgeom: return
    geom = scene.geoms[scene.ngeom]
    typ = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    mujoco.mjv_initGeom(geom, typ, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, np.float32))
    mujoco.mjv_connector(geom, typ, width, np.asarray(start, np.float64), np.asarray(end, np.float64))
    scene.ngeom += 1


class View:
    def __init__(self, width, height, domain=Domain()):
        self.plant = Plant(domain)
        self.renderer = mujoco.Renderer(self.plant.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth, self.camera.elevation = 132, -35
        self.camera.distance = 5.2
        self.camera.lookat[:] = [0, 0, 1.3]

    def draw(self, snapshot, target_path=None, trail=None, prediction=None, hero=False, phase=0):
        self.plant.restore(np.asarray(snapshot))
        animate_propellers(self.plant, phase)
        if hero:
            self.camera.lookat[:] = self.plant.data.qpos[:3]
            self.camera.distance = 1.8
            self.camera.azimuth = 120+phase*5
            self.camera.elevation = -22
        self.renderer.update_scene(self.plant.data, camera=self.camera)
        scene = self.renderer.scene
        if target_path is not None:
            for a, b in zip(target_path[:-1], target_path[1:]): line(scene, a, b, [.6, .69, .73, .35], .009)
        if trail is not None:
            for a, b in zip(trail[:-1], trail[1:]): line(scene, a, b, [.12, .84, .65, .9], .013)
        if prediction is not None and len(prediction):
            path = np.asarray(prediction)[:, :3]
            for a, b in zip(path[:-1], path[1:]): line(scene, a, b, [.97, .64, .22, 1.], .018)
        position = self.plant.data.qpos[:3]
        R = self.plant.data.xmat[self.plant.body].reshape(3, 3)
        for axis, color in enumerate(([.9, .3, .3, .9], [.3, .9, .5, .9], [.3, .6, 1., .9])):
            line(scene, position, position+.30*R[:, axis], color, .012, True)
        line(scene, [*position[:2], .02], position, [.4, .55, .6, .25], .005)
        return Image.fromarray(self.renderer.render())

    def close(self): self.renderer.close()


def header(canvas, stage, timecode=None):
    d = ImageDraw.Draw(canvas)
    d.rectangle((48, 33, 57, 62), fill=MINT)
    d.text((76, 28), "JEPA GYM", font=font(25), fill=INK)
    d.text((76, 66), stage, font=font(15, True), fill=MUTED)
    d.text((1570, 32), "MUJOCO  3.13", font=font(19, True), fill=MUTED)
    if timecode is not None: d.text((1630, 62), f"T + {timecode:05.2f} s", font=font(17, True), fill=INK)
    d.line((48, 102, 1872, 102), fill=(42, 56, 67), width=1)


def panel(canvas, xy, trace, step, label, accent, picture):
    x, y = xy; d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((x, y, x+896, y+820), radius=14, fill=(20, 28, 36), outline=(47, 61, 72), width=1)
    canvas.paste(picture, (x+8, y+58))
    d = ImageDraw.Draw(canvas)
    d.rectangle((x+23, y+22, x+29, y+41), fill=accent)
    d.text((x+43, y+15), label, font=font(23), fill=INK)
    d.text((x+655, y+20), "RECORDED TELEMETRY", font=font(13, True), fill=MUTED)
    record = trace["samples"][step]
    if "heading_error_deg" in record:
        d.text((x+438, y+22), f"YAW ERR {record['heading_error_deg']:+.1f} deg", font=font(12, True), fill=MUTED)
    middle = ("HEADING", np.degrees(np.arctan2(record["state"][9], record["state"][6])), "deg") if trace.get("heading", "fixed") != "fixed" else ("SPEED", np.linalg.norm(record["state"][3:6]), "m/s")
    for offset, title, value, unit in ((24, "POSITION ERROR", record["error"], "m"),
                                       (244, *middle),
                                       (464, "ALTITUDE", record["state"][2], "m")):
        d.text((x+offset, y+602), title, font=font(12, True), fill=MUTED)
        d.text((x+offset, y+624), f"{value:.2f}", font=font(33, True), fill=INK)
        d.text((x+offset+170 if title == "HEADING" else x+offset+113, y+644), unit, font=font(13, True), fill=MUTED)
    d.text((x+692, y+602), "MOTOR COMMAND / N", font=font(11, True), fill=MUTED)
    for i, u in enumerate(record["action"]):
        left = x+701+i*40
        d.rounded_rectangle((left, y+632, left+16, y+674), radius=3, fill=(44, 56, 65))
        d.rounded_rectangle((left, y+674-min(42, u/12*42), left+16, y+674), radius=3, fill=accent)
        d.text((left-4, y+680), f"{u:.1f}", font=font(10, True), fill=MUTED)
    d.text((x+24, y+701), f"TRACKING ERROR  /  {trace['samples'][-1]['time']:g} s  /  0 - 0.5 m", font=font(12, True), fill=MUTED)
    gx, gy, gw, gh = x+24, y+802, 846, 66
    for fraction in (0, .5, 1): d.line((gx, gy-gh*fraction, gx+gw, gy-gh*fraction), fill=(39, 51, 61))
    pts = [(gx+j/(len(trace["samples"])-1)*gw, gy-min(1., r["error"]/.5)*gh) for j, r in enumerate(trace["samples"][:step+1])]
    if len(pts)>1: d.line(pts, fill=accent, width=2)


def footer(canvas, message, predictions=True):
    d = ImageDraw.Draw(canvas)
    d.text((51, 968), message, font=font(17), fill=INK)
    description = "measured states + recorded predictions" if predictions else "measured MuJoCo states"
    d.text((51, 1010), f"SIMULATION REPLAY  /  {description}  /  synchronized telemetry", font=font(13, True), fill=MUTED)
    legends = [(1110, (153, 176, 186), "REFERENCE"), (1340, MINT, "ACTUAL")]
    if predictions: legends.append((1550, AMBER, "PREDICTED .75 s"))
    for x, color, label in legends:
        d.line((x, 976, x+32, 976), fill=color, width=3)
        d.text((x+45, 967), label, font=font(13, True), fill=MUTED)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--preview", action="store_true")
    a = p.parse_args(); out = a.run/"video"; out.mkdir(exist_ok=True)
    protocol_path = a.run/"protocol.json"
    protocol = json.loads(protocol_path.read_text())
    require_x_artifact(protocol)
    video_config = protocol["video"]
    heading = video_config.get("heading", "fixed")
    hero = View(1920, 840)
    snap = hero.plant.snapshot()
    canvas = Image.new("RGB", (1920, 1080), BG)
    canvas.paste(hero.draw(snap, hero=True, phase=0), (0, 112))
    header(canvas, "01 / JEPA MODEL COMPARISON")
    d = ImageDraw.Draw(canvas)
    d.text((80, 140), "CHARCOAL\nFLIGHT LAB", font=font(44), fill=INK, spacing=8)
    d.text((84, 266), "STATE18  /  MOTOR4\n400 Hz physics · 20 Hz control", font=font(17, True), fill=MUTED, spacing=9)
    footer(canvas, "Quadrotor testbed  /  rotor forces, motor lag, rigid-body motion, ground contact", predictions=False)
    canvas.save(out/"gym-preview.png")
    hero.close()
    if a.preview: return
    result = json.loads((a.run/"results.json").read_text())
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out/"jepa-gym-comparison.mp4")]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=(out/"ffmpeg.log").open("w"))
    for _ in range(90): encoder.stdin.write(canvas.tobytes())
    sources = []
    frame_count = 90
    segments = [("physics", video_config["kind"], heading, 12, "comparison-physics"),
                ("learned", video_config["kind"], heading, 12, "comparison-learned")]
    if "yaw_kind" in video_config:
        segments.append(("learned", video_config["yaw_kind"], video_config["yaw_heading"], 8, "comparison-yaw"))
    for segment, (kind, task, mode, seconds, preview_name) in enumerate(segments, 2):
        title = f"{segment:02d} / {mode.upper()} HEADING / {kind.upper()} READOUT"
        traces = []
        for variant in ("standard", "opf"):
            label = f"{variant}-{kind}-{video_config['training_seed']}"
            domain_index = video_config["domain_seed"]-protocol["domain_seeds"]["combination"]
            path = a.run/"evaluation"/trace_name(label, "combination", domain_index, task, mode, protocol["version"])
            traces.append(json.loads(path.read_text())); sources.append(path)
        for trace in traces: require_x_artifact(trace)
        views = [View(880, 520, Domain(**trace["domain"])) for trace in traces]
        target_path = np.array([r["target"][:3] for r in traces[0]["samples"]])
        last_index = min(len(trace["samples"]) for trace in traces)-1
        clip_frames = min(seconds*30, (last_index+1)*30//20)
        for frame in range(clip_frames):
            t = frame/30; index = min(last_index, int(t/.05)); fraction = (t/.05)%1
            canvas = Image.new("RGB", (1920, 1080), BG); header(canvas, title, min(12., t+.05))
            for i, (trace, view) in enumerate(zip(traces, views)):
                first = np.asarray(trace["samples"][index]["snapshot"])
                second = np.asarray(trace["samples"][min(index+1, last_index)]["snapshot"])
                snapshot = first*(1-fraction)+second*fraction
                if np.dot(first[3:7], second[3:7]) < 0: snapshot[3:7] = first[3:7]*(1-fraction)-second[3:7]*fraction
                snapshot[3:7] /= np.linalg.norm(snapshot[3:7])
                trail = np.array([r["state"][:3] for r in trace["samples"][max(0, index-100):index+1]])
                pic = view.draw(snapshot, target_path, trail, trace["predictions"][index], hero=task == "hover", phase=t)
                panel(canvas, (48+i*928, 122), trace, index, "LE-WM / JEPA BASELINE" if i == 0 else "LE-WM / OPF VARIANT", BLUE if i == 0 else MINT, pic)
            footer(canvas, "HELD-OUT CONDITIONS  /  higher mass + slower motors")
            if frame == clip_frames//2: canvas.save(out/f"{preview_name}.png")
            encoder.stdin.write(canvas.tobytes())
            frame_count += 1
        for view in views: view.close()
    canvas = Image.new("RGB", (1920, 1080), BG); header(canvas, f"{len(segments)+2:02d} / MEASURED RESULTS / {len(protocol['seeds'])} TRAINING SEEDS")
    d = ImageDraw.Draw(canvas)
    d.text((80, 147), "Does factorization help this drone?", font=font(42), fill=INK)
    d.text((84, 216), "Held-out heavy + slow combinations  /  lower regret and tracking error are better", font=font(20), fill=MUTED)
    for x, txt in ((86, "READOUT / MODEL"), (640, "ACTION REGRET"), (920, "PREDICTION RMSE"), (1230, "TRACKING RMSE"), (1540, "HEADING RMSE")):
        d.text((x, 310), txt, font=font(17, True), fill=MUTED)
    selected = [r for r in result["aggregate"] if r["split"] == "combination"]
    for i, row in enumerate(selected):
        y = 370+i*95
        d.rounded_rectangle((64, y-15, 1856, y+62), radius=8, fill=(22, 31, 40))
        label = f'{row["readout"].upper()} / {"OPF VARIANT" if row["variant"] == "opf" else "JEPA BASELINE"}'
        values = [(86, label), (680, f'{row["regret"]:.3f}'), (950, f'{row["prediction_rmse"]:.3f} m'), (1260, f'{row["flight_rmse"]:.3f} m')]
        if "heading_rmse_deg" in row: values.append((1570, f'{row["heading_rmse_deg"]:.2f} deg'))
        for x, txt in values:
            d.text((x, y), txt, font=font(24, True), fill=MINT if row["variant"] == "opf" else INK)
    total_flights = sum(row["flights"] for row in result["aggregate"])
    total_contacts = sum(row["contacts"] for row in result["aggregate"])
    d.text((86, 830), f"All conditions: {total_flights} flights / {total_contacts} contact steps. Matched data and parameter counts.", font=font(24), fill=INK)
    d.text((86, 879), "Local LE-WM models; OPF adapted from JEPA-Anything. Neither is an upstream pretrained checkpoint.", font=font(19), fill=MUTED)
    d.text((86, 927), "Shared geometric action prior and readout design. Video seed and domain selected before evaluation. Simulation only.", font=font(19), fill=MUTED)
    canvas.save(out/"results-preview.png")
    for _ in range(210): encoder.stdin.write(canvas.tobytes())
    frame_count += 210
    encoder.stdin.close(); assert encoder.wait() == 0
    video = out/"jepa-gym-comparison.mp4"
    (out/"manifest.json").write_text(json.dumps({"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "frames": frame_count, "fps": 30, "seconds": frame_count/30, "airframe": "x",
        "sources": {str(p.relative_to(a.run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "results_sha256": hashlib.sha256((a.run/"results.json").read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "render_scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "scene_revision": "Single X-frame drone with matching thrust positions and mixer",
        "rendering": "MuJoCo EGL; position and quaternion interpolation between recorded steps; decorative propeller rotation is illustrative"}, indent=2))


if __name__ == "__main__": main()

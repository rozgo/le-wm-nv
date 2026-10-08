"""Recorded heading-control demonstration, explicitly using geometric control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from .environment import Plant, reference, geometric, SCENE
from .render import View, header, footer, font, BG, INK, MUTED, MINT


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output/"jepa-gym-heading-demo.mp4"
    pipe = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=(args.output/"ffmpeg.log").open("w"))
    traces = []
    for mode, kind, seconds in (("tangent", "figure8", 6), ("sweep", "hover", 4)):
        plant = Plant(); plant.reset(reference(0, kind, radius=1.2, period=8, heading=mode)[0])
        samples = []
        for step in range(seconds*20):
            target, acc = reference(step*.05, kind, radius=1.2, period=8, heading=mode)
            state, contact = plant.step(geometric(plant.state(), target, acc, plant.trim))
            target = reference((step+1)*.05, kind, radius=1.2, period=8, heading=mode)[0]
            samples.append({"snapshot": plant.snapshot().tolist(), "state": state.tolist(), "target": target.tolist(), "contact": contact})
        view = View(1920, 840)
        target_path = np.array([r["target"][:3] for r in samples])
        for frame in range(seconds*30):
            index = min(len(samples)-1, int(frame/30/.05))
            record = samples[index]; state = record["state"]
            trail = np.array([r["state"][:3] for r in samples[:index+1]])
            picture = view.draw(record["snapshot"], target_path if mode == "tangent" else None,
                                trail if mode == "tangent" else None, hero=mode == "sweep", phase=frame/30)
            canvas = Image.new("RGB", (1920, 1080), BG); canvas.paste(picture, (0, 112))
            header(canvas, "HEADING CONTROL / GEOMETRIC CONTROLLER", (index+1)*.05)
            draw = ImageDraw.Draw(canvas)
            draw.text((80, 142), "FACE THE PATH" if mode == "tangent" else "INDEPENDENT YAW", font=font(37), fill=INK)
            draw.text((83, 199), "MuJoCo physics / recorded flight", font=font(17, True), fill=MUTED)
            yaw = np.degrees(np.arctan2(state[9], state[6]))
            draw.text((82, 260), f"{yaw:+06.1f} deg", font=font(34, True), fill=MINT)
            draw.text((83, 310), f"Position error  {np.linalg.norm(np.array(state[:3])-record['target'][:3]):.3f} m", font=font(17, True), fill=MUTED)
            footer(canvas, "Heading feature demonstration  /  learned-model comparison is a separate fixed-heading test", predictions=False)
            if frame == seconds*15: canvas.save(args.output/f"heading-{mode}.png")
            pipe.stdin.write(canvas.tobytes())
        traces.append({"airframe": "x", "heading": mode, "controller": "geometric", "samples": samples}); view.close()
    pipe.stdin.close(); assert pipe.wait() == 0
    trace_path = args.output/"heading-traces.json"; trace_path.write_text(json.dumps(traces))
    (args.output/"manifest.json").write_text(json.dumps({"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(), "frames": 300, "fps": 30,
        "airframe": "x",
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "scope": "MuJoCo heading-control feature demonstration using geometric control, not a learned-model benchmark",
        "rendering": "Recorded 20 Hz states displayed at 30 fps; decorative propeller animation"}, indent=2))


if __name__ == "__main__": main()

"""Render synchronized UAV comparison replays from measured simulator traces.

No generated flight motion: positions interpolate recorded 20 Hz states, and
attitude uses the nearest recorded rotation. Both panels share camera, scale,
time and reference. Aggregate callouts are read from the benchmark reports.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
BG, PANEL, WHITE, MUTED = "#08121e", "#102133", "#eff6fc", "#9fb6c9"
COLORS = ("#56dbe8", "#ffbe73")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@lru_cache(maxsize=32)
def font(size, bold=False):
    family = "DejaVu Sans:style=" + ("Bold" if bold else "Book")
    path = subprocess.check_output(["fc-match", "-f", "%{file}", family], text=True)
    return ImageFont.truetype(path, size)


def text(draw, xy, value, size=26, color=WHITE, bold=False):
    draw.text(xy, value, font=font(size, bold), fill=color, anchor="lt")


def aggregate(report):
    rows = report["results"]
    errors = [r["position_vector_rmse_m"] for r in rows]
    complete = all(x is not None and np.isfinite(x) for x in errors)
    return {"runs": len(rows), "passes": report["tracking_successful_runs"],
        "mean_rmse_m": float(np.mean(errors)) if complete else None,
        "worst_rmse_m": max(errors) if complete else None,
        "p95_ms": report["aggregate_p95_plan_ms"],
        "contacts": sum(r["ground_contact"] for r in rows),
        "deadline_misses": report["control_deadline_misses"]}


class Flight:
    def __init__(self, path):
        self.path, self.report = Path(path), read(path)
        trace = self.report["trace"]
        self.times = np.array([0] + [r["time_seconds"] for r in trace])
        self.position = np.array([[0, 0, 1]] + [r["state"]["position"] for r in trace])
        initial_reference = [0, 0, 1.2 if self.report["reference"] == "figure_eight" else 1.0]
        self.reference = np.array([initial_reference] + [r["reference"][:3] for r in trace])
        self.rotation = np.array([np.eye(3).flatten().tolist()] + [r["state"]["rotation_world_from_body"] for r in trace]).reshape(-1, 3, 3)
        assert len(trace) == 400 and self.report["finite"]

    def sample(self, seconds):
        index = min(np.searchsorted(self.times, seconds, side="right") - 1, len(self.times) - 2)
        index = max(index, 0)
        fraction = np.clip((seconds - self.times[index]) / (self.times[index + 1] - self.times[index]), 0, 1)
        position = self.position[index] * (1 - fraction) + self.position[index + 1] * fraction
        reference = self.reference[index] * (1 - fraction) + self.reference[index + 1] * fraction
        return index, position, reference, self.rotation[index]


class Video:
    def __init__(self, root):
        self.root = root
        self.flights = {scenario: [Flight(root / "traces" / f"{model}-{scenario}.json") for model in ("skyjepa", "opf")]
            for scenario in ("circle", "figure-eight")}
        for flights in self.flights.values():
            a, b = [x.report for x in flights]
            for key in ("domain", "samples", "horizon", "dt_seconds", "reference"):
                assert a[key] == b[key], (key, "unmatched comparison")
            assert np.allclose(flights[0].times, flights[1].times)
            assert np.allclose(flights[0].reference, flights[1].reference)
        self.summary = {population: {name: aggregate(read(root / "control" / f"{name}-{population}.json"))
            for name in ("skyjepa", "opf", "nominal")} for population in ("in-range", "shifted")}
        # Fix framing from both full trajectories. The panels never zoom independently.
        positions = np.concatenate([flight.position for flights in self.flights.values() for flight in flights])
        self.scale = min(110., 780 / max(np.ptp(positions[:, :2], axis=0).max() * 1.5, 7.))

    def projection(self, points, center):
        points = np.asarray(points)
        projected = np.stack((.82 * points[..., 0] + .57 * points[..., 1],
            .29 * points[..., 0] - .42 * points[..., 1] - .85 * points[..., 2]), axis=-1)
        return projected * self.scale + np.asarray(center)

    def panel(self, image, flight, time, side):
        draw = ImageDraw.Draw(image)
        x = 48 + side * 920
        color = COLORS[side]
        draw.rounded_rectangle((x, 176, x + 904, 835), 22, fill=PANEL, outline="#24425a", width=2)
        text(draw, (x + 28, 201), "SkyJEPA" if side == 0 else "JEPA-Anything / OPF", 33, color, True)
        text(draw, (x + 28, 249), "Existing UAV model" if side == 0 else "Experimental UAV adaptation", 23, MUTED)
        outer = draw
        scene = Image.new("RGB", (856, 450), PANEL)
        draw = ImageDraw.Draw(scene)
        center = (428, 350)
        for value in range(-4, 5):
            for points in (([value, -4, 0], [value, 4, 0]), ([-4, value, 0], [4, value, 0])):
                draw.line([tuple(p) for p in self.projection(points, center)], fill="#20384a", width=1)
        target = self.projection(flight.reference, center)
        for i in range(0, len(target) - 2, 4):
            draw.line([tuple(p) for p in target[i:i + 3]], fill="#637c91", width=2)
        index, pos, ref, rotation = flight.sample(time)
        trail = self.projection(flight.position[:index + 1], center)
        if len(trail) > 1:
            draw.line([tuple(p) for p in trail], fill=color, width=4)
        p, r = self.projection([pos, ref], center)
        draw.line((tuple(p), tuple(r)), fill="#ec7288", width=2)
        draw.ellipse((r[0] - 5, r[1] - 5, r[0] + 5, r[1] + 5), outline=WHITE, width=2)
        # The glyph is enlarged for legibility; its pose comes from the trace.
        arms = np.array([[-.22, -.22, 0], [.22, .22, 0], [-.22, .22, 0], [.22, -.22, 0]])
        ends = self.projection(arms @ rotation.T + pos, center)
        for a, b in ((0, 1), (2, 3)):
            draw.line((tuple(ends[a]), tuple(ends[b])), fill="#050b12", width=15)
            draw.line((tuple(ends[a]), tuple(ends[b])), fill=color, width=6)
        for end in ends:
            draw.ellipse((end[0] - 14, end[1] - 7, end[0] + 14, end[1] + 7), fill="#142c40", outline=color, width=2)
        draw.ellipse((p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9), fill=WHITE)
        image.paste(scene, (x + 24, 290))
        draw = outer
        error = np.linalg.norm(pos - ref)
        text(draw, (x + 28, 754), f"POSITION ERROR   {error:.3f} m", 25, WHITE, True)
        text(draw, (x + 28, 794), f"This flight RMSE  {flight.report['position_vector_rmse_m']:.3f} m", 22, MUTED)

    def frame(self, frame):
        seconds = frame / FPS
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        text(draw, (48, 32), "SkyJEPA vs JEPA-Anything", 49, WHITE, True)
        text(draw, (50, 99), "Same flight data. Same rotor simulator. Same MPPI controller.", 27, MUTED)
        if seconds < 40:
            scenario = "circle" if seconds < 20 else "figure-eight"
            elapsed = seconds % 20
            text(draw, (1358, 54), f"{scenario.upper()}  ·  {elapsed:04.1f} s", 27, WHITE, True)
            for side, flight in enumerate(self.flights[scenario]):
                self.panel(image, flight, elapsed, side)
            text(draw, (50, 865), "Solid: executed trajectory     Dashed: target     Pink: instantaneous error", 24, MUTED)
            draw.line((50, 928, 1870, 928), fill="#21394e", width=4)
            draw.line((50, 928, 50 + 1820 * elapsed / 20, 928), fill=COLORS[0], width=4)
            text(draw, (50, 959), "Recorded simulation replay · 1× speed · shared camera and scale · 512 candidates / 15 steps", 23, WHITE)
            text(draw, (50, 1006), "Training seed 7 · fixed video domain 9182027 · drone glyph enlarged for visibility", 21, MUTED)
        else:
            text(draw, (50, 179), "Measured on fresh test domains", 38, WHITE, True)
            text(draw, (50, 238), "63 flights per condition: hover, circle and figure-eight · lower error is better", 26, MUTED)
            columns = [(670, "SkyJEPA", COLORS[0]), (1090, "OPF adaptation", COLORS[1]), (1510, "Nominal physics", "#b7d39b")]
            for x, name, color in columns:
                text(draw, (x, 326), name, 29, color, True)
            for row, (population, title) in enumerate((("in-range", "Unseen drones"), ("shifted", "Heavier / slower motors"))):
                y = 412 + row * 240
                text(draw, (50, y), title, 28, WHITE, True)
                text(draw, (50, y + 55), "Mean tracking RMSE", 24, MUTED)
                text(draw, (50, y + 106), "Tracking passes / planning p95", 24, MUTED)
                for (x, _, color), name in zip(columns, ("skyjepa", "opf", "nominal")):
                    value = self.summary[population][name]
                    error = "incomplete" if value["mean_rmse_m"] is None else f"{value['mean_rmse_m']:.3f} m"
                    text(draw, (x, y + 47), error, 38, color, True)
                    text(draw, (x, y + 109), f"{value['passes']}/{value['runs']}  /  {value['p95_ms']:.1f} ms", 26, WHITE)
            text(draw, (50, 917), "Single-seed engineering comparison; extra OPF parameters and a different latent objective.", 24, WHITE)
            text(draw, (50, 965), "Both learned models use the same geometric warm start and physics-prober design.", 24, MUTED)
            text(draw, (50, 1010), "Simulation evidence only. These results do not establish general superiority or real-flight performance.", 21, MUTED)
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = Video(args.artifact_root)
    previews = []
    for second in (4, 13, 24, 35, 44):
        path = args.output_dir / f"preview-{second:02d}.png"
        video.frame(second * FPS).save(path)
        previews.append(str(path))
    if args.preview_only:
        print(json.dumps({"previews": previews})); return
    target = args.output_dir / "skyjepa-vs-jepa-anything.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(48 * FPS):
            process.stdin.write(video.frame(frame).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    sources = sorted((args.artifact_root / "traces").glob("*.json")) + sorted((args.artifact_root / "control").glob("*.json"))
    report = {"video": target.name, "sha256": sha(target), "width": W, "height": H, "fps": FPS,
        "duration_seconds": 48, "renderer_sha256": sha(__file__), "summary": video.summary,
        "source_sha256": {str(p.relative_to(args.artifact_root)): sha(p) for p in sources},
        "visualization": "1x replay of measured simulation; interpolated position, recorded attitude, enlarged drone glyph; shared camera and scale"}
    (args.output_dir / "video-manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Render body-frame inspection views of the actual compiled MuJoCo asset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .environment import SCENE, reference
from .render import View, BG, INK, MUTED, MINT, font, line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1920, 1200), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 28), "JEPA GYM / AIRFRAME REVIEW", font=font(32), fill=INK)
    draw.text((50, 80), "Body axes: red +X nose / green +Y left / blue +Z up. Parked two-blade propellers.", font=font(20), fill=MUTED)
    for i, (title, azimuth, elevation, yaw) in enumerate((
        ("TOP / heading 0 deg", 90, -89.9, 0),
        ("TOP / heading 90 deg", 90, -89.9, np.pi/2),
        ("SIDE / skids parallel to nose", 90, -5, 0),
        ("FRONT QUARTER / X frame", 40, -28, 0),
    )):
        view = View(900, 430)
        view.plant.reset(reference(yaw/.45, "hover", heading="sweep")[0])
        view.camera.azimuth, view.camera.elevation, view.camera.distance = azimuth, elevation, 1.05
        view.camera.lookat[:] = [0, 0, 1.64]
        picture = view.draw(view.plant.snapshot(), phase=0)
        # Lift the X/Y aids clear of the arms, which otherwise hide both arrows.
        R = view.plant.data.xmat[view.plant.body].reshape(3, 3)
        origin = view.plant.data.qpos[:3]+.13*R[:, 2]
        for axis, color in ((0, [.9, .3, .3, 1.]), (1, [.3, .9, .5, 1.])):
            line(view.renderer.scene, origin, origin+.18*R[:, axis], color, .007, True)
        picture = Image.fromarray(view.renderer.render())
        x, y = 48+(i % 2)*924, 136+(i//2)*490
        canvas.paste(picture, (x, y+39)); view.close()
        draw = ImageDraw.Draw(canvas)
        draw.text((x+10, y), title, font=font(20), fill=INK)
    draw.text((50, 1134), "X frame / 0.24 m motor radius / rotors 45 deg from nose / nose between front motors", font=font(20), fill=MINT)
    path = args.output/"drone-geometry-review.png"; canvas.save(path)
    (args.output/"manifest.json").write_text(json.dumps({
        "airframe": "x",
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "Four views of the compiled MuJoCo asset at two body headings; parked blade phase is illustrative",
    }, indent=2))


if __name__ == "__main__": main()

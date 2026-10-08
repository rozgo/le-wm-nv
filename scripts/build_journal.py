# /// script
# requires-python = ">=3.10"
# dependencies = ["imageio-ffmpeg>=0.5", "Pillow>=10"]
# ///
"""Build the project's engineering journal into build/journal (ignored).

The journal lives in site/journal/: journal.json (title, theme, hero, metrics, media list with repository
sources), content.html (its chapters, as <section id=...><h2>...</h2>...</section>), template.html and
journal.css, shared in layout with the MuJoCo Sandbox journals (github.com/rozgo/mujoco-sandbox).
Media are referenced in the HTML as media/<key>.mp4, media/<key>.webp and media/<key>_poster.webp.
Videos are re-encoded for the web with the bundled imageio-ffmpeg (H.264, faststart, no audio) and
cached under build/journal_media; images become WebP. Publishes nothing.

Usage:   uv run --script scripts/build_journal.py
Preview: python3 -m http.server 8792 --bind 127.0.0.1 --directory build/journal
Publish: scripts/publish_journal.sh
"""

import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"site/journal"
OUT = ROOT/"build/journal"
CACHE = ROOT/"build/journal_media"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TOKENS = ("bg", "text", "muted", "card", "line", "link", "chip", "soft", "bad", "bad-soft", "accent", "accent-2",
          "accent-3", "on-accent", "hero", "hero-text", "film", "metric", "toc", "tile-1", "tile-2", "tile-3", "tile-4",
          "tile-text")


def cached(key, suffix, make):
    """Run make(path) once per key; later builds reuse the file."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE/f"{hashlib.sha256(key.encode()).hexdigest()[:20]}{suffix}"
    if not path.exists():
        tmp = path.with_name(path.stem+".tmp"+suffix)
        make(tmp)
        tmp.rename(path)
    return path


def source(rel):
    path = ROOT/rel
    if not path.exists():
        raise FileNotFoundError(f"media source missing: {rel}")
    return path


def webp(src, out, max_width=1800):
    digest = hashlib.sha256(src.read_bytes()).hexdigest()

    def make(path):
        image = Image.open(src).convert("RGB")
        if image.width > max_width:
            image = image.resize((max_width, round(image.height*max_width/image.width)), Image.Resampling.LANCZOS)
        image.save(path, "WEBP", quality=84, method=6)
    shutil.copy2(cached(f"img:{digest}:{max_width}", ".webp", make), out)


def video(src, out, poster, spec):
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    width, crf = spec.get("max_width", 1600), spec.get("crf", 27)
    start, duration = spec.get("start"), spec.get("duration")
    trim = (["-ss", str(start)] if start is not None else [])+(["-t", str(duration)] if duration is not None else [])

    def encode(path):
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", *trim, "-i", str(src), "-an",
                        "-vf", f"scale='min({width},iw)':-2", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    encoded = cached(f"vid:{digest}:{width}:{crf}:{start}:{duration}", ".mp4", encode)
    shutil.copy2(encoded, out)
    if spec.get("poster"):
        webp(source(spec["poster"]), poster)
        return

    def frame(path):
        png = path.with_suffix(".png")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(spec.get("poster_t", 1.0)), "-i", str(encoded),
                        "-frames:v", "1", str(png)], check=True)
        Image.open(png).convert("RGB").save(path, "WEBP", quality=84, method=6)
        png.unlink()
    shutil.copy2(cached(f"poster:{digest}:{width}:{crf}:{start}:{duration}:{spec.get('poster_t', 1.0)}", ".webp", frame),
                 poster)


def build_media(spec, folder):
    media = folder/"media"
    media.mkdir(parents=True, exist_ok=True)
    made = set()
    for key, item in spec.items():
        src = source(item["src"])
        if src.suffix.lower() in (".mp4", ".mov", ".gif"):
            video(src, media/f"{key}.mp4", media/f"{key}_poster.webp", item)
            made |= {f"media/{key}.mp4", f"media/{key}_poster.webp"}
        else:
            webp(src, media/f"{key}.webp", item.get("max_width", 1800))
            made.add(f"media/{key}.webp")
    return made


def tokens(theme):
    missing = [t for t in TOKENS if t not in theme["tokens"]]
    if missing:
        raise KeyError(f"theme tokens missing: {missing}")
    return " ".join(f"--{k}: {theme['tokens'][k]};" for k in TOKENS)


def hero_media(hero):
    key = hero["media"]
    if hero.get("kind", "video") == "image":
        return f'<img src="media/{key}.webp" alt="{html.escape(hero.get("alt", ""))}">'
    return (f'<video src="media/{key}.mp4" poster="media/{key}_poster.webp" controls muted playsinline '
            f'preload="metadata"></video>')


def toc(content):
    out = []
    for sid, title in re.findall(r'<section id="([^"]+)"[^>]*>\s*<h2>(.*?)</h2>', content, re.S):
        label = re.sub(r"<[^>]+>", "", title)
        out.append(f'<a href="#{sid}">{label.replace(" · ", " ")}</a>')
    return "".join(out)


def check_references(page, made):
    refs = set(re.findall(r'(?:src|poster|href)="(media/[^"]+)"', page))
    missing = sorted(refs-made)
    if missing:
        raise RuntimeError(f"media referenced but not built: {missing}")
    unused = sorted(m for m in made-refs if not m.endswith("_poster.webp"))
    if unused:
        print(f"  built but unused: {unused}")


def main():
    spec = json.loads((SRC/"journal.json").read_text())
    content = (SRC/"content.html").read_text()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    made = build_media(spec["media"], OUT)
    css = (SRC/"journal.css").read_bytes()
    (OUT/"journal.css").write_bytes(css)
    page = (SRC/"template.html").read_text()
    metrics = "".join(f'<div class="metric"><div class="value">{m["value"]}</div><div class="label">{m["label"]}</div></div>'
                      for m in spec["metrics"])
    fields = {"title": html.escape(spec["title"]), "description": html.escape(spec["description"]),
              "theme_color": spec["theme"]["theme_color"], "scheme": spec["theme"]["scheme"],
              "css_version": hashlib.sha256(css).hexdigest()[:12], "tokens": tokens(spec["theme"]),
              "eyebrow": spec["eyebrow"], "heading": spec["heading"], "lede": spec["lede"],
              "hero_media": hero_media(spec["hero"]), "hero_caption": spec["hero"]["caption"], "metrics": metrics,
              "toc": toc(content), "scope": spec["scope"], "content": content, "footer": spec["footer"]}
    for k, v in fields.items():
        page = page.replace("{{"+k+"}}", v)
    leftover = re.findall(r"\{\{\w+\}\}", page)
    if leftover:
        raise RuntimeError(f"unfilled template fields {leftover}")
    check_references(page, made)
    (OUT/"index.html").write_text(page)
    (OUT/".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"journal: {len(made)} media files, {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

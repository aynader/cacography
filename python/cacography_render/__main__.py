"""cacography render <jobname>: write by hand every page LuaLaTeX recorded.

Reads <jobname>.cacography/, draws each page whose layout changed since it was last
drawn, and stamps it so the next LuaLaTeX run puts the drawing in the PDF.
"""
import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .aspects import Aspects, parse_options
from . import render3d, surfaces
from .camera import CAMERAS, photograph, studio
from .page import PageWriter
from .paper import PAPERS, make_sheet
from .pen import Pen, subseed
from .symbols import SymbolBank

def _models():
    """Where the handwriting models are: CACOGRAPHY_MODELS, else the source tree's models/
    folder, else models/ in the cache folder (where the separate download unpacks to)."""
    if os.environ.get("CACOGRAPHY_MODELS"):
        return Path(os.environ["CACOGRAPHY_MODELS"])
    here = Path(__file__).resolve().parents[2] / "models"
    return here if (here / "hand").exists() else surfaces.cache_dir() / "models"


MODELS = _models()
RENDER_VERSION = "0.13.2"  # bump when a change to the drawing should redraw existing pages


def find_math_font():
    try:
        out = subprocess.run(["kpsewhich", "latinmodern-math.otf"], capture_output=True, text=True, timeout=60)
        if out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    sys.exit("cacography: cannot find latinmodern-math.otf (is a TeX distribution installed?)")


def main():
    ap = argparse.ArgumentParser(prog="cacography render", description=__doc__)
    ap.add_argument("jobname")
    ap.add_argument("--dpi", type=int, default=200)
    # The same seed gives the same page only on the CPU: GPU arithmetic differs slightly.
    ap.add_argument("--device", default="cpu", help="cpu (reproducible) or cuda (faster, not reproducible)")
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--all", action="store_true", help="redraw pages even when unchanged")
    ap.add_argument("--flat", action="store_true", help="photograph with formulas, not by tracing light in 3D")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    folder = Path(f"{args.jobname}.cacography")
    doc_path = folder / "document.json"
    if not doc_path.exists():
        sys.exit(f"cacography: {doc_path} not found. Run LuaLaTeX on {args.jobname}.tex first.")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    if doc["seed"] == "random":
        seed = secrets.randbelow(2**31)
        print(f"seed=random: using seed {seed} (put seed={seed} in the options to get this page again)")
    else:
        seed = int(doc["seed"]) if doc["seed"].isdigit() else int.from_bytes(hashlib.sha256(doc["seed"].encode()).digest()[:4], "little")
    sloppiness = float(doc.get("sloppiness", 0.5))
    wear = float(doc.get("wear", 0.3))
    paper, camera = doc["paper"], doc.get("camera", "phone")
    if paper not in PAPERS:
        sys.exit(f"cacography: unknown paper={paper} (choose from {', '.join(PAPERS)})")
    if camera not in CAMERAS:
        sys.exit(f"cacography: unknown camera={camera} (choose from {', '.join(CAMERAS)})")

    t0 = time.time()
    if not (args.models / "hand").exists():
        sys.exit(f"cacography: no handwriting models in {args.models}. They are a separate download (non-commercial "
                 "licence); unpack them there, or point CACOGRAPHY_MODELS at them.")
    pen = Pen(args.models, seed, sloppiness, folder / "words.npz", device=args.device)
    print(f"writer: blend of {pen.writers} (seed {seed}), device {args.device}")
    bank = SymbolBank(args.models / "symbols", seed, pen.slant) if (args.models / "symbols" / "bank.npz").exists() else None
    writer = PageWriter(pen, doc, args.dpi, find_math_font(), bank)
    # One room for the whole document: the same desk, lamp and phone on every page.
    try:
        aspects = Aspects(doc, seed)
        background = aspects.value("background")
        if not surfaces.valid(background):
            raise ValueError(f"background={background} (choose from {', '.join(surfaces.names())})")
        if background.startswith("file:"):
            # Your own photo of a surface, found from the document's folder.
            path = Path(background[5:]).expanduser()
            path = path if path.is_absolute() else (folder.parent / path)
            if not path.is_file():
                raise ValueError(f"background={background}: no such file {path}")
            aspects.values["background"] = f"file:{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}"
    except ValueError as e:
        sys.exit(f"cacography: unknown {e}")
    room = studio(aspects)
    traced = camera == "phone" and not args.flat and render3d.available()
    if camera == "phone" and not args.flat and not traced:
        print("note: photographing with formulas; tracing light in 3D needs the mitsuba package and an NVIDIA GPU")
    settings = f"{pen.tag}|{paper}|{doc['ink']}|{camera}|{wear}|{args.dpi}|{aspects.values}|{aspects.rerolls}|{traced}|{RENDER_VERSION}"
    if writer.mess != 1.0:
        # Sloppiness away from 0.5 now loosens the layout too: those pages draw again.
        settings += f"|mess{writer.mess:.3f}"
    drawn = 0
    for n in range(1, int(doc["pages"]) + 1):
        body = (folder / f"page-{n}.json").read_bytes()
        # LuaLaTeX swaps a drawing in only when its stamp matches the page's layout.
        stamp = hashlib.md5(body).hexdigest()
        made_with = hashlib.md5(f"{stamp}|{settings}".encode()).hexdigest()
        record = folder / f"page-{n}.drawn"
        scene = json.loads(body)
        if args.all or not record.exists() or record.read_text().strip() != made_with:
            # A page's own settings (\cacographypage) on top of the document's.
            options = parse_options(scene.get("options", ""))
            try:
                page_aspects = aspects.for_page(options)
                page_wear = float(options.get("wear", wear))
            except ValueError as e:
                sys.exit(f"cacography: page {n}: {e}")
            ink, rules_from = writer.render(scene)
            sheet = make_sheet(paper, scene["width"], scene["height"], args.dpi, rules_from,
                               writer.spacing * writer.px, page_wear, page_aspects, n)
            sheet.erasures, sheet.rubbed = writer.erasures, writer.rubbed
            # One pen for the whole document: the same ink on every page.
            sheet.lay_ink(ink, doc["ink"], np.random.default_rng(subseed(seed, "ink")))
            if traced:
                photo, quality = render3d.photograph(sheet, room, page_aspects, n)
            else:
                photo, quality = photograph(sheet, room, camera, page_aspects, n)
            photo.save(folder / f"page-{n}.jpg", quality=quality, subsampling=0 if camera != "phone" else 2)
            (folder / f"page-{n}.stamp").write_text(stamp)
            record.write_text(made_with)
            drawn += 1
        else:
            # Still count this page's words, so later pages keep their seeds.
            writer.skip(scene)
    pen.save()
    s = pen.stats
    print(f"drew {drawn} of {doc['pages']} pages in {time.time() - t0:.0f}s: {s['words']} words, {s['cached']} from cache,"
          f" {s['tries']} attempts, {s['failed']} never passed the checks")
    if writer.missing:
        print("not handwritten yet (drawn from the font):", " ".join(f"{c}x{n}" for c, n in sorted(writer.missing.items())))
    if not os.environ.get("CACOGRAPHY_BUILDING"):
        print(f"now run LuaLaTeX on {args.jobname}.tex again")


if __name__ == "__main__":
    main()

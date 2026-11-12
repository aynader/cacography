"""Make the release files in dist/:

- cacography-ctan.zip: the LaTeX package for CTAN, laid out as CTAN asks (one top folder).
- cacography-<version>.tar.gz and the wheel: the Python package for PyPI (needs uv or `build`).
- cacography-models.zip: the handwriting models, the separate non-commercial download, from
  models/ (or --models).

    python tools/make_release.py [--no-models] [--models DIR]
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXAMPLES = ["notes.tex", "writing-features.tex", "graph-demo3.tex", "qm-handled.tex"]

MODELS_LICENSE = """cacography handwriting models: terms of use

These files (hand/, checker/, symbols/) are trained on, or taken from, these datasets:

  BRUSH       Brown University; non-commercial research use only
  IAM-OnDB    University of Bern; non-commercial research use only
  MathWriting Google; CC BY-NC-SA 4.0

So they may be used for non-commercial research only, and anything made with them inherits
those terms. They come with no warranty. The cacography code that uses them is under its own
licences (MIT, and LPPL 1.3c for the LaTeX package).
"""


def ctan():
    out = DIST / "cacography-ctan.zip"
    manual = ROOT / "docs" / "cacography-manual.pdf"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("cacography.sty", "cacography.lua", "LICENSE"):
            z.write(ROOT / "latex" / f, f"cacography/tex/{f}")
        z.write(ROOT / "latex" / "README.md", "cacography/README.md")
        z.write(manual, "cacography/doc/cacography-manual.pdf")
        z.write(ROOT / "docs" / "cacography-manual.tex", "cacography/doc/cacography-manual.tex")
        for f in EXAMPLES:
            z.write(ROOT / "examples" / f, f"cacography/doc/examples/{f}")
    print("wrote", out)


def python_package():
    if shutil.which("uv"):
        cmd = ["uv", "build", "--out-dir", str(DIST), str(ROOT)]
    else:
        try:
            import build  # noqa: F401
        except ImportError:
            print("skipped the Python package: install uv, or pip install build")
            return
        cmd = [sys.executable, "-m", "build", "--outdir", str(DIST), str(ROOT)]
    subprocess.run(cmd, check=True)
    for leftover in (ROOT / "build", ROOT / "cacography.egg-info"):
        shutil.rmtree(leftover, ignore_errors=True)


def models(src):
    if not (src / "hand" / "weights.pt").exists():
        print(f"skipped the models: none in {src}")
        return
    out = DIST / "cacography-models.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, "models/" + f.relative_to(src).as_posix())
        z.writestr("models/MODELS-LICENSE", MODELS_LICENSE)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    DIST.mkdir(exist_ok=True)
    ctan()
    python_package()
    if not args.no_models:
        models(args.models)

"""cacography: build handwritten notes from a LaTeX document.

    cacography notes.tex            lay out, write every page by hand, put the pages in
    cacography render notes         only write the pages (after a LuaLaTeX run)
    cacography setup                put the LaTeX package where TeX finds it, fetch the models
    cacography doctor               say what is missing, and how to fix it

Anything after the file name goes to the renderer, e.g. `cacography notes.tex --device cuda`.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from importlib import metadata, resources
from pathlib import Path

# The handwriting models: a separate, non-commercial download, published with each release.
MODELS_URL = os.environ.get(
    "CACOGRAPHY_MODELS_URL",
    "https://github.com/aynader/cacography/releases/latest/download/cacography-models.zip")


def version():
    try:
        return metadata.version("cacography")
    except metadata.PackageNotFoundError:
        return "source"


def _say(ok, what, fix=""):
    mark = "ok " if ok else "-- "
    print(f"  {mark}{what}" + (f"\n       {fix}" if fix and not ok else ""))
    return ok


def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _kpse(*args):
    out = _run(["kpsewhich", *args])
    return out.stdout.strip() if out and out.returncode == 0 else ""


def _models_dir():
    from .__main__ import MODELS
    return MODELS


def _shipped_tex():
    """The LaTeX files that came with this Python package (None when run from a source tree)."""
    try:
        here = resources.files("cacography_render") / "tex"
        if (here / "cacography.sty").is_file():
            return Path(str(here))
    except (ModuleNotFoundError, TypeError):
        pass
    # A source tree: latex/ in the repository, tex/ in the workshop it is made from.
    for tree in (Path(__file__).resolve().parents[2] / d for d in ("latex", "tex")):
        if (tree / "cacography.sty").is_file():
            return tree
    return None


def _sty_version(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("\\ProvidesPackage{cacography}"):
            return line.split(" v", 1)[1].split()[0] if " v" in line else ""
    return ""


def _miktex():
    return shutil.which("initexmf") is not None


def _refresh(home):
    """Make TeX see new files under TEXMFHOME."""
    if _miktex():
        report = _run(["initexmf", "--report"])
        roots = report.stdout.lower() if report else ""
        if str(home).lower().replace("/", "\\") not in roots.replace("/", "\\"):
            _run(["initexmf", f"--register-root={home}"])
        _run(["initexmf", "--update-fndb"])
    elif shutil.which("mktexlsr"):
        _run(["mktexlsr", str(home)])


# -- setup -------------------------------------------------------------------

def setup(args):
    print(f"cacography {version()}: setting up")
    ok = True

    # 1. The LaTeX package, unless TeX already has it (from CTAN, or a link to a source tree).
    home = _kpse("-var-value", "TEXMFHOME")
    if not home:
        print("  -- no TeX distribution found (kpsewhich is missing). Install TeX Live or MiKTeX first.")
        return 1
    home = Path(home)
    target = home / "tex" / "latex" / "cacography"
    found = _kpse("cacography.sty")
    shipped = _shipped_tex()
    if target.is_symlink() or _is_junction(target):
        print(f"  ok LaTeX package: {target} links to a source tree; left alone")
    elif found and not args.force and Path(found).resolve().parent != target.resolve() \
            and _sty_version(found) >= _sty_version(shipped / "cacography.sty"):
        print(f"  ok LaTeX package: TeX already has v{_sty_version(found)} ({found})")
    elif shipped is None:
        print("  -- this install carries no LaTeX files; get cacography from CTAN (MiKTeX / TeX Live)")
        ok = False
    else:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("cacography.sty", "cacography.lua"):
            shutil.copyfile(shipped / name, target / name)
        print(f"  ok LaTeX package: copied v{_sty_version(target / 'cacography.sty')} to {target}")

    # 2. How LuaLaTeX starts the renderer when it may (--shell-escape): this very Python.
    cfg_dir = home / "tex" / "latex" / "cacography-config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    exe = sys.executable
    command = (f'"{exe}"' if " " in exe else exe) + " -m cacography_render.cli"
    (cfg_dir / "cacography-config.lua").write_text(
        "-- Written by `cacography setup`: how LuaLaTeX starts the renderer.\n"
        f"return {{ command = [[{command}]] }}\n", encoding="utf-8")
    print(f"  ok renderer command for LuaLaTeX: {cfg_dir / 'cacography-config.lua'}")
    _refresh(home)

    # 3. The handwriting models.
    models = _models_dir()
    if (models / "hand" / "weights.pt").exists() and not args.models:
        print(f"  ok handwriting models: {models}")
    else:
        ok &= _fetch_models(args.models or MODELS_URL)

    print()
    return doctor(args) if ok else 1


def _is_junction(path):
    """A Windows junction (what `mklink /J` makes): TeX follows it like a link."""
    try:
        return bool(os.lstat(path).st_file_attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except (OSError, AttributeError):
        return False


def _fetch_models(source):
    from .surfaces import cache_dir
    dest = cache_dir() / "models"
    src = Path(source) if not str(source).startswith(("http://", "https://")) else None
    try:
        if src is not None and src.is_dir():
            inner = src / "models" if (src / "models" / "hand").exists() else src
            shutil.copytree(inner, dest, dirs_exist_ok=True)
        else:
            if src is None:
                print(f"     downloading the handwriting models from {source} ...")
                tmp = Path(tempfile.mkdtemp()) / "models.zip"
                req = urllib.request.Request(source, headers={"User-Agent": "cacography-setup"})
                with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                src = tmp
            with zipfile.ZipFile(src) as z:
                for info in z.infolist():
                    name = Path(info.filename)
                    parts = name.parts[1:] if name.parts and name.parts[0] == "models" else name.parts
                    if not parts or info.is_dir() or ".." in parts:
                        continue
                    out = dest.joinpath(*parts)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(info))
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        return _say(False, f"handwriting models: could not get them from {source} ({e})",
                    "download cacography-models.zip from the release page, then: cacography setup --models <zip>")
    ok = (dest / "hand" / "weights.pt").exists()
    return _say(ok, f"handwriting models: {dest}", "the archive held no hand/weights.pt")


# -- doctor ------------------------------------------------------------------

def doctor(args=None):
    print(f"cacography {version()}: checking")
    good = True
    lualatex = shutil.which("lualatex")
    good &= _say(bool(lualatex), f"LuaLaTeX: {lualatex or 'not found'}",
                 "install TeX Live (tug.org/texlive) or MiKTeX (miktex.org)")
    if lualatex:
        sty = _kpse("cacography.sty")
        good &= _say(bool(sty), f"LaTeX package: {sty or 'not found'}" + (f" (v{_sty_version(sty)})" if sty else ""),
                     "run: cacography setup")
        cfg = _kpse("cacography-config.lua")
        _say(bool(cfg), "LuaLaTeX can start the renderer itself" + (f" ({cfg})" if cfg else ""),
             "optional, for compiling in an editor: cacography setup")
        font = _kpse("latinmodern-math.otf")
        good &= _say(bool(font), "font latinmodern-math.otf", "install the lm-math package of your TeX distribution")
    models = _models_dir()
    good &= _say((models / "hand" / "weights.pt").exists(), f"handwriting models: {models}",
                 "run: cacography setup   (or set CACOGRAPHY_MODELS to the folder)")
    import torch
    cuda = torch.cuda.is_available()
    _say(True, f"PyTorch {torch.__version__}" + (f", GPU {torch.cuda.get_device_name(0)}" if cuda else ", CPU only"))
    from . import render3d
    traced = render3d.available()
    _say(traced, "3D photos (Mitsuba 3 on an NVIDIA GPU)" if traced else
         "3D photos: not available, pages are photographed with formulas (no props, simpler light)",
         "needs an NVIDIA GPU and driver; everything else works without one")
    from .surfaces import cache_dir
    _say(True, f"cache (textures, models): {cache_dir()}")
    print()
    if good:
        print("Ready. Try:  cacography notes.tex")
        print("In an editor: compile with LuaLaTeX and --shell-escape (twice, like references).")
    return 0 if good else 1


# -- build -------------------------------------------------------------------

def _lualatex(job):
    cmd = ["lualatex", "-interaction=nonstopmode", "-synctex=1", job]
    print("$", " ".join(cmd))
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in out.stdout.splitlines():
        # "not drawn yet" is what this command is about to do, so it isn't news here.
        if line.startswith("!") or "cacography" in line and ("Warning" in line or "Info" in line) \
                and "not drawn yet" not in line:
            print("  " + line)
    return out.returncode


def build(tex, rest):
    path = Path(tex)
    if path.suffix != ".tex":
        path = path.with_suffix(".tex")
    if not path.exists():
        print(f"cacography: {path} not found")
        return 1
    os.chdir(path.parent.resolve())
    job = path.stem
    if _lualatex(job) != 0 and not Path(f"{job}.cacography/document.json").exists():
        print(f"cacography: LuaLaTeX stopped with errors, see {job}.log")
        return 1
    from .__main__ import main as render_main
    os.environ["CACOGRAPHY_BUILDING"] = "1"  # the renderer needn't say to run LuaLaTeX again
    sys.argv = ["cacography render", job, *rest]
    try:
        render_main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"cacography: the renderer stopped ({e.code})")
            return 1
    code = _lualatex(job)
    print(f"cacography: {job}.pdf" if code == 0 else f"cacography: LuaLaTeX stopped with errors, see {job}.log")
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"cacography {version()}")
        return 0
    cmd = argv[0]
    if cmd == "render":
        from .__main__ import main as render_main
        sys.argv = ["cacography render", *argv[1:]]
        return render_main()
    if cmd in ("setup", "doctor"):
        import argparse
        ap = argparse.ArgumentParser(prog=f"cacography {cmd}")
        if cmd == "setup":
            ap.add_argument("--models", help="a cacography-models.zip or folder to install instead of downloading")
            ap.add_argument("--force", action="store_true", help="copy the LaTeX files even if TeX already has them")
        args = ap.parse_args(argv[1:])
        return setup(args) if cmd == "setup" else doctor(args)
    if cmd == "build":
        argv = argv[1:]
        if not argv:
            print("usage: cacography build notes.tex")
            return 1
    return build(argv[0], argv[1:])


if __name__ == "__main__":
    sys.exit(main())

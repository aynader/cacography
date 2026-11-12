"""Make the README's pictures in docs/img/: the showcase, and one row per option.

Every row writes the same short page with the same seed and changes one thing, so what
the option does is all that differs. Pages are drawn one at a time (one GPU job).

    python tools/make_showcase.py                 # everything still missing
    python tools/make_showcase.py --only angle    # one row again
    python tools/make_showcase.py --list
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docs" / "showcase-build"
IMG = ROOT / "docs" / "img"
DEMOS = ROOT / "examples" / "showcase"

# Everything pinned, so a row changes only what it names.
BASE = dict(
    seed="11", paper="notebook", ink="blue", sloppiness="0.5", camera="phone", wear="0.3",
    background="oak", light="lamp", angle="slight", framing="tight", shadow="none",
    crumple="none", folds="none", creases="none", scores="none", edges="clean", stains="none",
    grooves="none", backside="none", curl="none", age="none", props="none", underline="none",
)
PAPER_ASPECTS = ["crumple", "folds", "creases", "scores", "edges", "stains", "grooves", "backside",
                 "curl", "age", "props"]

PAGE = r"""\documentclass{article}
\usepackage[%s]{cacography}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\pagestyle{empty}
\begin{document}
\section*{The harmonic oscillator}
A mass on a spring feels $F = -kx$, so
\[ x'' = -\omega^2 x, \qquad \omega^2 = k/m. \]
Its energy \crossout{decays} stays put: $E = k A^2 / 2$.
\begin{itemize}
\item the period does not depend on the amplitude
\item for a pendulum: small angles only!
\end{itemize}
Ask if this is on the exam (it is).
\section*{With friction}
Now $x(t) = A e^{-\gamma t} \cos \omega t$, and the swing dies away:
\begin{center}
\begin{tikzpicture}
\begin{axis}[width=6.4cm,height=3.8cm,xlabel={$t$},ylabel={$x$},xmin=0,xmax=6]
\addplot[domain=0:6,samples=60]{exp(-x/3)*cos(deg(2*x))};
\end{axis}
\end{tikzpicture}
\end{center}
\end{document}
"""

# (name, values, how): "photo" shows the whole picture, "scan" the writing up close,
# "zoom" part of the photo up close, "page" the whole sheet scanned.
ROWS = [
    ("seed", ["3", "11", "42"], "scan"),
    ("sloppiness", ["0", "0.5", "1"], "close"),
    ("ink", ["blue", "red", "pencil"], "scan"),
    ("paper", ["notebook", "graph", "legal"], "scan"),
    ("underline", ["none", "headings"], "scan"),
    ("camera", ["phone", "scan", "none"], "photo"),
    ("background", ["oak", "marble", "plaid"], "photo"),
    ("light", ["lamp", "raking", "flash", "dim"], "photo"),
    ("angle", ["flat", "slight", "steep"], "photo"),
    ("framing", ["tight", "loose", "cropped"], "photo"),
    ("shadow", ["none", "phone"], "photo"),
    ("props", ["binder", "{pen, clip}", "{sheet, tape}"], "photo"),
    ("crumple", ["none", "0.5", "1"], "photo"),
    ("folds", ["half", "thirds", "quarters"], "photo"),
    ("creases", ["3", "12"], "photo"),
    ("edges", ["worn", "torn", "spiral"], "photo"),
    ("stains", ["coffee", "water"], "page"),
    ("curl", ["none", "0.5", "1"], "photo"),
    ("age", ["none", "0.5", "1"], "photo"),
    ("backside", ["none", "writing"], "zoom"),
    ("wear", ["0", "0.5", "1"], "photo"),
]
# Shown in a light that brings them out.
EXTRA = {k: dict(light="raking") for k in ("grooves", "creases", "folds", "crumple")}
EXTRA["backside"] = dict(light="window")
EXTRA["props"] = dict(framing="loose")

# The commands, each written on its own: (name, LaTeX, ink).
COMMANDS = [
    ("crossout", r"\crossout[line]{wrong} right, \crossout[double]{worse} better, \crossout[scribble]{oops} fine", "blue"),
    ("erased", r"Friction \erased{helps}{opposes} the motion.", "pencil"),
    ("note", r"\cacographynote*{check!}The integral converges for $s > 1$.", "blue"),
    ("boxed", r"So \boxed{E = mc^2} and \underline{nothing else}.", "blue"),
    ("tikz", r"""\begin{tikzpicture}
\begin{axis}[width=6cm, height=4cm,
    xlabel={$t$}, ylabel={$x$}]
\addplot[domain=0:6, samples=60]
    {exp(-x/3)*cos(deg(2*x))};
\end{axis}
\end{tikzpicture}""", "blue"),
]
REROLL = ["Mississippi", r"\reroll{Mississippi}", r"\reroll[3]{Mississippi}"]


def font(size):
    for name in ("lmmono10-regular.otf", "lmsans10-regular.otf"):
        path = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
        if path:
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def options(**over):
    o = dict(BASE, **over)
    return ", ".join(f"{k}={v}" for k, v in o.items())


def draw(job, tex):
    """Write the page and draw it: returns the photo. Skips what is already drawn."""
    folder = BUILD / f"{job}.cacography"
    photo = folder / "page-1.jpg"
    src = BUILD / f"{job}.tex"
    if photo.exists() and src.exists() and src.read_text(encoding="utf-8") == tex:
        return Image.open(photo).convert("RGB")
    BUILD.mkdir(parents=True, exist_ok=True)
    src.write_text(tex, encoding="utf-8")
    env = dict(os.environ, TEXINPUTS=f".{os.pathsep}{ROOT / 'latex'}{os.pathsep}")
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    run = subprocess.run(["lualatex", "-interaction=nonstopmode", job], cwd=BUILD, env=env,
                         capture_output=True, text=True, errors="replace")
    if not (folder / "document.json").exists():
        sys.exit(f"{job}: LuaLaTeX failed\n{run.stdout[-2000:]}")
    r = subprocess.run([sys.executable, "-m", "cacography_render.cli", "render", job, "--device", "cuda", "--all"],
                       cwd=BUILD, env=env, capture_output=True, text=True, errors="replace")
    if r.returncode or not photo.exists():
        sys.exit(f"{job}: renderer failed\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(f"  {job}: {r.stdout.strip().splitlines()[-2] if r.stdout.strip() else ''}")
    return Image.open(photo).convert("RGB")


def ink_box(im, pad=40):
    """The written part of a blank page: where it is darker than the paper."""
    g = ImageOps.grayscale(im)
    mask = g.point(lambda v: 255 if v < 150 else 0)
    box = mask.getbbox() or (0, 0, im.width, im.height)
    return (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))


def tile_row(tiles, captions, width=1200, gap=14):
    n = len(tiles)
    w = (width - gap * (n - 1)) // n
    tiles = [t.resize((w, round(t.height * w / t.width)), Image.LANCZOS) for t in tiles]
    h = max(t.height for t in tiles)
    f = font(26)
    row = Image.new("RGB", (width, h + 46), "white")
    d = ImageDraw.Draw(row)
    for i, (t, c) in enumerate(zip(tiles, captions)):
        x = i * (w + gap)
        row.paste(t, (x, 0))
        d.text((x + w // 2, h + 10), c, fill=(40, 40, 40), font=f, anchor="ma")
    return row


def save(im, name):
    IMG.mkdir(parents=True, exist_ok=True)
    im.save(IMG / name, quality=82, optimize=True, progressive=True)
    print(f"wrote docs/img/{name} ({(IMG / name).stat().st_size // 1024} KB)")


def option_row(name, values, how):
    tiles = []
    for v in values:
        over = dict(EXTRA.get(name, {}))
        over[name] = v
        if how in ("scan", "page", "close"):
            over["camera"] = "scan"
        if name == "wear":
            over.update({a: "random" for a in PAPER_ASPECTS})
        job = f"{name}-{v}".replace(" ", "").replace("{", "").replace("}", "").replace(",", "+").replace(".", "_")
        im = draw(job, PAGE % options(**over))
        if how == "scan":
            im = im.crop((int(im.width * 0.1), int(im.height * 0.07), int(im.width * 0.95), int(im.height * 0.68)))
        elif how == "close":  # the first few lines, big enough to see each word
            im = im.crop((int(im.width * 0.12), int(im.height * 0.08), int(im.width * 0.72), int(im.height * 0.315)))
        elif how == "blank":  # the empty lower half, where pressed-in writing shows
            im = im.crop((int(im.width * 0.1), int(im.height * 0.5), int(im.width * 0.85), int(im.height * 0.85)))
        elif how == "zoom":
            im = im.crop((int(im.width * 0.04), int(im.height * 0.02), int(im.width * 0.7), int(im.height * 0.4)))
        tiles.append(im)
    width = 1500 if how in ("scan", "close") and len(values) >= 3 or len(values) > 3 else 1200
    save(tile_row(tiles, [f"{name}={v}" for v in values], width), f"option-{name}.jpg")


def command_rows():
    for name, body, ink in COMMANDS:
        tex = (r"\documentclass{article}" "\n"
               rf"\usepackage[{options(paper='blank', camera='scan', ink=ink)}]{{cacography}}" "\n"
               r"\usepackage{pgfplots}\pgfplotsset{compat=1.17}\pagestyle{empty}" "\n"
               rf"\begin{{document}}{body}\end{{document}}" "\n")
        im = draw(f"cmd-{name}", tex)
        im = im.crop(ink_box(im))
        save(code_beside(body, im), f"command-{name}.jpg")
    tiles = []
    for i, body in enumerate(REROLL):
        tex = (r"\documentclass{article}" "\n"
               rf"\usepackage[{options(paper='blank', camera='scan')}]{{cacography}}\pagestyle{{empty}}" "\n"
               rf"\begin{{document}}{body}\end{{document}}" "\n")
        im = draw(f"cmd-reroll{i}", tex)
        tiles.append(im.crop(ink_box(im, 30)))
    save(tile_row(tiles, REROLL, 1200), "command-reroll.jpg")


def code_beside(code, im, width=1400, code_w=640):
    """The LaTeX on the left, what it wrote on the right."""
    f = font(24)
    lines = []
    for raw in code.splitlines():
        limit = (code_w - 40) // 15
        while len(raw) > limit:
            cut = max(raw.rfind(" ", 0, limit), raw.rfind(",", 0, limit) + 1)
            cut = cut if cut > 10 else limit
            lines.append(raw[:cut].rstrip())
            raw = "    " + raw[cut:].lstrip()
        lines.append(raw)
    pic_w = width - code_w - 20
    im = im.resize((pic_w, round(im.height * pic_w / im.width)), Image.LANCZOS)
    h = max(im.height, 34 * len(lines) + 40)
    out = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, code_w, h), fill=(246, 246, 243))
    for i, line in enumerate(lines):
        d.text((22, 20 + 34 * i), line, fill=(30, 30, 30), font=f)
    out.paste(im, (code_w + 20, (h - im.height) // 2))
    return out


# The four pages in the grid at the top, and the one shown beside its source.
HERO = ["d02-analysis-pruefung", "d31-cell-counts", "d43-monte-carlo-plan", "d44-klausur-letzte-minute"]
SOURCE = "d35-trip-budget"


def photo_of(name):
    """A showcase page's photo: as built in the workshop's demos, else drawn here."""
    built = ROOT.parent / "examples" / "demos2" / f"{name}.cacography" / "page-1.jpg"
    if built.exists():
        return Image.open(built).convert("RGB")
    return draw(name, (DEMOS / f"{name}.tex").read_text(encoding="utf-8"))


def showcase():
    photos = [photo_of(n) for n in HERO]
    # Two by two: a README is shown about 880 pixels wide, too narrow for four in a row.
    rows = [tile_row(photos[i:i + 2], ["", ""], 1400, 12) for i in (0, 2)]
    rows = [r.crop((0, 0, r.width, r.height - 46)) for r in rows]
    grid = Image.new("RGB", (1400, rows[0].height + rows[1].height + 12), "white")
    grid.paste(rows[0], (0, 0))
    grid.paste(rows[1], (0, rows[0].height + 12))
    save(grid, "showcase.jpg")
    text = (DEMOS / f"{SOURCE}.tex").read_text(encoding="utf-8")
    src = text.split(r"\begin{document}")[1].split(r"\end{document}")[0].strip()
    usep = [l for l in text.splitlines() if "cacography}" in l][0]
    code = usep.replace(", ", ",").replace(",", ", ") + "\n\n" + src
    save(code_beside(code, photo_of(SOURCE), 1600, 860), "source-to-photo.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="rows to make: option names, 'commands', 'showcase'")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(" ".join([r[0] for r in ROWS] + ["commands", "showcase"]))
        return
    want = set(args.only or [])
    if not want or "showcase" in want:
        showcase()
    for name, values, how in ROWS:
        if not want or name in want:
            option_row(name, values, how)
    if not want or "commands" in want:
        command_rows()


if __name__ == "__main__":
    main()

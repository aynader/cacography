"""Turn one page's scene into handwriting on paper.

TeX decides which words share a line and where math sits. The hand decides the rest:
words keep their natural widths and gaps, so lines end raggedly, squeeze when they run
out of room and spill a little into the margin. Each line snaps to the nearest ruled
line and gets its own small slope; each word its own wobble in size, height and angle.
Math keeps TeX's arrangement, written glyph by glyph.
"""
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from . import drawing
from .ink import Canvas, pen_of
from .marks import XH as MODEL_XH
from .marks import classify
from .model import CHAR_TO_ID, SCALE, XH
from .pen import subseed
from .symbols import label_of

PT_PER_MM = 72.27 / 25.4
PEN_MM = 0.42
TYPOGRAPHIC = {
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
    "∗": "*", "∕": "/", "∶": ":", "×": "x", "′": "'", " ": " ",
}
SPELLED = {"ß": "ss", "ẞ": "SS"}
SYMBOLS = {"-", "=", "+", ".", ","}
PRIMES = {"′": 1, "″": 2, "‴": 3, "'": 1}  # primes, one to three, drawn as short ticks
# Arrows a hand draws in two or three quick strokes: (heads at the right, at the left, shafts, bar at the tail)
ARROWS = {"→": (1, 0, 1, 0), "⟶": (1, 0, 1, 0), "←": (0, 1, 1, 0), "⟵": (0, 1, 1, 0), "↔": (1, 1, 1, 0), "⟷": (1, 1, 1, 0),
          "⇒": (1, 0, 2, 0), "⟹": (1, 0, 2, 0), "⇐": (0, 1, 2, 0), "⟸": (0, 1, 2, 0), "⇔": (1, 1, 2, 0), "⟺": (1, 1, 2, 0),
          "↦": (1, 0, 1, 1), "⟼": (1, 0, 1, 1)}
# Arrows pointing elsewhere: the arrow drawn pointing right, then turned (degrees, counter-clockwise).
TURNED_ARROWS = {"↑": ("→", 90), "↓": ("→", -90), "↕": ("↔", 90), "⇑": ("⇒", 90), "⇓": ("⇒", -90), "⇕": ("⇔", 90),
                 "↗": ("→", 45), "↘": ("→", -45), "↖": ("→", 135), "↙": ("→", -135)}
BULLETS = "•·"
MATH_AIR = 0.3  # least gap between neighbouring math glyphs' ink, in x-heights
MATH_JOIN = 1.5  # math closer than this, in points, was set with no space between
NUDGE_MOST, NUDGE_ROOM = 1.2, 0.55  # in x-heights: furthest a word slides, and the least gap it leaves
NUDGES = [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0, 1.2, -1.2]
NUMBER = re.compile(r"[(\[]?\d[\d.,:]*[)\]]?[.,:]?")
MARKS = {"̈": "diaeresis", "́": "acute", "̀": "grave", "̂": "circumflex", "̃": "tilde"}


def spell(text):
    """Split text into what the model writes and accents drawn on top.

    Returns (letters, [(letter index, accent)], [characters the model cannot write]).
    """
    letters, marks, missing = [], [], []
    for ch in unicodedata.normalize("NFKC", text):
        ch = SPELLED.get(TYPOGRAPHIC.get(ch, ch), TYPOGRAPHIC.get(ch, ch))
        for c in unicodedata.normalize("NFD", ch):
            if c in MARKS and letters:
                marks.append((len(letters) - 1, MARKS[c]))
            elif unicodedata.combining(c):
                continue
            elif c in CHAR_TO_ID and c != " ":
                letters.append(c)
            elif not c.isspace():
                missing.append(c)
    return "".join(letters), marks, missing


def is_vector(ch):
    """Bold math letters are vectors; by hand they get an arrow on top."""
    try:
        return "MATHEMATICAL BOLD" in unicodedata.name(ch)
    except ValueError:
        return False


@dataclass
class Unit:
    glyphs: list
    math: bool
    x0: float
    x1: float
    text: str = ""
    marks: list = field(default_factory=list)
    key: str = ""
    fig: bool = False  # a label inside a drawing: it stays where TeX put it
    cross: int = 0  # \crossout: 1 the writer's habit, 2 one line, 3 two, 4 a scribble
    erased: str = ""  # \erased{old}{new}: what was written here first, and rubbed out
    old_key: str = ""
    old_shape: list = None
    # filled in by layout
    x: float = 0.0
    width: float = 0.0


def page_units(scene):
    """Group the page's glyphs into lines of units, in reading order."""
    lines = {l["id"]: {"box": l, "units": []} for l in scene["lines"]}
    erased = {str(k): v for k, v in (scene.get("erased") or {}).items()}
    current = {}
    for g in scene["glyphs"]:
        if g["line"] not in lines or not g["t"]:
            continue
        key = (g["line"], g["word"])
        if key not in current:
            current[key] = Unit([], False, g["x"], g["x"] + g["w"], fig=bool(g.get("fig")))
            lines[g["line"]]["units"].append(current[key])
        u = current[key]
        u.glyphs.append(g)
        u.cross = max(u.cross, int(g.get("cross") or 0))
        if g.get("erased") and not u.erased:
            u.erased = erased.pop(str(g["erased"]), "")  # all of it under the first word of the group
        # A degree sign or an arrow is drawn like a symbol, so its unit is written as math.
        u.math = u.math or g["math"] or g["t"] == "°" or g["t"] in ARROWS or g["t"] in TURNED_ARROWS or g["t"] in "+=<>"
        u.x0, u.x1 = min(u.x0, g["x"]), max(u.x1, g["x"] + g["w"])
    turned = []
    for line in lines.values():
        for u in line["units"]:
            # Section, equation and page numbers are written glyph by glyph, like math.
            u.math = u.math or bool(NUMBER.fullmatch("".join(g["t"] for g in u.glyphs)))
        turned += _turned(line)
        line["units"].sort(key=lambda u: u.x0)
        line["units"] = _join_math(line["units"])
    for line in turned:
        line["units"].sort(key=lambda u: u.x0)
        line["units"] = _join_math(line["units"])
    return [l for l in list(lines.values()) + turned if l["units"]]


def _turned(line):
    """Take a drawing's rotated labels out of `line` into lines of their own.

    Each is turned back level about its first glyph, written like any line, and drawn
    turned again (see Turned). Returns the new lines.
    """
    groups = {}
    for u in list(line["units"]):
        rot = u.glyphs[0].get("rot")
        if rot:
            groups.setdefault(round(rot, 2), []).append(u)
            line["units"].remove(u)
    out = []
    for k, (rot, units) in enumerate(groups.items()):
        # The first glyph in reading order: furthest back along the turned baseline.
        first = min((g for u in units for g in u.glyphs), key=lambda g: g["x"] * np.cos(rot) - g["y"] * np.sin(rot))
        px, py = first["x"], first["y"]
        # PDF turns counter-clockwise with y up; on the page y runs down.
        c, s_ = np.cos(rot), np.sin(rot)
        for u in units:
            for g in u.glyphs:
                dx, dy = g["x"] - px, g["y"] - py
                g["x"], g["y"] = px + c * dx - s_ * dy, py + s_ * dx + c * dy
            u.x0 = min(g["x"] for g in u.glyphs)
            u.x1 = max(g["x"] + g["w"] for g in u.glyphs)
            u.fig = False  # written as a line of its own, then turned
        x0, x1 = min(u.x0 for u in units), max(u.x1 for u in units)
        box = dict(line["box"], id=f'{line["box"]["id"]}~{k}', x=x0, w=3 * (x1 - x0) + 20, y=py)
        out.append({"box": box, "units": units, "parent": line["box"]["id"], "pivot": (px, py), "rot": rot})
    return out


class Turned:
    """A canvas whose strokes are turned `rot` (counter-clockwise, as PDF counts) about `pivot` (px)."""

    def __init__(self, canvas, pivot, rot):
        self.canvas, self.pivot = canvas, np.asarray(pivot, float)
        c, s = np.cos(rot), np.sin(rot)
        self.m = np.array([[c, s], [-s, c]])

    def stroke(self, points, width, *args, **kwargs):
        p = np.asarray(points, float)
        self.canvas.stroke((p - self.pivot) @ self.m.T + self.pivot, width, *args, **kwargs)

    def glyph(self, *args, **kwargs):
        self.canvas.glyph(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.canvas, name)


def _join_math(units):
    """One unit for math TeX set together.

    TeX centres a fraction's rows with glue, which splits them into separate units though
    they overlap on the page. Laid out as one, they can be spaced and centred as one.
    """
    out = []
    for u in units:
        if u.fig or (out and out[-1].fig):
            # A drawing's labels are each their own: only overlapping ones are one label.
            if out and u.fig and out[-1].fig and u.math == out[-1].math and u.x0 < out[-1].x1 - 0.5 and \
                    abs(u.glyphs[0]["y"] - out[-1].glyphs[0]["y"]) < 0.5 * u.glyphs[0]["size"]:
                out[-1].glyphs += u.glyphs
                out[-1].x1 = max(out[-1].x1, u.x1)
            else:
                out.append(u)
            continue
        if out and u.math and out[-1].math and u.x0 < out[-1].x1 + MATH_JOIN:
            out[-1].glyphs += u.glyphs
            out[-1].x1 = max(out[-1].x1, u.x1)
            out[-1].cross = max(out[-1].cross, u.cross)
        else:
            out.append(u)
    return out


def strokes_of(steps):
    """Split model steps into strokes of working-pixel points."""
    xy = np.cumsum(steps[:, :2], 0) * SCALE
    out, start = [], 0
    for t in np.flatnonzero(steps[:, 2] > 0.5):
        out.append(xy[start : t + 1])
        start = t + 1
    if start < len(xy):
        out.append(xy[start:])
    return [s for s in out if len(s)]


def mm_to_pt(text):
    m = re.fullmatch(r"\s*([\d.]+)\s*(mm|pt|cm|in)\s*", text)
    value, unit = float(m.group(1)), m.group(2)
    return value * {"mm": PT_PER_MM, "cm": 10 * PT_PER_MM, "in": 72.27, "pt": 1.0}[unit]


class Inked:
    """Where earlier lines left ink, on a grid about a pen's width across."""

    def __init__(self, width, height, cell):
        self.cell = cell
        self.grid = np.zeros((int(height / cell) + 2, int(width / cell) + 2), bool)

    def _cells(self, points, shift=0.0):
        ij = np.floor((points + [shift, 0.0]) / self.cell).astype(int)
        keep = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.grid.shape[1]) & (ij[:, 1] < self.grid.shape[0])
        return ij[keep]

    def add(self, points):
        ij = self._cells(points)
        self.grid[ij[:, 1], ij[:, 0]] = True

    def hits(self, points, shift):
        ij = self._cells(points, shift)
        return int(self.grid[ij[:, 1], ij[:, 0]].sum())


def mess(sloppiness):
    """How far the page itself wanders, from `sloppiness`: each line's slant, each word's
    size, height and tilt, the gaps between words. Exactly 1 at the default 0.5, so default
    pages don't change; a tidy 0.2 at 0, and 1.8 times as loose at 1."""
    s = float(np.clip(float(sloppiness), 0.0, 1.0))
    return 0.2 + 1.6 * s if s <= 0.5 else 1.0 + 1.6 * (s - 0.5)


class PageWriter:
    def __init__(self, pen, doc, dpi, math_font, bank=None):
        self.pen = pen
        self.bank = bank
        self.doc = doc
        self.dpi = dpi
        self.px = dpi / 72.27
        self.math_font = math_font
        self.mess = mess(doc.get("sloppiness", 0.5))
        self.seen = {}
        self.spacing = mm_to_pt(doc["rulespacing"])
        self.symbol_size = float(np.random.default_rng(subseed(pen.seed, "symbol size")).uniform(0.85, 1.2))
        self.missing = {}
        # How this writer dots umlauts, from real i-dots in IAM-OnDB (handlab/dot_stats.py): a
        # tick about a tenth of an x-height long, 0.3 to 0.9 x-heights over the letter, with
        # each writer's own height, size and lean. About one in five writes a dash instead.
        habit = np.random.default_rng(subseed(pen.seed, "umlaut habit"))
        self.umlaut = dict(
            above=float(np.clip(habit.normal(0.6, 0.2), 0.3, 1.0)),
            length=float(np.clip(np.exp(habit.normal(np.log(0.11), 0.4)), 0.04, 0.28)),
            angle=float(habit.choice([-20.0, 30.0, 60.0, -60.0]) + habit.normal(0, 10)),
            spread=float(habit.uniform(0.45, 0.8)),
            shift=float(habit.normal(0.05, 0.05)),
            dash=bool(habit.random() < 0.2),
        )
        # How hard this writer presses, which sets how dark their ink comes out.
        self.pressure = float(np.clip(np.random.default_rng(subseed(pen.seed, "pressure")).normal(1.0, 0.1), 0.75, 1.2))

    # Keys -----------------------------------------------------------------

    def _key(self, kind, text, variant=None):
        """The word's identity, which seeds how it is written: the same text's nth use.

        A variant (from \\reroll) changes the key, so only that word is written anew.
        """
        n = self.seen.get((kind, text), 0)
        self.seen[(kind, text)] = n + 1
        return f"{kind}:{text}#{n}" + (f"~{variant}" if variant else "")

    def plan(self, lines):
        """Decide what the pen must write on this page, in reading order."""
        requests = []
        for line in lines:
            for u in line["units"]:
                if u.math:
                    for g in u.glyphs:
                        if g["t"] in ARROWS or g["t"] in TURNED_ARROWS or g["t"] == "°" or g["t"] in PRIMES:
                            g["symbol"], g["letters"] = g["t"], ""
                            continue
                        label = label_of(g["t"]) if self.bank else None
                        if label and self.bank.has(label):
                            g["bank"], g["bank_key"] = label, self._key("bank", label, g.get("v"))
                            continue
                        letters, _, missing = spell(g["t"])
                        g["letters"], g["missing"] = letters, missing
                        if letters in SYMBOLS:
                            # Too simple for the checker to verify alone: drawn as pen strokes.
                            g["symbol"], g["letters"] = letters, ""
                        elif letters:
                            g["key"] = self._key("math", letters, g.get("v"))
                            requests.append((g["key"], letters))
                else:
                    raw = "".join(g["t"] for g in u.glyphs)
                    u.text, u.marks, missing = spell(raw)
                    for m in missing:
                        if m not in BULLETS:
                            self.missing[m] = self.missing.get(m, 0) + 1
                    if u.text:
                        u.key = self._key("text", u.text, max((g.get("v") or 0 for g in u.glyphs), default=0))
                        requests.append((u.key, u.text))
                    if u.text and not u.erased and not u.cross and not u.fig:
                        u.erased = self._slip(u)
                    if u.erased:
                        old, _, _ = spell(u.erased)
                        if old:
                            u.old_key = self._key("erased", old)
                            requests.append((u.old_key, old))
        return requests

    def skip(self, scene):
        self.pen.keep(self.plan(page_units(scene)))

    # Layout ---------------------------------------------------------------

    def render(self, scene):
        """The page's ink (an amount per pixel, see ink.py), and where its first ruled line falls."""
        lines = page_units(scene)
        written = self.pen.write(self.plan(lines))
        self.dot_pool = self._own_dots(lines, written)
        W, H = int(round(scene["width"] * self.px)), int(round(scene["height"] * self.px))
        pen_px = PEN_MM * PT_PER_MM * self.px
        # Margin notes (\cacographynote) stand beside the text: they keep to the line they
        # were written next to rather than to the ruled lines, and lean a little.
        body_left = float(np.median([l["box"]["x"] for l in lines])) if lines else 0.0
        right = self._to_right_margin(lines, scene, body_left)
        margin = {l["box"]["id"] for l in lines if (l["box"]["x"] + l["box"]["w"] < body_left - 2 or l["box"]["id"] in right) and "rot" not in l}
        body = [l for l in lines if l["box"]["id"] not in margin and "rot" not in l]
        grid0 = body[0]["box"]["y"] if body else lines[0]["box"]["y"] if lines else 0.0
        shifts, previous_bottom = {}, -np.inf
        for line in body:
            box = line["box"]
            # Sit on the nearest ruled line that leaves room below the line before.
            lowest = previous_bottom + box["h"]
            snapped = grid0 + round((box["y"] - grid0) / self.spacing) * self.spacing
            if snapped < lowest:
                snapped = grid0 + np.ceil((lowest - grid0) / self.spacing) * self.spacing
            previous_bottom = snapped + box["d"]
            shifts[box["id"]] = snapped - box["y"]
        for line in lines:
            if "rot" in line:
                shifts[line["box"]["id"]] = shifts.get(line["parent"], 0.0)
            elif line["box"]["id"] in margin:
                near = min(body, key=lambda l: abs(l["box"]["y"] - line["box"]["y"]), default=None)
                shifts[line["box"]["id"]] = shifts[near["box"]["id"]] if near else 0.0
        page_rng = np.random.default_rng(subseed(self.pen.seed, "page", scene["page"]))
        canvas = Canvas(W, H, self.dpi / 25.4, pen_of(self.doc["ink"]),
                        np.random.default_rng(subseed(self.pen.seed, "ink", scene["page"])), self.pressure)
        inked = Inked(W, H, 1.5 * pen_px)
        self.size = (W, H)
        ghost, erasures = None, []
        headings = self._headings(lines)
        extra = 0.0  # pt every later line moved down, after one ran into the line above
        for line in lines:
            box = line["box"]
            if "rot" in line:
                shifts[box["id"]] = shifts.get(line["parent"], 0.0)
            else:
                shifts[box["id"]] += extra
            shift_y = shifts[box["id"]] * self.px
            slope = page_rng.normal(0, 0.006 * self.mess) if box["id"] not in margin else page_rng.normal(-0.03, 0.03)
            line["rules"] = [r for r in scene["rules"] if r["line"] == box["id"]]
            boxed = self._boxes(line)
            self._layout_line(line, written)
            left = line["units"][0].x
            tilt = lambda x, left=left, shift_y=shift_y, slope=slope: shift_y + (x - left) * slope  # noqa: E731
            here = canvas
            if "rot" in line:
                # A turned label: written level, then turned into place about its first glyph.
                slope = 0.0
                tilt = lambda x, shift_y=shift_y: shift_y  # noqa: E731
                here = Turned(canvas, (line["pivot"][0] * self.px, line["pivot"][1] * self.px + shift_y), line["rot"])
            else:
                if box["id"] not in margin:
                    # Handwriting runs larger than type: where a line would still write into
                    # the one above (a tall formula under a line of text), it goes down a line.
                    for _ in range(3):
                        pts = [self._unit_points(u, tilt) for u in line["units"]]
                        pts = np.concatenate(pts) if pts else np.zeros((0, 2))
                        if not len(pts) or inked.hits(pts, 0.0) <= max(6, 0.04 * len(pts)):
                            break
                        extra += self.spacing
                        shifts[box["id"]] += self.spacing
                        shift_y += self.spacing * self.px
                        tilt = lambda x, left=left, shift_y=shift_y, slope=slope: shift_y + (x - left) * slope  # noqa: E731
                self._clear_line_above(line, inked, tilt)
            for u in line["units"]:
                if u.math:
                    self._draw_math(here, u, tilt, pen_px)
                elif u.text:
                    if u.erased and here is canvas:
                        ghost = self._draw_erased(ghost, erasures, u, written, tilt, pen_px)
                    self._draw_word(here, u, tilt, pen_px)
                else:
                    self._draw_fallback(here, u, tilt, pen_px)
            for r in line["rules"]:
                self._draw_rule(canvas, r, line, tilt, pen_px)
            self._crossouts(here, line, tilt, pen_px)
            for b in boxed:
                self._draw_box(here, line, b, tilt, pen_px)
            if box["id"] in headings:
                self._underline(canvas, line, tilt, pen_px)
            line["tilt"] = tilt
        self._draw_drawings(canvas, scene, shifts, pen_px)
        by_id = {line["box"]["id"]: line for line in lines}
        for note in scene.get("notes", []):
            if note["line"] in by_id and note["from"] in by_id:
                self._note_arrow(canvas, note, by_id[note["from"]], by_id[note["line"]], pen_px, lines)
        self.erasures, self.rubbed = [], None
        if ghost is not None:
            # A rubbed-out word is never quite gone: a faint smear of what was there.
            self.rubbed = ghost.amount
            self.erasures = erasures
            return canvas.amount + ghost.amount * (0.16 if self.doc["ink"] == "pencil" else 0.4), grid0 * self.px
        return canvas.amount, grid0 * self.px

    def _slip(self, u):
        """Now and then on a well-used pencil page, a word written wrong first and rubbed out:
        the same word with two letters swapped, one left out or one too many. '' mostly."""
        if self.doc.get("ink") != "pencil" or len(u.text) < 4 or not u.text.isalpha():
            return ""
        wear = float(self.doc.get("wear", 0.3))
        rng = np.random.default_rng(subseed(self.pen.seed, "slip", u.key))
        if rng.random() > 0.03 * max(0.0, wear - 0.25) / 0.5:
            return ""
        w = list(u.text)
        k = int(rng.integers(1, len(w) - 1))
        how = rng.choice(["swap", "drop", "double"])
        if how == "swap":
            w[k], w[k + 1] = w[k + 1], w[k]
        elif how == "drop":
            del w[k]
        else:
            w.insert(k, w[k])
        return "".join(w)

    def _draw_erased(self, ghost, erasures, u, written, tilt, pen_px):
        """What was written under `u` first: drawn on its own layer (kept faint, and for the
        grooves it pressed in), and where it was rubbed out, a little wider than the words."""
        if not u.old_key or u.old_key not in written:
            return ghost
        if ghost is None:
            ghost = Canvas(*self.size, self.dpi / 25.4, pen_of(self.doc["ink"]),
                           np.random.default_rng(subseed(self.pen.seed, "erased ink")), self.pressure * 1.1)
        old = Unit(u.glyphs, False, u.x0, u.x1, text=u.erased, key=u.old_key)
        old.shape, old.marks = u.old_shape, []
        old.width = max(p[:, 0].max() for p in old.shape)
        old.x, old.squeeze = u.x + np.random.default_rng(subseed(self.pen.seed, "erased", u.key)).normal(0, 0.03) * u.width, 1.0
        self._draw_word(ghost, old, tilt, pen_px)
        xh = u.glyphs[0]["xh"] * self.px
        base = u.glyphs[0]["y"] * self.px + tilt(u.x)
        x0, x1 = min(u.x, old.x) - 0.6 * xh, max(u.x + u.width, old.x + old.width) + 0.6 * xh
        erasures.append((x0, base - 2.4 * xh, x1, base + 1.2 * xh))
        return ghost

    def _draw_drawings(self, canvas, scene, shifts, pen_px):
        """Graphs and figures (TikZ and the like) traced by hand: each path one stroke."""
        if not scene.get("draw"):
            return
        mm = self.dpi / 25.4
        for k, path in enumerate(drawing.paths(scene["draw"])):
            pts, closed, stroked, filled, width, line, dash = path
            rng = np.random.default_rng(subseed(self.pen.seed, "drawing", scene["page"], k))
            p = pts * self.px
            p[:, 1] += shifts.get(line, 0.0) * self.px
            span = np.ptp(p, axis=0)
            if max(span) < 0.15 * mm:
                continue
            small = filled and max(span) < 3.5 * mm  # an arrow tip or a plot mark: drawn as a small mark
            if filled and not small:
                # A shaded area (a bar, the area under a curve): hatched by hand, the lines
                # all leaning the same way on a page, closer for a darker shade, crossed for
                # a very dark one, left out for a very pale one (outlined only if TeX did).
                shade = path.fill_grey
                if shade >= 0.94 and not stroked:
                    # Too pale to shade: a light outline, so the shape is still there.
                    canvas.stroke(_by_hand(p, rng, mm, True), pen_px * 0.7)
                    continue
                if shade < 0.94:
                    lean = np.radians(45 + np.random.default_rng(subseed(self.pen.seed, "hatch", scene["page"])).normal(0, 10) + rng.normal(0, 3))
                    spacing = (1.0 + 2.6 * shade) * mm * rng.uniform(0.9, 1.1)
                    leans = [lean, lean + np.radians(rng.uniform(70, 100))] if shade < 0.25 else [lean]
                    for a in leans:
                        for seg in _hatch(p, spacing, a, rng, outlined=stroked):
                            canvas.stroke(_by_hand(seg, rng, mm, False), pen_px * 0.9)
                if not stroked:
                    continue
            if small and _roundish(p):
                # A plot mark: a dot filled in with a small tight spiral of the pen.
                c, radius = p.mean(0), 0.5 * float(max(span))
                turns = np.linspace(0, 1, 28)
                ang = rng.uniform(0, 2 * np.pi) + turns * 2 * np.pi * rng.uniform(1.6, 2.4)
                rr = radius * (0.15 + 0.75 * turns) * (1 + rng.normal(0, 0.05, len(turns)))
                canvas.stroke(c + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1), pen_px * 1.1)
                continue
            pale = path.stroke_grey > 0.3 and not filled
            # A pale helper line (a grid) is ruled lightly and carefully: thin, and straighter.
            traced = _by_hand(p, rng, mm, closed or small, steady=0.35 if pale else 1.0)
            weight = pen_px * (1.4 if small else 1.0) * np.clip(width / 0.4, 0.8, 1.6)
            if pale:
                weight = pen_px * 0.55
            if dash and not small:
                # Dashes by hand: about as long as asked, each its own little stroke.
                on, off = dash[0] * self.px, (dash[1] if len(dash) > 1 else dash[0]) * self.px
                steps = np.r_[0, np.cumsum(np.hypot(*np.diff(traced, axis=0).T))]
                at = rng.uniform(0, 0.3) * on
                while at < steps[-1]:
                    run = on * rng.uniform(0.8, 1.25)
                    keep = (steps >= at) & (steps <= at + run)
                    if keep.sum() >= 2:
                        canvas.stroke(traced[keep], weight)
                    at += run + off * rng.uniform(0.8, 1.2)
                continue
            canvas.stroke(traced, weight)
            if small:
                c = p.mean(0)
                for q in p[:: max(1, len(p) // 4)]:
                    canvas.stroke(np.stack([q, c]), pen_px * 1.2)

    def _note_arrow(self, canvas, note, source, target, pen_px, lines):
        """From a margin note to where it was written: from the end of the note, one stroke
        curving round to the place (or, when that is far along its line, to the line's
        start), and a head flicked on."""
        rng = np.random.default_rng(subseed(self.pen.seed, "note arrow", note["id"]))
        units = target["units"]
        if not units or not source["units"]:
            return
        xh = float(np.median([g["xh"] for u in units for g in u.glyphs])) * self.px
        # The note's own lines: from its first, down while they stay in the margin.
        ids = [l["box"]["id"] for l in lines]
        mine = [source]
        for l in lines[ids.index(source["box"]["id"]) + 1 :]:
            if abs(l["box"]["x"] - source["box"]["x"]) > 1 or not l["units"]:
                break
            mine.append(l)
        last = mine[-1]
        # A note in the right margin points back left, from where it starts.
        side = -1 if source["box"]["x"] > note["x"] else 1
        if side > 0:
            start_x = max(u.x + u.width for u in last["units"]) + rng.uniform(0.2, 0.5) * xh
            start_y = last["box"]["y"] * self.px + last["tilt"](start_x) - rng.uniform(0.3, 0.6) * xh
        else:
            first = mine[0]
            start_x = min(u.x for u in first["units"]) - rng.uniform(0.2, 0.5) * xh
            start_y = first["box"]["y"] * self.px + first["tilt"](start_x) - rng.uniform(0.3, 0.6) * xh
        # The place, moved as its words were.
        owner = min(units, key=lambda u: 0 if u.x0 <= note["x"] <= u.x1 else min(abs(u.x0 - note["x"]), abs(u.x1 - note["x"])))
        end_x = note["x"] * self.px + owner.x - owner.x0 * self.px + rng.normal(0, 0.3) * xh
        end_y = target["box"]["y"] * self.px + target["tilt"](end_x) - rng.uniform(1.1, 1.4) * xh
        if side * (end_x - start_x) > 12 * xh:
            if side > 0:
                end_x = min(u.x for u in units) - rng.uniform(0.2, 0.5) * xh
            else:
                end_x = max(u.x + u.width for u in units) + rng.uniform(0.2, 0.5) * xh
            end_y = target["box"]["y"] * self.px + target["tilt"](end_x) - rng.uniform(0.3, 0.6) * xh
        # Out away from the note first, then round towards the place.
        a, b = np.array([start_x, start_y]), np.array([end_x, end_y])
        # Two pulls, each set by hand, so the curve tightens unevenly; a slow waver across
        # it; sometimes a little hook where the pen came down.
        span = b - a
        c1 = a + np.array([span[0] * rng.uniform(0.3, 0.9) + side * rng.uniform(0.2, 1.0) * xh, span[1] * rng.uniform(-0.1, 0.25)])
        c2 = b + np.array([side * rng.uniform(0.0, 1.2) * xh, -span[1] * rng.uniform(0.2, 0.6)])
        t = np.linspace(0, 1, 32)[:, None] ** rng.uniform(0.85, 1.15)
        path = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t * t * c2 + t ** 3 * b
        tangent = np.gradient(path, axis=0)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1) / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9)
        waver = sum(rng.uniform(0.04, 0.12) * xh * np.sin(2 * np.pi * f * t[:, 0] + rng.uniform(0, 6)) for f in (rng.uniform(0.7, 1.4), rng.uniform(2, 3.5)))
        path += normal * (waver * np.sin(np.pi * t[:, 0]))[:, None]
        if rng.random() < 0.35:
            hook = path[0] + np.array([rng.uniform(-0.3, 0.1), rng.uniform(0.15, 0.35)]) * xh
            path = np.vstack([hook, path])
        canvas.stroke(path, pen_px)
        direction = path[-1] - path[-3]
        direction /= np.linalg.norm(direction) + 1e-9
        normal = np.array([-direction[1], direction[0]])
        length = rng.uniform(0.6, 0.9) * xh
        for side in (-1, 1):
            back = -direction * np.cos(0.5) + side * normal * np.sin(0.5)
            tip = path[-1] + rng.normal(0, 0.04 * xh, 2)
            canvas.stroke(np.stack([tip + back * length * rng.uniform(0.85, 1.15), tip]), pen_px)

    def _to_right_margin(self, lines, scene, body_left):
        """Notes marked for the right margin (\\cacographynote[right]) were set in the left
        one; move them, with the lines that follow them there, beside the text's right edge."""
        marked = {str(i) for i in scene.get("right_notes", [])} | set(scene.get("right_notes", []))
        if not marked:
            return set()
        body = [l for l in lines if abs(l["box"]["x"] - body_left) < 1]
        body_right = float(np.median([l["box"]["x"] + l["box"]["w"] for l in body])) if body else body_left
        moved = set()
        for k, line in enumerate(lines):
            if line["box"]["id"] not in marked:
                continue
            group = [line]
            for later in lines[k + 1 :]:
                if abs(later["box"]["x"] - line["box"]["x"]) > 1 or "rot" in later:
                    break
                group.append(later)
            dx = body_right + 10.0 - line["box"]["x"]
            # Beside a short line (a signature, a heading) the note is written just after
            # it, not across the page at the margin with a long arrow back.
            note = next((n for n in scene.get("notes", []) if str(n.get("from")) == str(line["box"]["id"])), None)
            anchor = next((l for l in lines if note is not None and l["box"]["id"] == note.get("line")), None)
            if anchor is not None and anchor["units"]:
                text_end = max(g["x"] + g["w"] for u in anchor["units"] for g in u.glyphs)
                if body_right - text_end > 0.35 * (body_right - body_left):
                    dx = text_end + 18.0 - line["box"]["x"]
            for l in group:
                l["box"] = dict(l["box"], x=l["box"]["x"] + dx)
                for u in l["units"]:
                    u.x0, u.x1 = u.x0 + dx, u.x1 + dx
                    for g in u.glyphs:
                        g["x"] += dx
                moved.add(l["box"]["id"])
        return moved

    def _crossed_points(self, u, tilt):
        """The ink of a unit's crossed-out part: the whole word, or the crossed glyphs of math."""
        if not u.math:
            return self._unit_points(u, tilt)
        parts = [p + [u.x + it["shift"], 0.0] for it in u.items if it["strokes"] and it["g"].get("cross") for p in it["strokes"]]
        out = []
        for p in parts:
            p = p.copy()
            p[:, 1] += tilt(p[:, 0])
            out.append(p)
        return np.concatenate(out) if out else np.zeros((0, 2))

    def _boxes(self, line):
        """\\boxed and \\fbox: four rules round some math. Taken out of the line's rules, to be
        drawn as one box by hand round the ink they enclose (pt: x0, y0, x1, y1)."""
        rules = line["rules"]
        flat = [r for r in rules if r["w"] > 3 * r["h"]]
        tall = [r for r in rules if r["h"] > 3 * r["w"]]
        boxes, used = [], set()
        for i, a in enumerate(flat):
            for b in flat[i + 1 :]:
                if abs(a["x"] - b["x"]) > 1 or abs(a["w"] - b["w"]) > 1 or abs(a["y"] - b["y"]) < 2:
                    continue
                top, bottom = (a, b) if a["y"] < b["y"] else (b, a)
                sides = [v for v in tall if (abs(v["x"] - a["x"]) < 1.5 or abs(v["x"] + v["w"] - a["x"] - a["w"]) < 1.5)
                         and v["y"] >= top["y"] - 1.5 and v["y"] + v["h"] <= bottom["y"] + bottom["h"] + 1.5]
                if len(sides) >= 2:
                    used |= {id(a), id(b)} | {id(v) for v in sides[:2]}
                    boxes.append((a["x"], top["y"], a["x"] + a["w"], bottom["y"] + bottom["h"]))
        line["rules"] = [r for r in rules if id(r) not in used]
        return boxes

    def _draw_box(self, canvas, line, box, tilt, pen_px):
        """A box drawn round a result: four quick strokes or one, corners overshooting."""
        rng = np.random.default_rng(subseed(self.pen.seed, "box", box[0], box[1]))
        inside = [u for u in line["units"] if u.x0 < box[2] and u.x1 > box[0]]
        pts = np.concatenate([self._unit_points(u, tilt) for u in inside]) if inside else np.zeros((0, 2))
        if len(pts) < 2:
            return
        xh = float(np.median([0.43 * g["size"] for u in inside for g in u.glyphs])) * self.px
        pad = rng.uniform(0.45, 0.75) * xh
        x0, x1 = pts[:, 0].min() - pad, pts[:, 0].max() + pad
        y0, y1 = pts[:, 1].min() - pad, pts[:, 1].max() + pad
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) + rng.normal(0, 0.1 * xh, (4, 2))
        if rng.random() < 0.5:
            # One stroke round, not quite closing where it began.
            path = np.vstack([corners, corners[:1] + rng.normal(0, 0.15 * xh, 2) + [0.3 * xh, 0]])
            dense = np.concatenate([np.linspace(path[k], path[k + 1], 10, endpoint=False) for k in range(4)] + [path[-1:]])
            canvas.stroke(dense + rng.normal(0, 0.03 * xh, dense.shape), pen_px)
        else:
            for k in range(4):
                a, b = corners[k], corners[(k + 1) % 4]
                d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
                a, b = a - d * rng.uniform(0, 0.3) * xh, b + d * rng.uniform(0.05, 0.4) * xh
                seg = np.linspace(a, b, 10) + rng.normal(0, 0.03 * xh, (10, 2))
                canvas.stroke(seg, pen_px)

    def _crossouts(self, canvas, line, tilt, pen_px):
        """Words in \\crossout struck through by hand: one wavering line overshooting both
        ends, two lines, or a scribble over them, as this writer does it."""
        runs, prev = [], None
        for u in line["units"]:
            if u.cross and not u.fig:
                if runs and prev is runs[-1][-1] and prev.cross == u.cross:
                    runs[-1].append(u)
                else:
                    runs.append([u])
            prev = u
        habit = np.random.default_rng(subseed(self.pen.seed, "crossout habit"))
        usual = int(habit.choice([2, 3, 4], p=[0.6, 0.18, 0.22]))
        # How this writer blacks words out: long loops scrubbed back and forth over them, a
        # coil of loops run along them, or a knot scrawled over a short word.
        scrub_habit = str(habit.choice(["scrub", "coil", "tangle"], p=[0.65, 0.15, 0.2]))
        text_xh = [g["xh"] for u in line["units"] if not u.math for g in u.glyphs]
        for run in runs:
            pts = np.concatenate([self._crossed_points(u, tilt) for u in run])
            if len(pts) < 2:
                continue
            rng = np.random.default_rng(subseed(self.pen.seed, "crossout", line["box"]["id"], run[0].x0))
            style = usual if run[0].cross == 1 else run[0].cross
            # Math fonts give no useful x-height: take it from the text, or from the size.
            xh = float(np.median(text_xh or [0.43 * g["size"] for u in run for g in u.glyphs])) * self.px
            crossed = [g for u in run for g in u.glyphs if g.get("cross")] or run[0].glyphs
            x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
            # Struck along the words as they were written: through the middle of their ink,
            # at the slope it runs (handwriting drifts off the ruled line).
            slope = float(np.clip(np.polyfit(pts[:, 0], pts[:, 1], 1)[0], -0.08, 0.08)) if x1 - x0 > xh else 0.0
            centre = float(np.median(pts[:, 1] - slope * pts[:, 0]))
            level = lambda x: centre + slope * np.atleast_1d(x)  # noqa: E731
            if style in (2, 3):
                # Two strokes are spaced by the ink's own height, so they stay apart on small
                # writing too (the font's x-height can be far off the hand's).
                lo_, hi_ = np.percentile(pts[:, 1] - slope * pts[:, 0], [15, 85])
                gap = max(0.3 * xh, 0.42 * float(hi_ - lo_), 2.2 * pen_px) * rng.uniform(0.9, 1.15)
                for k in range(1 if style == 2 else 2):
                    a = x0 - rng.uniform(0.1, 0.6) * xh
                    b = x1 + rng.uniform(0.1, 0.8) * xh
                    n = max(8, int((b - a) / (0.5 * xh)))
                    xs = np.linspace(a, b, n)
                    t = np.linspace(0, 1, n)
                    off = (k - 0.5) * gap + rng.normal(0, 0.04) * xh if style == 3 else rng.normal(-0.05, 0.08) * xh
                    ys = level(xs) + off + rng.normal(0, 0.012) * (xs - a) + rng.normal(0, 0.06) * xh * np.sin(np.pi * t)
                    ys += rng.normal(0, 0.03 * xh, n).cumsum() * 0.5  # the hand wavers
                    ys[-1] += rng.uniform(-0.2, 0.05) * xh  # and flicks off at the end
                    if k == 1 and rng.random() < 0.6:
                        xs, ys = xs[::-1], ys[::-1]  # the second stroke comes back
                    canvas.stroke(np.stack([xs, ys], 1), pen_px)
            else:
                kind = scrub_habit if rng.random() < 0.75 else str(rng.choice(["scrub", "coil", "tangle"]))
                if kind == "tangle" and x1 - x0 > 4 * xh:
                    kind = "scrub"  # a knot is for a short word; a long one is scrubbed
                # Points as far apart as the handwriting's, a little more: quick, pressed hard.
                typical = getattr(canvas, "typical", None)
                spacing = (4 * float(np.median(typical)) if typical else 0.2 * xh) * rng.uniform(1.0, 1.25)
                # Sized by the ink itself: handwriting is larger than the font it replaces.
                lo, hi = np.percentile(pts[:, 1] - slope * pts[:, 0], [6, 94])
                size = max(xh, float(hi - lo) / 1.1)
                mid = 0.5 * float(lo + hi)
                for stroke in _scribble(kind, x0, x1, lambda x: mid + slope * np.atleast_1d(x), size, rng):
                    canvas.stroke(_smooth_path(stroke, spacing), pen_px * (1.0 if kind == "scrub" else 1.08))

    def _headings(self, lines):
        """Lines to underline: headings (bold, larger than the text) if this writer does so.

        underline=headings always, none never; at random, about half of writers do.
        """
        choice = str(self.doc.get("underline", "random"))
        if choice == "none" or not lines:
            return set()
        if choice == "random" and np.random.default_rng(subseed(self.pen.seed, "underline habit")).random() >= 0.5:
            return set()
        sizes = [g["size"] for line in lines for u in line["units"] if not u.math for g in u.glyphs]
        if not sizes:
            return set()
        body = float(np.median(sizes))
        out = set()
        for line in lines:
            text = [g for u in line["units"] if not u.math for g in u.glyphs if g["t"].strip()]
            if text and np.mean([g.get("bold", False) for g in text]) > 0.8 and np.median([g["size"] for g in text]) > 1.1 * body:
                out.add(line["box"]["id"])
        return out

    def _ink_baseline(self, pts, xh):
        """Where the words' ink actually sits: (slope, height at x = 0) of its baseline, from
        the middle of the ink and the slope it runs at (handwriting drifts off the ruled line)."""
        if len(pts) < 4 or np.ptp(pts[:, 0]) < xh:
            return 0.0, (float(np.median(pts[:, 1])) + 0.45 * xh) if len(pts) else 0.0
        slope = float(np.clip(np.polyfit(pts[:, 0], pts[:, 1], 1)[0], -0.08, 0.08))
        rest = pts[:, 1] - slope * pts[:, 0]
        # The bottom of the small letters: well below the middle, but above the descenders.
        mid = float(np.median(rest))
        body = rest[(rest > mid) & (rest < mid + 0.8 * xh)]
        bottom = float(np.percentile(body, 85)) if len(body) > 4 else mid + 0.45 * xh
        return slope, min(max(bottom, mid + 0.3 * xh), mid + 0.6 * xh)

    def _underline(self, canvas, line, tilt, pen_px):
        """A heading underlined in one quick stroke, sometimes two, just under its words."""
        rng = np.random.default_rng(subseed(self.pen.seed, "underline", line["box"]["id"], line["box"]["y"]))
        units = line["units"]
        xh = float(np.median([g["xh"] for u in units for g in u.glyphs])) * self.px
        x0 = min(u.x for u in units) - rng.uniform(-0.2, 0.6) * xh
        x1 = max(u.x + u.width for u in units) + rng.uniform(-0.1, 1.0) * xh
        pts = np.concatenate([self._unit_points(u, tilt) for u in units if not u.fig] or [np.zeros((0, 2))])
        slope, base = self._ink_baseline(pts, xh)
        # Along the words, more or less: the stroke's own slope is a little off theirs.
        slope += rng.normal(0, 0.012)
        for k in range(2 if rng.random() < 0.12 else 1):
            y = base + rng.uniform(0.3, 0.5) * xh + k * rng.uniform(0.22, 0.3) * xh
            xs = np.linspace(x0 + rng.normal(0, 0.1) * xh, x1 + rng.normal(0, 0.15) * xh, 14)
            bow = rng.normal(0, 0.08) * xh * np.sin(np.linspace(0, np.pi, 14))
            ys = y + bow + slope * xs
            # The pen lifts off at the end with a short flick.
            ys[-1] += rng.uniform(-0.12, 0.04) * xh
            canvas.stroke(np.stack([xs, ys], 1), pen_px)

    def _unit_points(self, u, tilt):
        """Where a unit's strokes fall on the page, with the midpoints between their points."""
        if u.math:
            parts = [p + [u.x + it["shift"], 0.0] for it in u.items if it["strokes"] for p in it["strokes"]]
        elif u.text:
            baseline = u.glyphs[0]["y"] * self.px
            parts = [np.stack([u.x + p[:, 0] * u.squeeze, p[:, 1] + baseline], 1) for p in u.shape]
        else:
            return np.zeros((0, 2))
        out = []
        for p in parts:
            p = p.copy()
            p[:, 1] += tilt(p[:, 0])
            out.append(p)
            if len(p) > 1:
                out.append(0.5 * (p[1:] + p[:-1]))
        return np.concatenate(out) if out else np.zeros((0, 2))

    def _clear_line_above(self, line, inked, tilt):
        """Slide words along their line, within their gaps, to miss the ink of the line above.

        A careful hand sees what's already on the page: a tall letter steers around the tail
        of a g above it. Then this line's ink joins what the next line has to avoid.
        """
        units = line["units"]
        xh = units[0].glyphs[0]["xh"] * self.px
        room = NUDGE_ROOM * xh
        earliest = -np.inf
        for i, u in enumerate(units):
            points = self._unit_points(u, tilt)
            shift = 0.0
            if len(points) and inked.hits(points, 0.0) and not u.fig:
                latest = units[i + 1].x - u.x - u.width - room if i + 1 < len(units) else NUDGE_MOST * xh
                choices = [d * xh for d in NUDGES if earliest - u.x <= d * xh <= latest]
                if choices:
                    shift = min(choices, key=lambda d: (inked.hits(points, d), abs(d)))
            u.x += shift
            earliest = u.x + u.width + room
        for u in units:
            points = self._unit_points(u, tilt)
            if len(points):
                inked.add(points)

    def _word_scale(self, glyph):
        return glyph["xh"] * self.px / XH * self.pen.size  # page pixels per working pixel

    def _layout_line(self, line, written):
        box, units = line["box"], line["units"]
        xh_px = units[0].glyphs[0]["xh"] * self.px
        rng = np.random.default_rng(subseed(self.pen.seed, "line", units[0].key or units[0].x0))
        for u in units:
            if u.math:
                self._shape_math(u, written, line["rules"])
            elif u.text:
                self._shape_word(u, written)
                if u.old_key in written:
                    # The words after it were written after the old word: they leave its room.
                    old = Unit(u.glyphs, False, u.x0, u.x1, text=u.erased, key=u.old_key)
                    self._shape_word(old, written)
                    u.old_shape, u.width = old.shape, max(u.width, old.width)
            else:
                u.width = (u.x1 - u.x0) * self.px
        for u in units:
            if u.fig:
                u.x = 0.5 * (u.x0 + u.x1) * self.px - 0.5 * u.width
                u.squeeze = 1.0
        units = [u for u in units if not u.fig]
        if not units:
            return
        # The rows of an aligned display share one line: each row is spaced on its own,
        # starting where TeX started it.
        rows, last = [], {}
        for u in units:
            y, size = u.glyphs[0]["y"], u.glyphs[0]["size"]
            row = next((r for r, ry in enumerate(rows) if abs(ry - y) < 0.6 * size), None)
            if row is None:
                rows.append(y)
                row = len(rows) - 1
            u.row = row
        positions, ends = [], []
        for i, u in enumerate(units):
            prev = last.get(u.row)
            last[u.row] = u
            if prev is None:
                x = (u.x0 - units[0].x0) * self.px
            else:
                cursor = positions[units.index(prev)] + prev.width
                tex_gap = (u.x0 - prev.x1) * self.px
                if prev.math and u.math:
                    gap = max(tex_gap, 0.2 * xh_px)
                else:
                    # The writer's own spacing, a little steadier within a line than across a page.
                    k = xh_px / XH  # gaps were measured at true size, so they skip the size correction
                    median, spread = self.pen.gap
                    gap = max(0.8 * xh_px, rng.normal(median, 0.6 * spread * self.mess) * k)
                x = cursor + gap
                if tex_gap > 4 * xh_px:
                    # A deliberate space (an equation number, \quad, a centred block): keep TeX's.
                    x = max(x, (u.x0 - units[0].x0) * self.px)
            positions.append(x)
            ends.append(x + u.width)
        total = max(ends)
        left_gap = units[0].x0 - box["x"]
        right_gap = box["x"] + box["w"] - max(u.x1 for u in units)
        centred = left_gap > 20 and abs(left_gap - right_gap) < 15
        start = (box["x"] + box["w"] / 2) * self.px - total / 2 if centred else units[0].x0 * self.px
        limit = (box["x"] + box["w"]) * self.px + 0.4 * xh_px
        squeeze = 1.0
        if start + total > limit and total > 0:
            # Out of room: tighten the gaps, then the writing itself, then spill into the margin.
            squeeze = max(0.85, (limit - start) / total)
        for u, x in zip(units, positions):
            u.x = start + x * squeeze
            u.squeeze = squeeze

    # Drawing --------------------------------------------------------------

    def _place(self, strokes, k, x, baseline, tilt, jitter):
        scale, dy, angle = jitter
        c, s = np.cos(angle), np.sin(angle)
        out = []
        for st in strokes:
            p = st * k * scale
            p = np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1)
            p[:, 0] += x
            p[:, 1] += baseline + dy + tilt(p[:, 0])
            out.append(p)
        return out

    def _jitter(self, key, xh_px):
        rng = np.random.default_rng(subseed(self.pen.seed, key, "place"))
        m = self.mess
        return rng.normal(1.0, 0.03 * m), rng.normal(0, 0.06 * xh_px * m), rng.normal(0, np.radians(0.8) * m)

    def _shape_word(self, u, written):
        """The word's strokes exactly as they will be drawn, left edge at 0 on the baseline."""
        steps, chars = written[u.key]
        g0 = u.glyphs[0]
        strokes = strokes_of(steps)
        placed = self._place(strokes, self._word_scale(g0), 0.0, 0.0, lambda x: 0.0, self._jitter(u.key, g0["xh"] * self.px))
        if len(u.text) <= 3:
            # Alone, a letter or two comes out of the pen far too big (a lone capital "A"
            # twice the height of the words round it): hold it to a tall letter's height.
            xh = g0["xh"] * self.px
            tall = -min(p[:, 1].min() for p in placed)
            if tall > 2.6 * xh:
                placed = [p * (2.0 * xh / tall) for p in placed]
        left = min(p[:, 0].min() for p in placed)
        u.shape = [p - [left, 0] for p in placed]
        u.width = max(p[:, 0].max() for p in u.shape)
        u.chars = chars

    def _draw_word(self, canvas, u, tilt, pen_px):
        baseline = u.glyphs[0]["y"] * self.px
        placed = []
        for p in u.shape:
            q = p.copy()
            q[:, 0] = u.x + q[:, 0] * u.squeeze
            q[:, 1] += baseline + tilt(q[:, 0])
            placed.append(q)
            canvas.stroke(q, pen_px)
        if u.marks:
            self._draw_marks(canvas, u, np.concatenate(placed), u.chars, u.glyphs[0]["xh"] * self.px, pen_px)

    def _own_dots(self, lines, written):
        """The writer's own i and j dots on this page: each dot's shape, and its height over
        its letter, in x-heights. Umlauts borrow them, so they match the i-dots beside them."""
        pool = []
        for line in lines:
            for u in line["units"]:
                if u.math or not u.text or u.key not in written or not any(c in "ij" for c in u.text):
                    continue
                steps, chars = written[u.key]
                xy = np.cumsum(steps[:, :2], 0)
                stems = [k for k, c in enumerate(u.text) if c in "ij" and (chars == k).sum() >= 3]
                for _, a, b, kind in classify(steps, u.text):
                    if kind != "dot" or not stems:
                        continue
                    dot = xy[a:b]
                    centre = dot.mean(0)
                    k = min(stems, key=lambda k: abs(np.median(xy[chars == k, 0]) - centre[0]))
                    own = chars == k
                    own[a:b] = False  # a dot is labelled as part of its letter
                    if own.sum() >= 3:
                        pool.append(dict(shape=(dot - centre) / MODEL_XH, above=float((xy[own, 1].min() - centre[1]) / MODEL_XH)))
        return pool

    def _umlaut(self, mine, xh, key, index):
        """Two dots, or one dash, over a vowel: this writer's habit, never quite the same twice.

        The dots are the writer's own i-dots from the same page when there are any, at the
        height this writer dots an i; otherwise ticks drawn from the measured habit.
        """
        habit = self.umlaut
        rng = np.random.default_rng(subseed(self.pen.seed, "umlaut", key, index))
        pool = [d for d in getattr(self, "dot_pool", []) if 0.15 < d["above"] < 1.4]
        left, right = np.percentile(mine[:, 0], [15, 85])
        cx = (left + right) / 2 + rng.normal(habit["shift"], 0.06) * xh
        top = float(mine[:, 1].min())
        capital = np.ptp(mine[:, 1]) > 1.3 * xh  # less room under the line above
        above = float(np.median([d["above"] for d in pool])) if pool else habit["above"]
        # Over a capital there is little room under the line above, so even a writer who dots
        # their i high keeps an umlaut close.
        above = min(above * 0.5, 0.3) if capital else above
        y = top - xh * max(0.12 if capital else 0.2, above + rng.normal(0, 0.06 if capital else 0.08))
        half = max(right - left, 0.5 * xh) * habit["spread"] / 2 * rng.uniform(0.85, 1.15)
        if habit["dash"]:
            lean = np.radians(rng.normal(-4, 5))
            xs = np.linspace(-0.85, 0.85, 4) * half
            ys = xs * np.tan(lean) - 0.04 * xh * (1 - (xs / half) ** 2) + rng.normal(0, 0.01, 4) * xh
            return [np.stack([cx + xs, y + ys], 1)]
        dots = []
        if pool:
            for side, pick in zip((-1, 1), rng.choice(len(pool), 2)):
                c = np.array([cx + side * half + rng.normal(0, 0.03) * xh, y + rng.normal(0, 0.05) * xh])
                dots.append(c + pool[pick]["shape"] * xh * rng.uniform(0.9, 1.1))
            return dots
        for side in (-1, 1):
            length = xh * habit["length"] * rng.uniform(0.7, 1.3)
            turn = np.radians(habit["angle"] + rng.normal(0, 15))
            d = np.array([np.cos(turn), np.sin(turn)]) * length / 2
            c = np.array([cx + side * half + rng.normal(0, 0.03) * xh, y + rng.normal(0, 0.05) * xh])
            dots.append(np.array([c - d, c + d]))
        return dots

    def _draw_marks(self, canvas, u, points, owner, xh, pen_px):
        for index, kind in u.marks:
            mine = points[: len(owner)][owner == index]
            if not len(mine):
                continue
            cx, top = float(np.median(mine[:, 0])), float(mine[:, 1].min())
            y = top - 0.45 * xh
            if kind == "diaeresis":
                for dot in self._umlaut(mine, xh, u.key or "", index):
                    canvas.stroke(dot, pen_px)
            elif kind == "acute":
                canvas.stroke([[cx - 0.08 * xh, y + 0.15 * xh], [cx + 0.18 * xh, y - 0.25 * xh]], pen_px)
            elif kind == "grave":
                canvas.stroke([[cx - 0.18 * xh, y - 0.25 * xh], [cx + 0.08 * xh, y + 0.15 * xh]], pen_px)
            elif kind == "circumflex":
                canvas.stroke([[cx - 0.2 * xh, y + 0.1 * xh], [cx, y - 0.2 * xh], [cx + 0.2 * xh, y + 0.1 * xh]], pen_px)
            elif kind == "tilde":
                canvas.stroke([[cx - 0.25 * xh, y], [cx - 0.08 * xh, y - 0.12 * xh], [cx + 0.08 * xh, y], [cx + 0.25 * xh, y - 0.12 * xh]], pen_px)

    def _glyph_strokes(self, g, cx, base, xh, written):
        """One math glyph as strokes, centred on `cx`, or None if only the font can draw it."""
        if g.get("bank"):
            parts = self.bank.write(g["bank"], g["bank_key"])
            ink = np.concatenate(parts)
            size = np.ptp(ink, axis=0)
            # As large as fits TeX's box for the glyph: writers copy a symbol's proportions.
            box_w, box_h = g["w"] * self.px, (g["h"] + g["d"]) * self.px
            k = min(box_w / max(size[0], 1e-3), box_h / max(size[1], 1e-3))
            middle = 0.5 * (ink.min(0) + ink.max(0))
            mid_y = base + 0.5 * (g["d"] - g["h"]) * self.px
            strokes = [(p - middle) * k + [cx, mid_y] for p in parts]
        elif g.get("letters"):
            steps, _ = written[g["key"]]
            parts = strokes_of(steps)
            lo = min(s[:, 0].min() for s in parts)
            hi = max(s[:, 0].max() for s in parts)
            k = self._word_scale(g)
            strokes = self._place([s - [lo, 0] for s in parts], k, cx - (hi - lo) * k / 2, base, lambda x: 0.0, self._jitter(g["key"], xh))
        elif g.get("symbol"):
            return self._symbol_strokes(g, cx, base, xh)
        else:
            return None
        if is_vector(g["t"]):
            top = min(p[:, 1].min() for p in strokes) - 0.35 * xh
            a, b = cx - 0.45 * xh, cx + 0.5 * xh
            strokes.append(np.array([[a, top], [b, top - 0.04 * xh]]))
            strokes.append(np.array([[b - 0.22 * xh, top - 0.18 * xh], [b, top - 0.04 * xh], [b - 0.2 * xh, top + 0.14 * xh]]))
        return strokes

    def _shape_math(self, u, written, rules):
        """Lay a math unit's glyphs out by hand: TeX's arrangement, moved apart where ink collides.

        Handwriting runs wider than type. A glyph keeps at least the shift of everything TeX
        set before it, and moves further if its ink would run into a glyph at the same height.
        Glyphs TeX stacked (a numerator and its denominator, an index and an exponent) overlap
        in x but not in height, so they don't push each other. x is relative to the unit's start.
        """
        items = []
        for g in sorted(u.glyphs, key=lambda g: g["x"]):
            tex_left, tex_right = (g["x"] - u.x0) * self.px, (g["x"] + g["w"] - u.x0) * self.px
            base, xh = g["y"] * self.px, g["xh"] * self.px
            strokes = self._glyph_strokes(g, 0.5 * (tex_left + tex_right), base, xh, written)
            if strokes:
                ink = np.concatenate(strokes)
                box = (ink[:, 0].min(), ink[:, 0].max(), ink[:, 1].min(), ink[:, 1].max())
            elif g.get("missing"):
                for m in g["missing"]:
                    self.missing[m] = self.missing.get(m, 0) + 1
                box = (tex_left, tex_right, base - g["h"] * self.px, base + g["d"] * self.px)
            else:
                continue
            items.append(dict(g=g, strokes=strokes, tex_left=tex_left, tex_right=tex_right, left=box[0], right=box[1], top=box[2], bottom=box[3], shift=0.0, bar=None))
        for r in rules:
            left, right = (r["x"] - u.x0) * self.px, (r["x"] + r["w"] - u.x0) * self.px
            for it in items:
                if left - 1 <= 0.5 * (it["tex_left"] + it["tex_right"]) <= right + 1:
                    it["bar"] = (r["y"] + r["h"] / 2) * self.px
        pen_px = PEN_MM * PT_PER_MM * self.px
        def same_row(p, it):
            # TeX's boxes overlap in height: the same row (sub- and superscripts included),
            # not the row above or below in an aligned block.
            a, b = p["g"], it["g"]
            return min(a["y"] + a["d"], b["y"] + b["d"]) > max(a["y"] - a["h"], b["y"] - b["h"])

        for i, it in enumerate(items):
            shift = max((p["shift"] for p in items[:i] if p["tex_right"] <= it["tex_left"] + 0.5 * self.px
                         and (same_row(p, it) or it["bar"] is not None)), default=0.0)
            xh = it["g"]["xh"] * self.px
            for p in items[:i]:
                if not same_row(p, it) and it["bar"] is None and p["bar"] is None:
                    continue  # another row of an aligned block: TeX keeps them apart
                shared = min(p["bottom"], it["bottom"]) - max(p["top"], it["top"])
                if shared > 0.1 * min(p["bottom"] - p["top"], it["bottom"] - it["top"]):
                    # Centre lines this far apart leave air between strokes the pen's width.
                    shift = max(shift, p["right"] + p["shift"] + max(MATH_AIR * xh, 1.5 * pen_px) - it["left"])
                # A fraction's bar runs past its math, so something level with it needs room too.
                if it["bar"] is not None and p["bar"] != it["bar"] and p["top"] - 0.2 * xh < it["bar"] < p["bottom"] + 0.2 * xh:
                    shift = max(shift, p["right"] + p["shift"] + 0.45 * xh - it["left"])
            it["shift"] = shift
        self._centre_fractions(u, items, rules)
        # Measure the unit from its ink, so gaps between units are gaps between ink.
        left = min((it["left"] + it["shift"] for it in items), default=0.0)
        for it in items:
            it["shift"] -= left
        for r in rules:
            if r.get("owner") is u:
                r["span"] = (r["span"][0] - left, r["span"][1] - left)
        u.items = items
        u.width = max((it["right"] + it["shift"] for it in items), default=0.0)

    def _centre_fractions(self, u, items, rules):
        """Bars span their math, with numerator and denominator centred over each other."""
        for r in rules:
            left, right = (r["x"] - u.x0) * self.px, (r["x"] + r["w"] - u.x0) * self.px
            mine = [it for it in items if left - 1 <= 0.5 * (it["tex_left"] + it["tex_right"]) <= right + 1]
            if not mine:
                continue
            y = (r["y"] + r["h"] / 2) * self.px
            above = [it for it in mine if it["top"] + it["bottom"] < 2 * y]
            below = [it for it in mine if it["top"] + it["bottom"] >= 2 * y]

            def extent(group):
                return min(it["left"] + it["shift"] for it in group), max(it["right"] + it["shift"] for it in group)

            lo, hi = extent(mine)
            if above and below:
                for group in (above, below):
                    a, b = extent(group)
                    for it in group:
                        it["shift"] += 0.5 * (lo + hi) - 0.5 * (a + b)
            r["owner"], r["span"] = u, (lo, hi)

    def _draw_math(self, canvas, u, tilt, pen_px):
        for it in u.items:
            if it["strokes"]:
                for p in it["strokes"]:
                    q = p.copy()
                    q[:, 0] += u.x + it["shift"]
                    q[:, 1] += tilt(q[:, 0])
                    canvas.stroke(q, pen_px)
            else:
                x = u.x + it["shift"] + it["tex_left"]
                canvas.glyph(it["g"]["t"], x, it["g"]["y"] * self.px + tilt(x), it["g"]["size"] * self.px, self.math_font)

    def _symbol_strokes(self, g, cx, base, xh):
        """Plus, minus, equals and math's full stops and commas as a hand draws them."""
        rng = np.random.default_rng(subseed(self.pen.seed, "symbol", g["x"], g["y"]))
        if g["symbol"] in ".,":
            x, y = cx + rng.normal(0, 0.03 * xh), base - rng.uniform(0.0, 0.08) * xh
            if g["symbol"] == ".":
                return [np.array([[x, y], [x + 0.05 * xh, y + rng.normal(0.02, 0.02) * xh]])]
            return [np.array([[x + 0.04 * xh, y - 0.04 * xh], [x + 0.03 * xh, y + 0.15 * xh], [x - 0.09 * xh, y + 0.38 * xh]])]
        if g["symbol"] in PRIMES:
            # Primes: short ticks leaning to the left, one per prime, high beside the letter.
            n = PRIMES[g["symbol"]]
            top = base - g["h"] * self.px
            length = max(0.55 * xh, 0.7 * g["h"] * self.px)
            gap = 0.8 * g["w"] * self.px / n
            ticks = []
            for k in range(n):
                x = cx + (k - 0.5 * (n - 1)) * gap + rng.normal(0, 0.03 * xh)
                y = top + rng.normal(0, 0.04 * xh)
                ticks.append(np.array([[x + 0.15 * length, y], [x - 0.12 * length, y + length * rng.uniform(0.9, 1.1)]]))
            return ticks
        if g["symbol"] == "°":
            # A small loop at cap height, seldom quite closed.
            r = 0.2 * xh * rng.uniform(0.85, 1.15)
            a = rng.uniform(0, 2 * np.pi) + np.linspace(0, 2 * np.pi * rng.uniform(0.9, 1.08), 14)
            ring = r * (1 + rng.normal(0, 0.06, 14))
            return [np.stack([cx + ring * np.cos(a), base - 1.15 * xh + ring * np.sin(a)], 1)]
        axis = base - 0.55 * xh  # TeX's math axis sits about half an x-height up
        if g["symbol"] in TURNED_ARROWS:
            flat, degrees = TURNED_ARROWS[g["symbol"]]
            vertical = abs(np.sin(np.radians(degrees))) > 0.9
            # Its length is the glyph's height when it points up or down.
            size = (g["h"] + g["d"]) if vertical else g["w"]
            mid = base + 0.5 * (g["d"] - g["h"]) * self.px if vertical else axis
            strokes = self._arrow(dict(g, symbol=flat, w=size * 0.8), cx, mid, xh, rng)
            c, s_ = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
            return [np.stack([cx + c * (p[:, 0] - cx) + s_ * (p[:, 1] - mid), mid - s_ * (p[:, 0] - cx) + c * (p[:, 1] - mid)], 1) for p in strokes]
        if g["symbol"] in ARROWS:
            return self._arrow(g, cx, axis, xh, rng)
        # Writers make these about an x-height across, whatever size the font sets them.
        half = 0.5 * xh * self.symbol_size * rng.uniform(0.88, 1.12)

        def dash(y, a=-half, b=half):
            lean = rng.normal(0, 0.05 * xh)
            return np.array([[cx + a, y - lean], [cx + 0.1 * (a + b), y + rng.normal(0, 0.02 * xh)], [cx + b, y + lean]])

        if g["symbol"] == "-":
            return [dash(axis)]
        if g["symbol"] == "=":
            return [dash(axis - 0.2 * xh), dash(axis + 0.2 * xh, -half * rng.uniform(0.85, 1.0), half * rng.uniform(0.9, 1.1))]
        top, bottom = axis - half, axis + half
        return [dash(axis), np.array([[cx + rng.normal(0, 0.04 * xh), top], [cx + rng.normal(0, 0.04 * xh), bottom]])]

    def _arrow(self, g, cx, axis, xh, rng):
        """An arrow as a hand draws it: a shaft that bows and wavers a little, then a head
        flicked on in one V or two strokes, never quite meeting the shaft's end. How big
        and how open the heads are, and how they're drawn, is the writer's habit."""
        right, left, shafts, bar = ARROWS[g["symbol"]]
        habit = np.random.default_rng(subseed(self.pen.seed, "arrow habit"))
        head_len = xh * habit.uniform(0.42, 0.68)
        spread = np.radians(habit.uniform(24, 40)) + (0.12 if shafts == 2 else 0.0)
        one_stroke = habit.random() < 0.6
        length = max(g["w"] * self.px * habit.uniform(1.0, 1.3), 2.2 * xh) * rng.uniform(0.92, 1.08)
        a, b = cx - length / 2, cx + length / 2
        lean = rng.normal(0, 0.03) - 0.02  # shafts drift up a little to the right
        out = []
        gap = 0.2 * xh * rng.uniform(0.85, 1.15)
        offsets = [0.0] if shafts == 1 else [-gap, gap]
        for off in offsets:
            t = np.linspace(0, 1, 9)
            x = a + (b - a) * t + rng.normal(0, 0.03 * xh) * (t > 0.9)
            y = axis + off + lean * (x - cx) + rng.normal(0, 0.05 * xh) * np.sin(np.pi * t) + rng.normal(0, 0.012 * xh, len(t)).cumsum() * 0.5
            shorten = 0.18 * xh if shafts == 2 else 0.0  # a double shaft stops short of the head's tip
            x = np.clip(x, a + shorten * left, b - shorten * right)
            out.append(np.stack([x, y], 1))

        def head(tip_x, tip_y, direction):
            tip = np.array([tip_x + rng.normal(0, 0.05 * xh), tip_y + rng.normal(0, 0.05 * xh)])
            arms = []
            for side in (-1, 1):
                angle = spread * rng.uniform(0.85, 1.15) * side
                length = head_len * rng.uniform(0.85, 1.15) * (1.25 if shafts == 2 else 1.0)
                back = np.array([-direction * np.cos(angle), np.sin(angle)]) * length
                mid = tip + back * 0.5 + rng.normal(0, 0.03 * xh, 2)
                arms.append(np.stack([tip + back, mid, tip]))
            if one_stroke:
                return [np.concatenate([arms[0], arms[1][::-1][1:]])]
            return arms

        tip_y = axis + lean * (b - cx)
        if right:
            out += head(b, tip_y, 1)
        if left:
            out += head(a, axis + lean * (a - cx), -1)
        if bar:
            y0 = axis + lean * (a - cx)
            out.append(np.array([[a + rng.normal(0, 0.02 * xh), y0 - 0.3 * xh], [a, y0 + 0.3 * xh]]))
        return out

    def _draw_fallback(self, canvas, u, tilt, pen_px):
        for g in u.glyphs:
            cx = u.x + (g["x"] + g["w"] / 2 - u.x0) * self.px
            base = g["y"] * self.px + tilt(cx)
            xh = g["xh"] * self.px
            if g["t"] in BULLETS:
                r = 0.22 * xh
                canvas.stroke([[cx - r * 0.3, base - 0.5 * xh], [cx + r * 0.3, base - 0.5 * xh + 0.05 * xh]], 2.2 * r)
            else:
                canvas.glyph(g["t"], u.x + (g["x"] - u.x0) * self.px, base, g["size"] * self.px, self.math_font)

    def _draw_rule(self, canvas, r, line, tilt, pen_px):
        """Fraction bars and underlines, drawn freehand and moved with their math."""
        rng = np.random.default_rng(subseed(self.pen.seed, "rule", r["x"], r["y"]))
        if "span" in r:
            start, end = r["owner"].x + r["span"][0], r["owner"].x + r["span"][1]
        else:
            cx = r["x"] + r["w"] / 2
            owner = min(line["units"], key=lambda u: 0 if u.x0 <= cx <= u.x1 else min(abs(u.x0 - cx), abs(u.x1 - cx)))
            dx = owner.x - owner.x0 * self.px
            start, end = r["x"] * self.px + dx, (r["x"] + r["w"]) * self.px + dx
            if not owner.math:
                # \underline in text: under the words' own ink, along the slope they run at.
                under = [u for u in line["units"] if not u.fig and u.x < end and u.x + u.width > start]
                pts = np.concatenate([self._unit_points(u, tilt) for u in under] or [np.zeros((0, 2))])
                xh = owner.glyphs[0]["xh"] * self.px
                if len(pts):
                    slope, base = self._ink_baseline(pts, xh)
                    slope += rng.normal(0, 0.012)
                    x0, x1 = start - rng.uniform(0, 3) * self.px, end + rng.uniform(0, 4) * self.px
                    xs = np.linspace(x0, x1, 12)
                    ys = base + rng.uniform(0.35, 0.55) * xh + slope * xs + rng.normal(0, 0.6) * np.sin(np.linspace(0, np.pi, 12))
                    canvas.stroke(np.stack([xs, ys], 1), pen_px)
                    return
        x0 = start - rng.uniform(0, 3) * self.px
        x1 = end + rng.uniform(0, 4) * self.px
        y = (r["y"] + r["h"] / 2) * self.px
        xs = np.linspace(x0, x1, 12)
        bow = rng.normal(0, 0.6) * np.sin(np.linspace(0, np.pi, 12))
        ys = y + bow + rng.normal(0, 0.012) * (xs - x0) + np.array([tilt(x) for x in xs])
        canvas.stroke(np.stack([xs, ys], 1), pen_px)


def _smooth_path(ctrl, spacing):
    """A smooth curve through control points, resampled `spacing` apart (the ink then sees
    the speed of a hand moving fast but not flying)."""
    p = np.asarray(ctrl, dtype=np.float64)
    if len(p) < 3:
        return p
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = (np.arange(8) / 8)[:, None]
    a, b, c, d = q[:-3, None], q[1:-2, None], q[2:-1, None], q[3:, None]
    out = 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t**2 + (3 * b - a - 3 * c + d) * t**3)
    out = np.vstack([out.reshape(-1, 2), p[-1:]])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(out, axis=0).T))]
    n = max(3, int(s[-1] / max(spacing, 1e-6)))
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, out[:, 0]), np.interp(u, s, out[:, 1])], 1)


def _scribble(kind, x0, x1, level, xh, rng):
    """Control points of the strokes that black out the ink between x0 and x1 (centred on
    level(x)), as people do it in real notes:

    scrub   long loops scrubbed back and forth along the words, each at another height, the
            turns narrow and rounded, the ends overshooting unevenly; dense in the middle,
            looser at the edges; sometimes a tail flung off at the end
    coil    a chain of loops run along the words like a row of e's, sometimes twice
    tangle  a knot of overlapping curls scrawled over a short word
    """
    width = x1 - x0
    out = []
    if kind == "scrub":
        # Few enough passes that each line still shows, with paper between: a pen, not a
        # highlighter.
        sweeps = int(np.clip(round(rng.uniform(6, 9) + rng.uniform(0, 0.08) * width / xh), 6, 11))
        half = rng.uniform(0.55, 0.75) * xh
        lift = rng.uniform(-0.15, 0.05) * xh  # a little above the middle of the ink, usually
        tilt = rng.normal(0, 0.05) * xh / max(width, xh)
        # Over the top, back under the bottom: each pair of sweeps makes one long loop, and
        # the loops land at different heights, so the words are covered, not barred.
        ys, side = [], rng.choice([-1, 1])
        for _ in range(sweeps):
            ys.append(float(side * rng.uniform(0.1, 1.05) * half))
            side = -side if rng.random() < 0.8 else side
        right = rng.random() < 0.8  # most start on the left and sweep right first
        ctrl = []
        start = x0 - rng.uniform(0.1, 0.5) * xh
        for k, y in enumerate(ys):
            # Each sweep's own ends: overshooting the words by varying amounts, one now and
            # then turning back early.
            lo = x0 - rng.uniform(-0.15, 0.6) * xh
            hi = x1 + rng.uniform(-0.15, 0.7) * xh
            if rng.random() < 0.15:
                if (k % 2 == 0) == right:
                    hi = x0 + rng.uniform(0.55, 0.85) * width
                else:
                    lo = x0 + rng.uniform(0.15, 0.45) * width
            a, b = (lo, hi) if (k % 2 == 0) == right else (hi, lo)
            if k == 0:
                a = start if right else x1 + rng.uniform(0.1, 0.5) * xh
            n = max(3, int(abs(b - a) / (0.6 * xh)) + 2)
            xs = np.linspace(a, b, n)
            # No two sweeps parallel: each runs at its own slant and sways its own way, so
            # they cross one another at shallow angles.
            u = np.linspace(0, 1, n)
            own = rng.normal(0, 0.35) * half * (u - 0.5) * (1 if b > a else -1)
            sway = rng.uniform(0.12, 0.3) * half * np.sin(2 * np.pi * rng.uniform(0.4, 1.3) * u + rng.uniform(0, 6.3))
            bow = rng.normal(0, 0.12) * xh * np.sin(np.pi * u)
            yy = level(xs) + lift + y + bow + own + sway + tilt * (xs - 0.5 * (x0 + x1)) + rng.normal(0, 0.04 * xh, n)
            pts = np.stack([xs, yy], 1)
            if ctrl:
                # The turn: a narrow loop just past the end of the last sweep.
                prev = ctrl[-1]
                d = np.sign(prev[0] - ctrl[-2][0]) or 1.0  # the way the last sweep was going
                tip = [prev[0] + d * rng.uniform(0.08, 0.3) * xh, 0.5 * (prev[1] + pts[0, 1])]
                ctrl.append(tip)
                pts = pts[1:]
            ctrl.extend(pts.tolist())
        if rng.random() < 0.3:
            # Flung off at the end: a long tail past the words, falling or rising away.
            last = np.array(ctrl[-1])
            d = np.sign(last[0] - ctrl[-3][0]) or 1.0
            ctrl.append((last + [d * rng.uniform(0.6, 1.8) * xh, rng.choice([-1, 1]) * rng.uniform(0.3, 0.9) * xh]).tolist())
        ctrl = np.array(ctrl)
        if width > 14 * xh and rng.random() < 0.6:
            # A long scrub is done in two goes.
            cut = len(ctrl) // 2 + int(rng.integers(-3, 4))
            out += [ctrl[: cut + 1], ctrl[cut:]]
        else:
            out.append(ctrl)
    elif kind == "coil":
        for k in range(1 if rng.random() < 0.55 else 2):
            pitch = rng.uniform(0.6, 0.95) * xh
            rx, ry = rng.uniform(0.35, 0.55) * xh, rng.uniform(0.45, 0.7) * xh
            a, b = x0 - rng.uniform(0.1, 0.4) * xh, x1 + rng.uniform(0.0, 0.5) * xh
            if k:
                a, b = b, a
            loops = max(2, int(abs(b - a) / pitch))
            t = np.linspace(0, 2 * np.pi * loops, loops * 9 + 1)
            sx = 1 if b > a else -1
            r_x = rx * np.exp(rng.normal(0, 0.12, loops + 2))[np.minimum((t / (2 * np.pi)).astype(int), loops)]
            r_y = ry * np.exp(rng.normal(0, 0.15, loops + 2))[np.minimum((t / (2 * np.pi)).astype(int), loops)]
            xs = a + sx * (np.linspace(0, abs(b - a), len(t)) + r_x * np.sin(t))
            drift = rng.normal(0, 0.05 * xh, len(t)).cumsum() * 0.3
            off = (rng.uniform(-0.2, 0.2) * xh) if k else rng.uniform(-0.1, 0.05) * xh
            ys = level(xs) + off + r_y * np.cos(t) - 0.1 * xh + drift  # loops at the top, like e's
            out.append(np.stack([xs, ys], 1))
    else:
        # A knot: the pen hops about over the word, curling round at each place it lands,
        # so loops pile on loops; it goes over the word about twice, back and forth.
        half_h = rng.uniform(0.45, 0.65) * xh
        hand = rng.choice([-1, 1])
        ctrl = []
        curls = int(np.clip(7 + 2.2 * width / xh, 7, 18))
        for k in range(curls):
            # Along the word, then back along it: the hand sweeps over it while it curls.
            u = (k + rng.uniform(-0.4, 0.4)) / (curls - 1) * 2
            u = u if u <= 1 else 2 - u
            cx = x0 + float(np.clip(u, -0.05, 1.05)) * width
            cy = level(cx)[0] + rng.uniform(-1, 1) * half_h - 0.05 * xh
            r = rng.uniform(0.25, 0.5) * xh
            ry = r * rng.uniform(0.8, 1.5)
            start = rng.uniform(0, 2 * np.pi)
            if rng.random() < 0.2:
                hand = -hand
            for a in start + hand * np.linspace(0, rng.uniform(1.3, 2.0) * np.pi, 4):
                ctrl.append([cx + r * np.cos(a), cy + ry * np.sin(a)])
        out.append(np.array(ctrl))
    return out


def _roundish(p):
    """Whether a small closed shape is a round dot (a plot mark), not an arrow tip."""
    if len(p) < 10:
        return False
    d = np.hypot(*(p - p.mean(0)).T)
    return float(d.std() / (d.mean() + 1e-9)) < 0.12


def _hatch(poly, spacing, angle, rng, outlined=True):
    """Parallel-ish lines about `spacing` apart at `angle` across a closed polygon (N, 2), as
    a hand hatches: each line starts and stops on its own (mostly a little short of the edge
    where the pen comes down, past or short of it where it lifts), the hand's angle drifts
    every few lines, and now and then an area is filled zigzag, back and forth in one stroke.
    Returns polylines."""
    c, s = np.cos(angle), np.sin(angle)
    turn = np.array([[c, s], [-s, c]])
    centre = poly.mean(0)
    q = (poly - centre) @ turn.T
    a, b = q, np.roll(q, -1, axis=0)
    zigzag = rng.random() < 0.25
    # Against a drawn outline the pen stops near it; with nothing to stop at, the ends wander.
    spread = (0.4 if outlined else 1.0) * spacing
    lines = []
    y = q[:, 1].min() + spacing * rng.uniform(0.3, 0.7)
    drift, k = 0.0, 0
    while y < q[:, 1].max():
        if k % int(rng.integers(3, 6)) == 0:
            drift = 0.8 * drift + rng.normal(0, np.radians(1.5))
        k += 1
        cross = (a[:, 1] - y) * (b[:, 1] - y) < 0
        xs = np.sort(a[cross, 0] + (y - a[cross, 1]) * (b[cross, 0] - a[cross, 0]) / (b[cross, 1] - a[cross, 1]))
        row = []
        for x0, x1 in zip(xs[0::2], xs[1::2]):
            x0 = x0 + abs(rng.normal(0.15, 0.5)) * spread * (1 if rng.random() < 0.8 else -0.6)
            x1 = x1 + rng.normal(-0.1, 0.6) * spread
            if x1 - x0 > 0.4 * spacing:
                mid = 0.5 * (x0 + x1)
                half = 0.5 * (x1 - x0)
                dy = np.tan(drift) * half
                row.append(np.array([[x0, y - dy + rng.normal(0, 0.05) * spacing], [mid, y + rng.normal(0, 0.04) * spacing],
                                     [x1, y + dy + rng.normal(0, 0.05) * spacing]]))
        lines.append(row)
        y += spacing * rng.uniform(0.82, 1.18)
    out = []
    if zigzag and lines and all(len(r) <= 1 for r in lines):
        # One stroke: along a line, back along the next, joined at alternate ends.
        chain = []
        for i, r in enumerate(l for l in lines if l):
            seg = r[0] if i % 2 == 0 else r[0][::-1]
            chain.append(seg)
        if chain:
            out.append(np.concatenate(chain))
    else:
        out = [seg for r in lines for seg in r]
    return [p @ turn + centre for p in out]


def _by_hand(p, rng, mm, closed, steady=1.0):
    """A path as a hand traces it: even steps along it, a slow waver across it that grows
    with its length, a straight run bowed a little, and a closed shape that doesn't
    quite meet where it started."""
    seg = np.hypot(*np.diff(p, axis=0).T)
    length = seg.sum()
    if length < 1e-6:
        return p
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(length / (0.5 * mm)))
    t = np.linspace(0, length, n)
    q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
    tangent = np.gradient(q, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1) / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9)
    u = t / length
    amp = steady * min(0.35 * mm, 0.004 * length + 0.08 * mm)
    waver = sum(rng.normal(0, amp) * np.sin(2 * np.pi * f * u + rng.uniform(0, 6)) for f in (rng.uniform(0.6, 1.2), rng.uniform(1.5, 3.0)))
    waver += rng.normal(0, 0.03 * mm, n).cumsum() * 0.15
    waver *= np.sin(np.pi * u) ** 0.5 if not closed else 1.0
    q = q + normal * waver[:, None]
    if closed:
        q[-1] += rng.normal(0, 0.15 * mm, 2)
    else:
        q[-1] += (q[-1] - q[-2]) / (np.linalg.norm(q[-1] - q[-2]) + 1e-9) * rng.uniform(0, 0.4) * mm  # a little run past the end
    return q

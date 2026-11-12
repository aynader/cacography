"""A sheet of paper: what it is printed with, how it has been handled, and the ink on it.

The sheet is built flat, in its own millimetres, as three things the camera needs:
its colour (linear reflectance, ink included), its relief (height in millimetres, from
the grain up to folds), and its outline (holes and torn edges cut out). The camera
then bends it, lights it and photographs it.
"""
from dataclasses import dataclass

import numpy as np

from . import handling
from .imaging import band, blur, rgb, sample, smoothstep, wave

MM_PER_IN = 25.4

PAPERS = {
    # colour, rules, margin line(s), grain roughness (rms slope), holes, torn top edge
    "notebook": dict(colour=(246, 246, 240), rule=(112, 156, 208), rule_mm=0.20, margin=(226, 108, 122),
                     margins=(1.25,), rough=0.016, holes=True, torn=False),
    "wide": dict(colour=(246, 246, 240), rule=(112, 156, 208), rule_mm=0.20, margin=(226, 108, 122),
                 margins=(1.25,), rough=0.016, holes=True, torn=False),
    "legal": dict(colour=(250, 238, 160), rule=(120, 152, 188), rule_mm=0.18, margin=(214, 92, 98),
                  margins=(1.25, 1.31), rough=0.022, holes=False, torn=True),
    "graph": dict(colour=(245, 246, 242), rule=(150, 186, 212), rule_mm=0.14, margin=None,
                  margins=(), rough=0.014, holes=False, torn=False),
    "blank": dict(colour=(248, 248, 247), rule=None, rule_mm=0, margin=None,
                  margins=(), rough=0.010, holes=False, torn=False),
}

INKS = {
    # colour: one pass at the writer's usual pressure. build: how quickly more ink stops
    # adding darkness. pores: how much a light line misses the paper's pores. gloss: its shine.
    "blue": dict(colour=(38, 50, 132), build=0.9, pores=0.25, graphite=False, gloss=0.5),
    "black": dict(colour=(40, 40, 48), build=0.9, pores=0.25, graphite=False, gloss=0.5),
    "red": dict(colour=(178, 40, 52), build=0.9, pores=0.25, graphite=False, gloss=0.5),
    "pencil": dict(colour=(92, 92, 98), build=0.6, pores=0.0, graphite=True, gloss=1.0),
}


@dataclass
class Sheet:
    albedo: np.ndarray  # H x W x 3, linear reflectance
    relief: np.ndarray  # H x W, height in mm of the grain and creases, for shading
    mask: np.ndarray  # H x W, 1 where there is paper
    bend: np.ndarray  # coarse height in mm, one sample per BEND_MM, for geometry
    grain: np.ndarray  # H x W, the paper's fine surface height, in standard deviations
    ppmm: float
    width_mm: float
    height_mm: float
    sheen: np.ndarray = None  # H x W, how glossy the ink on it is, set by lay_ink
    grooves: float = 0.0  # how deep the pen pressed its strokes in, 0..1
    ghost: object = None  # random numbers for the grooves left by writing on the sheet above, or None
    smudge: object = None  # random numbers for a smear of ink, or None
    backside: object = None  # random numbers for writing on the back that shows through, or None
    age: float = 0.0  # how old the sheet is, 0..1: its ink faded
    wear: float = 0.3  # how handled the sheet is, which makes props likelier
    ink: str = "blue"  # which ink it is written in (set by lay_ink), for the pen lying by it
    erasures: list = None  # where words were rubbed out, (x0, y0, x1, y1) px, set before lay_ink
    rubbed: np.ndarray = None  # the rubbed-out words' own ink (their grooves stay in the paper)

    BEND_MM = 2.0

    def lift(self, x_mm, y_mm):
        """Height of the sheet above the desk at these points (extends past the edges)."""
        return sample(self.bend, x_mm / self.BEND_MM, y_mm / self.BEND_MM)

    def _rub(self, graphite):
        """Where a word was rubbed out: the paper a little grey with spread graphite (or
        scuffed paler, under an ink eraser), its fibres roughed up, streaked along the
        line the way an eraser goes back and forth, and the old strokes' grooves still in it."""
        h, w = self.relief.shape
        rng = np.random.default_rng(int(abs(self.relief[h // 2, w // 2]) * 1e6) % (2**31))
        patch = np.zeros((h, w), np.float32)
        for x0, y0, x1, y1 in self.erasures:
            i0, i1, j0, j1 = max(0, int(y0)), min(h, int(y1) + 1), max(0, int(x0)), min(w, int(x1) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            ys, xs = np.mgrid[i0:i1, j0:j1].astype(np.float32)
            cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
            ry, rx = 0.5 * (y1 - y0), 0.5 * (x1 - x0)
            d = np.maximum(np.abs(xs - cx) / rx, np.abs(ys - cy) / ry)
            patch[i0:i1, j0:j1] = np.maximum(patch[i0:i1, j0:j1], smoothstep(1.0, 0.65, d))
        if not patch.any():
            return
        streak = np.clip(0.6 + 0.4 * band((h, w), rng, 0.4 * self.ppmm, 4 * self.ppmm, stretch=5.0), 0, 1.3)
        patch = blur(patch, 0.8 * self.ppmm) * streak
        if graphite:
            self.albedo *= (1 - rng.uniform(0.035, 0.06) * patch)[..., None] * np.array([1.0, 1.0, 1.01], np.float32)
        else:
            self.albedo = self.albedo + (np.array([0.95, 0.95, 0.93], np.float32) - self.albedo) * (0.25 * patch)[..., None]
        self.relief += 0.006 * patch * band((h, w), rng, 0.15 * self.ppmm, 0.6 * self.ppmm)
        if self.rubbed is not None:
            self.relief -= (0.01 if graphite else 0.025) * blur(np.clip(self.rubbed, 0, 1.5), 0.6 * self.ppmm / 7.87)

    def lay_ink(self, amount, ink, rng):
        """Colour the sheet with `amount` of ink per pixel (see ink.py).

        Ink absorbs light, so where it lies thicker, at crossings and pressed-down strokes,
        it is darker and more saturated, and light strokes are paler.
        """
        spec = INKS.get(ink, INKS["blue"])
        self.ink = ink
        # No two pens hold quite the same ink: a slight shift in hue and strength.
        absorb = -np.log(np.clip(rgb(spec["colour"]), 1e-4, 1)) * np.exp(rng.normal(0, 0.05, 3)).astype(np.float32)
        a = amount
        if self.erasures:
            self._rub(spec["graphite"])
        if self.smudge is not None:
            a = a + handling.smudge(a, self.smudge, self.ppmm, spec["graphite"])
        if self.grooves > 0:
            # A ballpoint presses its line into the paper; a pencil less so. Writing on the
            # sheet above leaves fainter grooves of other words (here: this page's own
            # writing turned around and moved, which reads as nothing in particular).
            depth = self.grooves * (0.012 if spec["graphite"] else 0.03)
            self.relief -= depth * blur(np.clip(amount, 0, 1.5), 0.6 * self.ppmm / 7.87)
            if self.ghost is not None:
                g = self.ghost
                other = np.roll(amount[::-1, ::-1], (int(g.uniform(0.2, 0.8) * self.ppmm * 7), int(g.uniform(-40, 40) * self.ppmm)), (0, 1))
                self.relief -= depth * g.uniform(0.2, 0.45) * blur(np.clip(other, 0, 1.5), 1.2 * self.ppmm / 7.87)
        if spec["graphite"]:
            # Graphite catches on the paper's high points; pressing harder reaches the hollows.
            # The lead is wider than the grain, so the hollows thin the line rather than gap it.
            reach = 0.9 - 1.2 * np.clip(a, 0, 2)
            a = blur(a, 0.5) * (0.45 + 0.55 * smoothstep(reach - 1.8, reach + 1.8, blur(self.grain, 0.6)))
        else:
            # A light ballpoint line misses the deepest pores.
            a = a * (1 - spec["pores"] * smoothstep(1.2, 2.5, -self.grain) * np.exp(-a))
        if self.backside is not None:
            # Paper lets a little light through: writing on the back shows, mirrored,
            # soft and paler, in the ink's colour gone grey, and the pen pressed it in from
            # behind, so it stands up a little on this side. Here it is this page's
            # writing, mirrored and moved by part of a line, which reads as other words.
            g = self.backside
            back = np.roll(amount[:, ::-1], (int(g.uniform(0.25, 0.75) * 7.1 * self.ppmm), int(g.uniform(-20, 20) * self.ppmm)), (0, 1))
            back = blur(np.clip(back, 0, 1.5), 0.4 * self.ppmm)
            tint = 0.5 * absorb / absorb.mean() + 0.5
            self.albedo *= np.exp(-g.uniform(0.15, 0.28) * back[..., None] * tint)
            if not spec["graphite"]:
                self.relief += g.uniform(0.004, 0.012) * blur(back, 0.3 * self.ppmm)
        if self.age > 0:
            # Old ink fades and loses its colour, ballpoint more than graphite.
            fade = self.age * (0.15 if spec["graphite"] else 0.4)
            absorb = absorb * (1 - fade) + fade * 0.5 * absorb.mean()
        thick = (1 - np.exp(-spec["build"] * a)) / (1 - np.exp(-spec["build"]))
        self.albedo *= np.exp(-thick[..., None] * absorb)
        # Ballpoint ink dries with an oily shine; graphite shines more.
        self.sheen = (spec["gloss"] * np.clip(thick, 0, 1)).astype(np.float32)


def make_sheet(kind, width_pt, height_pt, dpi, rules_from_px, rule_step_px, wear, aspects, page):
    """A fresh sheet `kind` (see PAPERS), then `wear` (0..1) worth of handling.

    Crumple, folds, creases, scores, edges, stains and pen grooves each follow their own
    aspect (aspects.py, handling.py); where one is left at random, `wear` decides how
    likely it is.
    """
    rng = aspects.rng("paper", page)
    spec = PAPERS.get(kind, PAPERS["notebook"])
    ppmm = dpi / MM_PER_IN
    w = int(round(width_pt / 72.27 * dpi))
    h = int(round(height_pt / 72.27 * dpi))
    wmm, hmm = w / ppmm, h / ppmm
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # The paper itself: cloudy where fibres clumped, finer speckle, a faint tint per sheet.
    base = rgb(spec["colour"]) * (1 + rng.normal(0, 0.006, 3)).astype(np.float32)
    machine = rng.normal(np.pi / 2, 0.05)  # fibres run down the sheet
    formation = band((h, w), rng, 0.8 * ppmm, 14 * ppmm, slope=0.8, stretch=1.6, angle=machine)
    speck = band((h, w), rng, 0.12 * ppmm, 0.8 * ppmm, slope=0.3, stretch=2.5, angle=machine)
    shade = 1 + 0.010 * formation + 0.006 * speck
    albedo = base[None, None, :] * shade[..., None]
    albedo *= (1 + 0.004 * band((h, w), rng, 3 * ppmm, 30 * ppmm)[..., None] * np.array([1.0, 0.4, -0.8], np.float32))

    # Grain relief: scaled to the paper's measured roughness (rms surface slope).
    grain = band((h, w), rng, 0.08 * ppmm, 0.6 * ppmm, slope=0.5, stretch=1.8, angle=machine)
    gy, gx = np.gradient(grain)
    grain *= spec["rough"] / (np.sqrt(np.mean(gx**2 + gy**2)) * ppmm + 1e-9)

    _print_rules(albedo, spec, kind, xs, ys, ppmm, rules_from_px, rule_step_px, w, h, rng)

    # Handling, each part from its own aspect. Heights go into one field at full resolution:
    # its broad part bends the mesh, and the rest becomes the normal map.
    H = np.zeros((h, w), np.float32)
    crumple_rng = aspects.rng("crumple", page)
    level = handling.amount(aspects.value("crumple"), crumple_rng, 0.25 + 0.5 * wear, 0.15, 0.5 + 0.5 * wear)
    if level > 0:
        H += handling.on_sheet(handling.crumple(level, wmm, hmm, crumple_rng), (h, w), ppmm)
        # Fibres break pale along a ridge; a valley gathers a little grime.
        ridge, valley = handling.crease_lines(H, ppmm)
        albedo *= (1 + 0.045 * np.sqrt(level) * ridge - 0.03 * (0.3 + wear) * np.sqrt(level) * valley)[..., None]

    mask = np.ones((h, w), np.float32)
    edges = handling.edges(aspects.value("edges"), wear, mask, albedo, H, ppmm, aspects.rng("edges", page))
    if spec["holes"] and edges != "spiral":
        _punch(mask, xs, ys, ppmm, wmm, hmm, rng)
    if spec["torn"] and edges != "torn":
        tear = (0.25 * np.abs(np.sin(np.pi * xs[0] / (1.0 * ppmm))) + 0.18 * wave(w, rng, 0.3 * ppmm, 6 * ppmm)) * ppmm
        mask *= np.clip(ys - (0.6 * ppmm + tear[None, :]), 0, 1)

    # Folds and creases: a sharp kink with a crisp line where the fibres broke, pale on
    # a ridge, darker where grime settles in a valley.
    folds = _folds(aspects.value("folds"), wear, wmm, hmm, aspects.rng("folds", page))
    for crease in folds + _small_creases(aspects.value("creases"), wear, wmm, hmm, aspects.rng("creases", page)):
        H += _crease_height(crease, xs / ppmm, ys / ppmm)
        line = _crease_line(crease, xs / ppmm, ys / ppmm)
        dirt = crease["dirt"] * line * (0.6 + 0.4 * np.tanh(band((h, w), rng, 0.5 * ppmm, 8 * ppmm)))
        albedo *= (1 - dirt[..., None] * np.array([0.9, 1.0, 1.15], np.float32))
        if crease["depth"] < 0:
            albedo *= (1 + 0.03 * line)[..., None]
    score_rng = aspects.rng("scores", page)
    value = aspects.value("scores")
    count = 0 if value == "none" else score_rng.poisson(0.8 * wear) if value == "random" else int(float(value))
    for s in handling.scores(count, wmm, hmm, score_rng):
        handling.draw_score(H, albedo, ppmm, s)
    _grime(albedo, wear, xs, ys, ppmm, w, h, rng)
    stain_rng = aspects.rng("stains", page)
    stains = handling.choose_stains(aspects.value("stains"), wear, stain_rng, False)
    if "coffee" in stains:
        handling.coffee(albedo, H, xs / ppmm, ys / ppmm, ppmm, wmm, hmm, stain_rng)
    if "water" in stains:
        handling.water(albedo, H, ppmm, wmm, hmm, stain_rng)
    edge = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys)) / ppmm
    age = (0.01 + 0.04 * wear) * np.exp(-edge / 3.0)
    albedo *= (1 - age[..., None] * np.array([0.6, 0.9, 1.6], np.float32))
    albedo *= 1 - 0.03 * wear * np.array([0.0, 0.2, 0.7], np.float32)  # yellowing
    age_rng = aspects.rng("age", page)
    old = handling.amount(aspects.value("age"), age_rng, 0.1 + 0.25 * wear, 0.15, 0.4 + 0.5 * wear)
    if old > 0.05:
        handling.age(old, albedo, ppmm, wmm, hmm, age_rng)

    cell = Sheet.BEND_MM
    bh, bw = int(np.ceil(hmm / cell)) + 1, int(np.ceil(wmm / cell)) + 1
    by, bx = np.mgrid[0:bh, 0:bw].astype(np.float32) * cell
    broad = blur(H, 0.6 * cell * ppmm)
    bend = sample(broad, bx * ppmm, by * ppmm).astype(np.float32) + _curl(bx, by, wmm, hmm, wear, rng)
    curl_rng = aspects.rng("curl", page)
    bend += handling.curl(handling.amount(aspects.value("curl"), curl_rng, 0.15 + 0.3 * wear, 0.2, 0.8), bx, by, wmm, hmm, curl_rng)
    bend -= bend[: int(hmm / cell), : int(wmm / cell)].min()
    groove_rng = aspects.rng("grooves", page)
    grooves = handling.amount(aspects.value("grooves"), groove_rng, 1.0, 0.3, 1.0)
    return Sheet(albedo.astype(np.float32), (H - broad + grain).astype(np.float32), mask, bend,
                 (grain / (grain.std() + 1e-9)).astype(np.float32), ppmm, wmm, hmm,
                 grooves=grooves, ghost=groove_rng if groove_rng.random() < 0.4 else None,
                 smudge=stain_rng if "smudge" in stains else None, backside=_backside(aspects, page), age=old, wear=wear)


def _print_rules(albedo, spec, kind, xs, ys, ppmm, first_px, step_px, w, h, rng):
    """Offset-printed lines: thin, straight, with a little unevenness in the ink."""
    if spec["rule"] is None:
        return
    ink = rgb(spec["rule"])
    width = spec["rule_mm"] * ppmm
    cover = np.zeros((h, w), np.float32)
    if kind == "graph":
        cellpx, border = 5.0 * ppmm, 6.0 * ppmm
        inside = (xs >= border - 1) & (xs <= w - border + 1) & (ys >= border - 1) & (ys <= h - border + 1)
        for coord in (xs - border, ys - border):
            d = np.abs((coord + cellpx / 2) % cellpx - cellpx / 2)
            cover = np.maximum(cover, np.clip(width / 2 + 0.5 - d, 0, 1))
        cover *= inside
    else:
        y0 = first_px - np.floor((first_px - 1.0 * MM_PER_IN * ppmm) / step_px) * step_px
        last = h - 0.5 * MM_PER_IN * ppmm
        row = ys[:, 0]
        d = np.abs((row - y0 + step_px / 2) % step_px - step_px / 2)
        rows = np.clip(width / 2 + 0.5 - d, 0, 1) * ((row > y0 - step_px / 2) & (row < last))
        cover = np.broadcast_to(rows[:, None], (h, w)).copy()
    cover *= 0.9 + 0.1 * np.tanh(band((h, w), rng, 0.4 * ppmm, 20 * ppmm))
    albedo *= (1 - cover[..., None] * (1 - ink))
    for inches in spec["margins"]:
        x0 = inches * MM_PER_IN * ppmm + rng.normal(0, 0.15) * ppmm
        m = np.clip(0.25 * ppmm / 2 + 0.5 - np.abs(xs[0] - x0), 0, 1)[None, :]
        m = m * (0.9 + 0.1 * np.tanh(band((h, w), rng, 0.4 * ppmm, 20 * ppmm)))
        albedo *= (1 - m[..., None] * (1 - rgb(spec["margin"])))


def _punch(mask, xs, ys, ppmm, wmm, hmm, rng):
    """Binder holes: three on US letter, two (80 mm apart) on anything else."""
    if abs(hmm - 279.4) < 3:
        centres, radius, inset = [hmm / 2 - 108, hmm / 2, hmm / 2 + 108], 3.2, 9.5
    else:
        centres, radius, inset = [hmm / 2 - 40, hmm / 2 + 40], 3.0, 12.0
    for cy in centres:
        cx, cy = inset + rng.normal(0, 0.15), cy + rng.normal(0, 0.15)
        d = np.hypot(xs - cx * ppmm, ys - cy * ppmm) / ppmm
        mask *= np.clip((d - radius) * ppmm + 0.5, 0, 1)


def _folds(choice, wear, wmm, hmm, rng):
    """Folded in half, in thirds (for an envelope) or in quarters, then flattened out."""
    if choice == "none" or (choice == "random" and rng.random() >= 0.7 * wear):
        return []
    if choice == "random":
        choice = "half" if rng.random() < 0.5 else "thirds"
    lines = []  # (across the sheet?, how many parts)
    if choice == "quarters":
        lines = [(True, 2), (False, 2)]
    else:
        lines = [(rng.random() < 0.75, 2 if choice == "half" else 3)]
    out = []
    for across, parts in lines:
        for i in range(1, parts):
            pos = (hmm if across else wmm) * i / parts + rng.normal(0, 4)
            angle = rng.normal(0, 0.012) + (0 if across else np.pi / 2)
            point = (wmm / 2, pos) if across else (pos, hmm / 2)
            # A flattened fold still bends a few degrees each side of the line.
            reach = rng.uniform(15, 40)
            out.append(dict(point=point, angle=angle, length=None, depth=reach * rng.uniform(0.04, 0.1) * rng.choice([-1, 1]),
                            reach=reach, dirt=rng.uniform(0.015, 0.04)))
    return out


def _small_creases(choice, wear, wmm, hmm, rng):
    """Short creases from handling: `choice` many, or with random, more on a worn sheet."""
    count = 0 if choice == "none" else rng.poisson(2.5 * wear) if choice == "random" else int(float(choice))
    out = []
    for _ in range(count):
        reach = rng.uniform(3, 10)
        out.append(dict(point=(rng.uniform(0, wmm), rng.uniform(0, hmm)), angle=rng.uniform(0, np.pi),
                        length=rng.uniform(15, 90), depth=reach * rng.uniform(0.03, 0.1) * rng.choice([-1, 1]),
                        reach=reach, dirt=rng.uniform(0.0, 0.02)))
    return out


def _crease_frame(c, x, y):
    px, py = c["point"]
    ca, sa = np.cos(c["angle"]), np.sin(c["angle"])
    along = (x - px) * ca + (y - py) * sa
    across = -(x - px) * sa + (y - py) * ca
    if c["length"] is None:
        fade = np.ones_like(along)
    else:
        fade = smoothstep(c["length"] / 2, c["length"] / 2 - min(10.0, c["length"] / 3), np.abs(along))
    return across, fade


def _crease_height(c, x, y):
    """A flattened fold: a sharp kink along the line that relaxes away from it."""
    across, fade = _crease_frame(c, x, y)
    r = np.sqrt(across**2 + 0.15**2)
    return (c["depth"] * (1 - np.exp(-r / c["reach"])) * fade).astype(np.float32)


def _crease_line(c, x, y):
    across, fade = _crease_frame(c, x, y)
    return (np.exp(-((across / 0.14) ** 2)) * fade).astype(np.float32)


def _curl(x, y, wmm, hmm, wear, rng):
    """Loose paper never lies flat: a gentle roll, some waviness, a lifted corner or two."""
    angle = rng.uniform(0, np.pi)
    u = ((x - wmm / 2) * np.cos(angle) + (y - hmm / 2) * np.sin(angle)) / (0.5 * max(wmm, hmm))
    z = rng.uniform(0.5, 3.0) * u**2
    z += rng.uniform(0.2, 0.5 + 0.8 * wear) * band(x.shape, rng, 60 / Sheet.BEND_MM, 300 / Sheet.BEND_MM)
    for _ in range(rng.binomial(2, 0.25 + 0.4 * wear)):
        cx, cy = rng.choice([0, wmm]), rng.choice([0, hmm])
        reach = rng.uniform(30, 80)
        d = np.hypot(x - cx, y - cy)
        z += rng.uniform(1.5, 4 + 6 * wear) * smoothstep(reach, 0, d) ** 2
    return z.astype(np.float32)


def _grime(albedo, wear, xs, ys, ppmm, w, h, rng):
    """Faint grey where hands rested and held the sheet: along the edges and the bottom."""
    if wear <= 0:
        return
    patches = smoothstep(0.6, 2.2, band((h, w), rng, 6 * ppmm, 60 * ppmm))
    edge = np.minimum(np.minimum(xs, w - 1 - xs), h - 1 - ys) / ppmm
    where = np.exp(-edge / 25.0) + 0.3
    amount = 0.035 * wear * patches * where
    albedo *= (1 - amount[..., None] * np.array([0.95, 1.0, 1.08], np.float32))


def _backside(aspects, page):
    """Random numbers for writing on the back of the sheet, or None for a blank back."""
    rng = aspects.rng("backside", page)
    value = aspects.value("backside")
    if value == "none" or (value == "random" and rng.random() >= 0.35):
        return None
    return rng

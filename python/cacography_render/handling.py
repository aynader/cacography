"""How a sheet has been handled: crumpled, scored, worn at the edges, stained.

Each kind of handling is its own aspect (aspects.py) with its own seed, so a page can be
pristine or well used one thing at a time. Everything works in the sheet's millimetres.

Heights all go into one field at the sheet's resolution. paper.make_sheet splits it: the
broad part bends the 3D mesh, the fine part becomes the normal map, so a crease is
counted once. What a crease does to the colour (fibres broken pale on a ridge, dirt
caught in a valley) is returned beside it.

Crumpled paper smoothed flat again is a network of nearly straight creases meeting at
points, with flat facets between them that each catch the light a little differently.
It is drawn that way here, by splitting the sheet into ever smaller pieces the way
repeated folding does (see crumple).
"""
import numpy as np

from .imaging import band, blur, sample, smoothstep, wave

WORK_MM = 0.25  # crumple is drawn on a grid this fine, then resampled to the sheet
A4 = 210.0 * 297.0


def amount(value, rng, odds, low, high):
    """0..1 from an aspect's value: a number as given, 0 for none, or at random.

    At random, a sheet gets `low` to `high` with chance `odds`, otherwise a light touch.
    """
    if value == "none":
        return 0.0
    if value != "random":
        return float(np.clip(float(value), 0.0, 1.0))
    if rng.random() < odds:
        return float(rng.uniform(low, high))
    return float(rng.uniform(0.0, 0.08))


def _window(shape, ppmm, x0, y0, x1, y1):
    """The part of a grid (`ppmm` cells per mm) covering [x0, x1] x [y0, y1] mm, and its coordinates."""
    h, w = shape
    i0, i1 = max(0, int(np.floor(y0 * ppmm))), min(h, int(np.ceil(y1 * ppmm)) + 1)
    j0, j1 = max(0, int(np.floor(x0 * ppmm))), min(w, int(np.ceil(x1 * ppmm)) + 1)
    if i0 >= i1 or j0 >= j1:
        return None
    Y = (np.arange(i0, i1, dtype=np.float32) / ppmm)[:, None]
    X = (np.arange(j0, j1, dtype=np.float32) / ppmm)[None, :]
    return (slice(i0, i1), slice(j0, j1)), X, Y


# Crumple --------------------------------------------------------------------

def crumple(level, wmm, hmm, rng):
    """Height (mm) on a WORK_MM grid of a sheet crumpled `level` (0..1) and smoothed flat.

    Crushing a sheet folds it again and again, each new fold running across a piece the
    earlier ones made and stopping where it meets them. So the sheet is split that way
    here: a straight crease across the whole sheet, then one across each half, and so on,
    each newer crease shorter and fainter, until the pieces are as small as the crushing
    was hard. Every crease runs from one older crease to another, never stopping in the
    open. Each crease tilts the paper on either side of it, and the heights come from all
    the tilts together, so the bulges sit between the creases instead of on top of them.
    """
    gh, gw = int(np.ceil(hmm / WORK_MM)) + 2, int(np.ceil(wmm / WORK_MM)) + 2
    H = np.zeros((gh, gw), np.float32)
    if level <= 0:
        return H
    # 1 is a sheet balled up hard once and smoothed out; the build below goes further
    # than real paper ever looked (you judged its 0.5 the most a sheet should get).
    level = 0.5 * level
    ppmm = 1 / WORK_MM
    # The paper bends a little as it is folded, so no crease is ruler-straight.
    warp = 0.7 + 0.5 * level
    X = (np.arange(gw, dtype=np.float32) * WORK_MM)[None, :] + warp * band((gh, gw), rng, 30 * ppmm, 120 * ppmm)
    Y = (np.arange(gh, dtype=np.float32) * WORK_MM)[:, None] + warp * band((gh, gw), rng, 30 * ppmm, 120 * ppmm)
    gx, gy = np.zeros_like(H), np.zeros_like(H)
    kx, ky = np.zeros_like(H), np.zeros_like(H)  # the sharp bend right at each crease

    smallest = 6 + 50 * (1 - level) ** 1.5  # mm across the smallest pieces
    whole = np.sqrt(wmm * hmm)
    pad = 3.0
    pieces = [(np.array([[-pad, -pad], [wmm + pad, -pad], [wmm + pad, hmm + pad], [-pad, hmm + pad]]), 0)]
    while pieces:
        poly, gen = pieces.pop()
        centre, axes, spread = _shape(poly)
        size = 2 * np.sqrt(spread[0] * spread[1]) * 1.7
        if size < smallest * rng.uniform(0.75, 1.5) or (gen > 2 and rng.random() < 0.02 + 0.12 * (1 - level)):
            continue
        # Across the piece's long side, mostly, so the pieces stay chunky.
        long_side = np.arctan2(axes[1, 0], axes[0, 0])
        angle = long_side + np.pi / 2 + rng.normal(0, 0.45)
        through = centre + axes @ (rng.normal(0, 0.3, 2) * spread)
        normal = np.array([-np.sin(angle), np.cos(angle)])
        a, b = _split(poly, through, normal)
        if a is None:
            continue
        tilt = (size / whole) ** 0.4 * np.exp(rng.normal(0, 0.35)) * rng.choice([-1.0, 1.0])
        sharp = rng.uniform(0.15, 0.45)  # mm over which the paper turns at the crease
        lo, hi = poly.min(0), poly.max(0)
        win = _window(H.shape, ppmm, lo[0], lo[1], hi[0], hi[1])
        if win is not None:
            where, _, _ = win
            Xw, Yw = X[where], Y[where]
            inside = _inside(poly, Xw, Yw)
            d = (Xw - through[0]) * normal[0] + (Yw - through[1]) * normal[1]
            turn = np.tanh(d / sharp) * inside
            gx[where] += tilt * turn * normal[0]
            gy[where] += tilt * turn * normal[1]
            # The fold itself stays bent tighter than the facets either side of it.
            k = tilt * rng.uniform(1.0, 2.5) * turn * np.exp(-np.abs(d) / rng.uniform(0.8, 2.0))
            kx[where] += k * normal[0]
            ky[where] += k * normal[1]
        pieces += [(a, gen + 1), (b, gen + 1)]

    steep = np.sqrt(np.mean(gx * gx + gy * gy)) + 1e-9
    scale = (0.04 + 0.13 * level) / steep  # rms tilt, radians
    # Laid on a desk, its own weight presses the widest bulges down.
    H = _integrate((gx + kx) * scale, (gy + ky) * scale, WORK_MM, flatten=(0.6, 45.0))
    # No facet is quite flat: the paper between creases bows a little this way and that.
    H += (0.03 + 0.1 * level) * blur(band(H.shape, rng, 15 * ppmm, 60 * ppmm), 3 * ppmm)
    return (H - H.mean()).astype(np.float32)


def _shape(poly):
    """A convex polygon's centre, principal axes (columns) and spread along each (mm)."""
    centre = poly.mean(0)
    q = poly - centre
    vals, vecs = np.linalg.eigh(q.T @ q / len(q))
    order = vals.argsort()[::-1]
    return centre, vecs[:, order], np.sqrt(np.maximum(vals[order], 1e-6))


def _split(poly, point, normal):
    """Cut a convex polygon along the line through `point` across `normal`: (one side, other side)."""
    d = (poly - point) @ normal
    sides = ([], [])
    n = len(poly)
    for i in range(n):
        p, q, dp, dq = poly[i], poly[(i + 1) % n], d[i], d[(i + 1) % n]
        (sides[0] if dp >= 0 else sides[1]).append(p)
        if (dp >= 0) != (dq >= 0):
            cut = p + (q - p) * (dp / (dp - dq))
            sides[0].append(cut)
            sides[1].append(cut)
    if len(sides[0]) < 3 or len(sides[1]) < 3:
        return None, None
    return np.array(sides[0]), np.array(sides[1])


def _inside(poly, X, Y):
    """1 where (X, Y) lies inside a convex polygon, 0 outside."""
    e, f = poly[1] - poly[0], poly[2] - poly[1]
    sign = np.sign(e[0] * f[1] - e[1] * f[0])
    inside = np.ones(np.broadcast(X, Y).shape, bool)
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        inside &= sign * ((q[0] - p[0]) * (Y - p[1]) - (q[1] - p[1]) * (X - p[0])) >= 0
    return inside.astype(np.float32)


def _integrate(gx, gy, step, flatten=(0.0, 1.0)):
    """The height whose slopes come closest to (gx, gy), by FFT on a mirrored copy.

    `flatten` (share, mm) takes that share off the bulges wider than about that.
    """
    h, w = gx.shape
    fx = np.concatenate([gx, -gx[:, ::-1]], 1)
    fx = np.concatenate([fx, fx[::-1]], 0)
    fy = np.concatenate([gy, gy[:, ::-1]], 1)
    fy = np.concatenate([fy, -fy[::-1]], 0)
    kx = 2 * np.pi * np.fft.fftfreq(2 * w, step)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(2 * h, step)[:, None]
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1
    Z = (-1j * kx * np.fft.fft2(fx) - 1j * ky * np.fft.fft2(fy)) / k2
    Z[0, 0] = 0
    share, wide = flatten
    Z *= 1 - share * np.exp(-0.5 * k2 * wide * wide)
    return np.real(np.fft.ifft2(Z))[:h, :w].astype(np.float32)


def on_sheet(H, shape, ppmm):
    """A WORK_MM grid resampled onto the sheet's pixels."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return sample(H, xs / ppmm / WORK_MM, ys / ppmm / WORK_MM).astype(np.float32)


def crease_lines(H, ppmm):
    """Where the height kinks: (ridges, valleys), each 0..1, for colouring crease lines."""
    gy, gx = np.gradient(H, 1 / ppmm)
    curv = np.gradient(gx, 1 / ppmm, axis=1) + np.gradient(gy, 1 / ppmm, axis=0)
    curv = blur(curv, 0.6)
    scale = np.percentile(np.abs(curv), 99.5) + 1e-9
    return np.clip(-curv / scale, 0, 1), np.clip(curv / scale, 0, 1)


# Scores ---------------------------------------------------------------------

def scores(count, wmm, hmm, rng):
    """Thin pressed lines: a fingernail run along a ruler, the edge of a sheet pressed in."""
    out = []
    for _ in range(count):
        length = rng.uniform(15, 90)
        out.append(dict(point=rng.uniform([0, 0], [wmm, hmm]), angle=rng.uniform(0, np.pi), length=length,
                        depth=rng.uniform(0.015, 0.05), width=rng.uniform(0.1, 0.22), bow=rng.normal(0, 0.003),
                        dark=rng.uniform(0.01, 0.04)))
    return out


def draw_score(H, albedo, ppmm, s):
    px, py = s["point"]
    ca, sa = np.cos(s["angle"]), np.sin(s["angle"])
    half = s["length"] / 2
    pad = 1.0 + abs(s["bow"]) * half * half
    win = _window(H.shape, ppmm, px - half * abs(ca) - pad, py - half * abs(sa) - pad, px + half * abs(ca) + pad, py + half * abs(sa) + pad)
    if win is None:
        return
    where, X, Y = win
    along = (X - px) * ca + (Y - py) * sa
    across = -(X - px) * sa + (Y - py) * ca - s["bow"] * along * along
    fade = smoothstep(half, half - min(6.0, half / 3), np.abs(along))
    line = np.exp(-((across / s["width"]) ** 2)) * fade
    H[where] -= (s["depth"] * line).astype(np.float32)
    albedo[where] *= (1 - s["dark"] * line[..., None]).astype(np.float32)


# Edges ----------------------------------------------------------------------

def _ragged(n, rng, ppmm, base, rough):
    """How far in (mm) a torn edge runs, along `n` pixels: a slow wander and a fine fray."""
    return (base + rough * (0.8 * wave(n, rng, 8 * ppmm, 80 * ppmm) + 0.25 * wave(n, rng, 0.4 * ppmm, 4 * ppmm))
            + 0.08 * wave(n, rng, 0.1 * ppmm, 0.6 * ppmm)).astype(np.float32)


def edges(choice, wear, mask, albedo, H, ppmm, rng):
    """Cut the outline and mark the edges: clean, worn, torn (one side) or torn from a spiral binding.

    Returns the choice made, so the caller can leave out binder holes on a spiral page.
    """
    if choice == "random":
        choice = "worn" if rng.random() < 0.25 + 0.6 * wear else "clean"
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) / ppmm
    wmm, hmm = w / ppmm, h / ppmm
    if choice == "clean":
        return choice
    if choice == "worn":
        # Softened corners, a few nicks, a fraying line, a corner turned down and back.
        for cx, cy in ((0, 0), (wmm, 0), (0, hmm), (wmm, hmm)):
            r = rng.uniform(0.6, 2.0 + 3 * wear)
            ox, oy = cx + (r if cx == 0 else -r), cy + (r if cy == 0 else -r)
            near = (np.abs(xs - cx) < r) & (np.abs(ys - cy) < r)
            d = np.hypot(xs - ox, ys - oy)
            mask *= np.where(near, np.clip((r - d) * ppmm + 0.5, 0, 1), 1).astype(np.float32)
        fray = [_ragged(n, rng, ppmm, 0.0, 0.12 + 0.2 * wear) for n in (w, w, h, h)]
        inset = np.minimum.reduce([ys - np.abs(fray[0])[None, :], (hmm - ys) - np.abs(fray[1])[None, :],
                                   xs - np.abs(fray[2])[:, None], (wmm - xs) - np.abs(fray[3])[:, None]])
        mask *= np.clip(inset * ppmm + 0.5, 0, 1)
        for _ in range(rng.poisson(2 + 6 * wear)):
            side = rng.integers(4)
            t = rng.uniform(0, wmm if side < 2 else hmm)
            cx, cy = [(t, 0), (t, hmm), (0, t), (wmm, t)][side]
            r = rng.uniform(0.2, 0.9)
            mask *= np.clip((np.hypot(xs - cx, ys - cy) - r) * ppmm + 0.5, 0, 1)
        if rng.random() < 0.3 + 0.5 * wear:
            dog_ear(H, albedo, ppmm, wmm, hmm, rng)
        # Edges handled often wave a little: long, gentle waves that die out within a
        # couple of centimetres of the edge (short steep ones read as dark smudges).
        edge = np.minimum(np.minimum(xs, wmm - xs), np.minimum(ys, hmm - ys))
        wav = blur(band((h, w), rng, 20 * ppmm, 70 * ppmm), 3 * ppmm)
        wav /= wav.std() + 1e-9
        H += (rng.uniform(0.2, 0.4 + 0.6 * wear) * wav * np.exp(-edge / rng.uniform(12, 25))).astype(np.float32)
        return choice
    if choice == "torn":
        # Torn along one side: a ragged line, and the pale fibres the tear exposes.
        side = rng.integers(4)
        n = w if side < 2 else h
        depth = _ragged(n, rng, ppmm, rng.uniform(1.0, 3.0), rng.uniform(0.6, 1.6))
        if side < 2:
            dist = (ys if side == 0 else hmm - ys) - depth[None, :]
        else:
            dist = (xs if side == 2 else wmm - xs) - depth[:, None]
        _tear(mask, albedo, dist, ppmm, rng)
        return choice
    if choice == "spiral":
        # Torn out of a spiral notebook: down the left side, the paper between the wire's
        # holes is left as ragged tabs, some long, some torn away.
        pitch = 25.4 / rng.choice([3, 4])
        hole_r = rng.uniform(1.5, 2.1)
        hole_x = rng.uniform(3.4, 4.4)
        # The tear runs through the holes; between two holes a tab of paper may survive,
        # notched round by the holes on either side and ragged where it finally gave.
        tear = _ragged(h, rng, ppmm, hole_x + rng.uniform(-0.3, 0.8), 0.35)[:, None]
        k = np.round(ys / pitch)
        hole = np.hypot(xs - hole_x, ys - k * pitch) - hole_r  # to the nearest hole's rim
        n = int(hmm / pitch) + 3
        tab_rng = np.random.default_rng(rng.integers(2**31))
        ends = np.where(tab_rng.random(n) < 0.3, 99.0, tab_rng.uniform(-1.0, hole_x - hole_r * 0.5, n))
        end = ends[np.clip(np.floor(ys / pitch).astype(int), 0, n - 1)]
        end = end + 0.5 * np.abs(_ragged(h, rng, ppmm, 0, 1.0))[:, None]
        tab = np.minimum(xs - end, hole)
        _tear(mask, albedo, np.maximum(xs - tear, tab), ppmm, rng)
        return choice
    raise ValueError(f"edges={choice}")


def _tear(mask, albedo, dist, ppmm, rng):
    """Cut where `dist` (mm, positive on the paper) is below zero, with a fuzz of fibres."""
    fuzz = 0.5 + 0.5 * np.tanh(rng.standard_normal(mask.shape).astype(np.float32))
    mask *= np.clip(dist * ppmm * 1.2 + 0.3 + 0.4 * fuzz * (dist < 0.25), 0, 1).astype(np.float32)
    pale = np.exp(-np.maximum(dist, 0) / 0.25) * (dist > -0.5)
    albedo += (pale[..., None] * 0.35 * (np.float32(0.93) - albedo)).astype(np.float32)


def dog_ear(H, albedo, ppmm, wmm, hmm, rng):
    """A corner once folded over and pressed, then folded back: it stands up a little."""
    cx, cy = rng.choice([0, wmm]), rng.choice([0, hmm])
    a, b = rng.uniform(6, 22), rng.uniform(6, 22)  # where the fold meets the two edges
    sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
    win = _window(H.shape, ppmm, min(cx, cx + sx * a) - 1, min(cy, cy + sy * b) - 1, max(cx, cx + sx * a) + 1, max(cy, cy + sy * b) + 1)
    if win is None:
        return
    where, X, Y = win
    # Distance past the fold line, towards the corner.
    u = 1 - (np.abs(X - cx) / a + np.abs(Y - cy) / b)
    past = np.maximum(u, 0) * a * b / np.hypot(a, b)
    lift = np.tan(np.radians(rng.uniform(6, 30)))
    H[where] += (past * lift).astype(np.float32)
    line = np.exp(-((u * a * b / np.hypot(a, b)) / 0.15) ** 2) * (u > -0.2)
    albedo[where] *= (1 + 0.04 * line[..., None] * np.array([1.0, 1.0, 1.0], np.float32)).astype(np.float32)


# Stains ---------------------------------------------------------------------

def parse_stains(value):
    """'coffee, water' or 'coffee+smudge' as a set; 'random' and 'none' as themselves."""
    parts = {p.strip() for p in str(value).replace("+", ",").replace(" ", ",").split(",") if p.strip()}
    return parts or {"random"}


def choose_stains(value, wear, rng, pencil):
    parts = parse_stains(value)
    if parts == {"none"}:
        return set()
    if parts == {"random"}:
        chosen = set()
        if rng.random() < 0.3 * wear:
            chosen.add("coffee")
        if rng.random() < 0.15 * wear:
            chosen.add("water")
        if rng.random() < (0.5 if pencil else 0.2) * (0.3 + wear):
            chosen.add("smudge")
        return chosen
    return parts - {"random", "none"}


def coffee(albedo, H, x, y, ppmm, wmm, hmm, rng):
    """A mug's base, as real coffee stains go: a rim that wanders and breaks off, darker
    where the coffee dried at its edge, a blotchy tint inside, sometimes a second ring
    half over the first, and a drip or two that are never round."""
    radius = rng.uniform(30, 44)
    cx = rng.choice([rng.uniform(-radius * 0.5, radius), rng.uniform(wmm - radius, wmm + radius * 0.5)])
    cy = rng.uniform(radius, hmm - radius)
    strength = rng.uniform(0.12, 0.35)
    blotch = band(x.shape, rng, 2 * ppmm, 25 * ppmm)
    stain = _ring(x, y, cx, cy, radius, strength, rng, blotch)
    if rng.random() < 0.4:  # set down twice, not quite in the same place
        a = rng.uniform(0, 2 * np.pi)
        shift = rng.uniform(4, 0.5 * radius)
        stain = np.maximum(stain, _ring(x, y, cx + shift * np.cos(a), cy + shift * np.sin(a), radius * rng.uniform(0.97, 1.03),
                                        strength * rng.uniform(0.35, 0.7), rng, blotch, filled=0.1))
    for _ in range(rng.integers(0, 4)):
        a, d = rng.uniform(0, 2 * np.pi), radius + rng.uniform(2, 20)
        stain = np.maximum(stain, _ring(x, y, cx + d * np.cos(a), cy + d * np.sin(a), rng.uniform(1.2, 6), strength * rng.uniform(0.8, 1.4),
                                        rng, blotch, filled=0.7, rough=0.35))
    albedo *= (1 - stain[..., None] * np.array([0.55, 0.85, 1.25], np.float32)).astype(np.float32)
    # Wetted paper dries a touch wavy: a soft swell, never a ridge that throws a shadow.
    H += (0.025 * blur(np.clip(stain / strength, 0, 1.2), 3 * ppmm)).astype(np.float32)


def _around(theta, rng, harmonics, falloff):
    """Smooth random variation around a circle, unit-ish amplitude."""
    k = np.arange(1, harmonics + 1)[:, None, None]
    amp = rng.normal(0, 1, harmonics)[:, None, None] / k ** falloff
    return (amp * np.sin(k * theta[None] + rng.uniform(0, 2 * np.pi, harmonics)[:, None, None])).sum(0)


def _ring(x, y, cx, cy, radius, strength, rng, blotch, filled=0.3, rough=0.12):
    d = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    r = d - radius * (1 + rough * _around(theta, rng, 14, 1.2))
    width = (0.5 + 0.5 * np.abs(_around(theta, rng, 6, 1.0))) * max(radius / 40, 0.25)
    rim = np.exp(-np.maximum(r, -r * 0.3) ** 2 / (width + 1e-3))
    # The rim breaks off in places, and runs darker in others.
    along = smoothstep(-0.6, 0.6, _around(theta, rng, 5, 0.8) + 0.4)
    inside = (r < 0) * filled * (0.4 + 0.6 * (np.clip(d / radius, 0, 1) ** 3)) * (0.7 + 0.3 * np.tanh(blotch))
    return ((rim * along + inside) * strength).astype(np.float32)


def water(albedo, H, ppmm, wmm, hmm, rng):
    """A drop of water dried: a faint brownish tide line, and the paper cockled inside it."""
    h, w = H.shape
    cx, cy = rng.uniform(0.1 * wmm, 0.9 * wmm), rng.uniform(0.1 * hmm, 0.9 * hmm)
    size = rng.uniform(12, 35)
    pad = 1.6 * size
    win = _window(H.shape, ppmm, cx - pad, cy - pad, cx + pad, cy + pad)
    if win is None:
        return
    where, X, Y = win
    theta = np.arctan2(Y - cy, X - cx)
    k = np.arange(1, 7)[:, None, None]
    shape = 1 + (rng.normal(0, 0.12, 6)[:, None, None] * np.sin(k * theta[None] + rng.uniform(0, 6, 6)[:, None, None])).sum(0)
    r = np.hypot(X - cx, Y - cy) / (size * shape) - 1  # 0 on the tide line
    line = np.exp(-((r * size / 0.6) ** 2)) * rng.uniform(0.06, 0.12)
    inside = smoothstep(0.05, -0.1, r)
    albedo[where] *= (1 - line[..., None] * np.array([0.5, 0.8, 1.3], np.float32) - 0.01 * inside[..., None]).astype(np.float32)
    bumps = np.random.default_rng(rng.integers(2**31)).standard_normal((Y.shape[0], X.shape[1])).astype(np.float32)
    bumps = blur(bumps, 6 * ppmm)
    bumps /= bumps.std() + 1e-9
    # Cockled paper: low, broad waves inside, fading out gently past the tide line (a hard
    # edge to them would cast a shadow ring no real stain has).
    fade = smoothstep(0.35, -0.4, r)
    H[where] += (rng.uniform(0.03, 0.07) * (bumps + 0.6) * fade).astype(np.float32)


def smudge(amount, rng, ppmm, pencil):
    """Ink (or graphite) dragged by the side of a hand: a faint smear trailing from some strokes."""
    h, w = amount.shape
    out = np.zeros_like(amount)
    for _ in range(rng.integers(1, 3)):
        cx, cy = rng.uniform(0.15 * w, 0.85 * w), rng.uniform(0.15 * h, 0.85 * h)
        rx, ry = rng.uniform(15, 40) * ppmm, rng.uniform(5, 15) * ppmm
        ys, xs = np.ogrid[0:h, 0:w]
        region = np.exp(-(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)).astype(np.float32)
        angle = rng.normal(0.15, 0.3)  # mostly along the line of writing, as the hand moves
        dx, dy = np.cos(angle), np.sin(angle)
        trail, total = np.zeros_like(amount), 0.0
        steps = int(rng.uniform(1.5, 4.0) * ppmm)
        for s in range(1, steps, max(1, steps // 12)):
            weight = np.exp(-s / steps * 2.5)
            trail += np.roll(amount, (int(round(s * dy)), int(round(s * dx))), axis=(0, 1)) * weight
            total += weight
        trail = blur(trail / max(total, 1e-6), 0.8 * ppmm)
        out += trail * region * rng.uniform(0.12, 0.3) * (1.8 if pencil else 1.0)
    return out


# Curl -----------------------------------------------------------------------

def curl(level, x, y, wmm, hmm, rng):
    """Height (mm) at points (x, y) of a sheet whose edges roll up `level` (0..1).

    Thin paper that was rolled, carried in a bag or dried after getting damp keeps a set:
    the whole sheet bowed like part of a tube, or one or two edges and corners lifted
    off the desk, rising faster the nearer the edge.
    """
    z = np.zeros(np.broadcast(x, y).shape, np.float32)
    if level <= 0:
        return z
    kind = rng.choice(["bow", "edges", "corner"], p=[0.35, 0.4, 0.25])
    if kind == "bow":
        # Rolled across the short side or the long one; it lies on its middle.
        across = rng.random() < 0.7
        u = (x - wmm / 2) / (wmm / 2) if across else (y - hmm / 2) / (hmm / 2)
        z += level * rng.uniform(8, 22) * np.abs(u) ** rng.uniform(1.8, 2.6)
    elif kind == "edges":
        for side in rng.choice(4, size=rng.integers(1, 3), replace=False):
            d = (x, wmm - x, y, hmm - y)[side]
            reach = rng.uniform(25, 70)
            z += level * rng.uniform(6, 18) * smoothstep(reach, 0, d) ** 1.7
    else:
        cx, cy = rng.choice([0, wmm]), rng.choice([0, hmm])
        reach = rng.uniform(50, 110)
        d = np.hypot(x - cx, y - cy)
        z += level * rng.uniform(10, 25) * smoothstep(reach, 0, d) ** 1.8
    return z.astype(np.float32)


# Age ------------------------------------------------------------------------

def age(level, albedo, ppmm, wmm, hmm, rng):
    """Turn a sheet `level` (0..1) old: yellowed unevenly, deepest at the edges, with
    the small brown spots (foxing) that come with damp, mostly in clusters."""
    h, w = albedo.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) / ppmm
    edge = np.minimum(np.minimum(xs, wmm - xs), np.minimum(ys, hmm - ys))
    blotch = band((h, w), rng, 15 * ppmm, 120 * ppmm)
    yellow = level * (0.05 + 0.09 * np.exp(-edge / rng.uniform(10, 30)) + 0.03 * np.clip(blotch, -1.5, 2.5))
    albedo *= np.exp(-np.clip(yellow, 0, None)[..., None] * np.array([0.12, 0.4, 1.1], np.float32))
    size = wmm * hmm / A4
    clusters = rng.uniform([0, 0], [wmm, hmm], (max(1, rng.poisson(3 * level * size)), 2))
    for _ in range(rng.poisson(level * level * 60 * size)):
        cx, cy = clusters[rng.integers(len(clusters))] + rng.normal(0, rng.uniform(5, 30), 2)
        radius = float(np.exp(rng.normal(np.log(0.7), 0.6)))
        win = _window((h, w), ppmm, cx - 3 * radius, cy - 3 * radius, cx + 3 * radius, cy + 3 * radius)
        if win is None:
            continue
        where, X, Y = win
        theta = np.arctan2(Y - cy, X - cx)
        r = np.hypot(X - cx, Y - cy) / (radius * (1 + 0.25 * np.sin(3 * theta + rng.uniform(0, 6)) + 0.15 * np.sin(5 * theta + rng.uniform(0, 6))))
        spot = rng.uniform(0.15, 0.6) * (np.exp(-r * r * 1.5) + 0.6 * np.exp(-r * r * 12))
        albedo[where] *= np.exp(-spot[..., None] * np.array([0.35, 0.6, 1.0], np.float32))

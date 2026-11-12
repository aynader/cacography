"""Things lying with the sheet in the photo: tape, a paper clip, a binder clip, a staple, a pen, another sheet.

Each is the `props` aspect's own random draw, so adding a pen leaves the folds and the
light as they were. Everything works in the sheet's millimetres (x right, y down the
page, z up off the desk), the frame render3d builds the scene in.

Tape is thin enough to be paint: on the paper it changes the sheet's colour, gloss and
relief; past the paper's edge it is a thin layer over the desk that shows the desk's own
texture through a glossy coat (clear tape) or covers it (masking tape). Only the clip,
staple, pen and the other sheet are solid, so only they cast shadows. Props are drawn
when tracing in 3D; the formula camera leaves them out.
"""
from dataclasses import replace

import numpy as np
from PIL import Image

from .imaging import band, blur, sample, smoothstep

KINDS = ("tape", "clip", "binder", "staple", "pen", "sheet")
ODDS = {"tape": 0.06, "clip": 0.06, "binder": 0.04, "staple": 0.08, "pen": 0.1, "sheet": 0.08}  # each, at wear 0.4


def parse(value):
    return [p for p in str(value).replace("+", ",").replace(" ", ",").split(",") if p]


def choose(value, wear, rng):
    """Which props lie in this photo: as set, or at random (more on a well-used sheet)."""
    if value == "none":
        return []
    if value != "random":
        return parse(value)
    chosen = [k for k, p in ODDS.items() if rng.random() < p * (0.6 + wear)]
    holders = [k for k in ("clip", "binder", "staple") if k in chosen]
    for k in holders[1:]:
        chosen.remove(k)  # one thing holding the sheets together is enough
    return chosen


def dress(sheet, kinds, rng, ground):
    """The sheet with tape and staple marks on it, and the solid things around it.

    `ground(x, y)` is the height of the desk or cloth (mm). Returns (sheet, things): each
    thing is a dict with "verts" (N, 3), "faces" (M, 3), optional "uv" (N, 2), and a
    "material": {"kind": "steel" | "plastic" | "matte" | "paper" | "clear tape" | "masking tape", ...}.
    """
    sheet = replace(sheet, albedo=sheet.albedo.copy(), relief=sheet.relief.copy(), bend=sheet.bend.copy(),
                    sheen=np.zeros_like(sheet.relief) if sheet.sheen is None else sheet.sheen.copy())
    things = []
    for kind in kinds:
        sub = np.random.default_rng(rng.integers(2**31))
        if kind == "tape":
            for _ in range(1 if sub.random() < 0.6 else 2):
                things += _tape(sheet, sub, ground)
        elif kind == "clip":
            things.append(_clip(sheet, sub, ground))
        elif kind == "binder":
            things += _binder(sheet, sub, ground)
        elif kind == "staple":
            things.append(_staple(sheet, sub))
        elif kind == "pen":
            things += _pen(sheet, sub, ground)
        elif kind == "sheet":
            things.append(_other_sheet(sheet, sub, ground))
    return sheet, things


def top(sheet, x, y):
    """Height of the sheet's upper face at (x, y) mm (past its edge, its edge's height)."""
    return sheet.lift(np.asarray(x, np.float32), np.asarray(y, np.float32)) + 0.1


def _upper_hull(t, q):
    """Indices of the upper convex hull of the points (t, q), t increasing."""
    hull = []
    for i in range(len(t)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (t[b] - t[a]) * (q[i] - q[a]) - (q[b] - q[a]) * (t[i] - t[a]) >= 0:
                hull.pop()  # b lies on or under the line from a to i
            else:
                break
        hull.append(i)
    return hull


def _rest(sheet, ground, x, y, radius):
    """Heights of the axis of a rigid rod (pen, pencil) laid along (x, y), its radius per point
    `radius`, resting on the sheet and the desk under it as a real one does: a straight line
    on the two supports either side of its centre of weight, not floating at the height of
    the highest crease. Paper gives under it: a crumpled ridge is squashed by up to 1.5 mm
    (never below the desk), and the sheet is pressed down where the rod lies on it.

    Changes sheet.bend in place. Returns z (N,).
    """
    r = float(np.max(radius))
    across = np.array([-(y[-1] - y[0]), x[-1] - x[0]], np.float64)
    across /= np.linalg.norm(across) + 1e-9
    # What lies under the rod's width: the centre line and either side of it.
    rigid, soft = np.full(len(x), -np.inf), np.full(len(x), -np.inf)
    for d in (-0.6, -0.3, 0.0, 0.3, 0.6):
        px, py = x + d * r * across[0], y + d * r * across[1]
        drop = radius - np.sqrt(np.clip(radius**2 - (d * r) ** 2, 0, None))  # the round bottom rises off the axis
        on = sample(sheet.mask, px * sheet.ppmm, py * sheet.ppmm, fill=0.0) > 0.5
        g = ground(px, py)
        s = np.where(on, top(sheet, px, py), g)
        # A ridge of paper gives under the rod's weight, a little; the desk doesn't.
        give = np.where(on, np.minimum(1.5, 0.6 * np.clip(s - g - 0.1, 0, None)), 0.0)
        rigid = np.maximum(rigid, s - give - drop)  # off the axis the round bottom is higher: it touches lower supports later
        soft = np.maximum(soft, s - drop)
    q = rigid + radius
    t = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    hull = _upper_hull(t, q)
    weight = radius**2
    t_cm = float((t * weight).sum() / weight.sum())
    z = None
    for a, b in zip(hull[:-1], hull[1:]):
        if t[a] <= t_cm <= t[b]:
            z = q[a] + (q[b] - q[a]) * (t - t[a]) / max(t[b] - t[a], 1e-9)
            break
    if z is None:
        z = np.full(len(t), q.max())
    # Press the sheet down where it would poke through the rod.
    excess = soft + radius - z
    if (excess > 0.02).any():
        cell = sheet.BEND_MM
        gy, gx = np.mgrid[0 : sheet.bend.shape[0], 0 : sheet.bend.shape[1]].astype(np.float64) * cell
        ax = np.array([x[-1] - x[0], y[-1] - y[0]], np.float64)
        length = np.linalg.norm(ax) + 1e-9
        ax /= length
        u = (gx - x[0]) * ax[0] + (gy - y[0]) * ax[1]
        v = -(gx - x[0]) * ax[1] + (gy - y[0]) * ax[0]
        e = np.interp(u, t, np.clip(excess, 0, None), left=0.0, right=0.0)
        # The paper bends down over a few millimetres either side, not in a sharp groove.
        reach = 0.8 * r + 6.0
        press = e * np.exp(-np.clip(np.abs(v) - 0.5 * r, 0, None) ** 2 / (2 * (reach / 2) ** 2))
        floor = ground(gx.astype(np.float32), gy.astype(np.float32))
        sheet.bend[:] = np.maximum(sheet.bend - press.astype(sheet.bend.dtype), np.minimum(sheet.bend, floor + 0.05))
    return z


def crumbs(sheet, rng):
    """Crumbs of eraser by where words were rubbed out: little rolled bits of rubber, grey
    with graphite, a few brushed away from the spot."""
    things = []
    colour = np.array(rng.choice([(0.86, 0.55, 0.55), (0.85, 0.82, 0.76), (0.8, 0.72, 0.56)])) * rng.uniform(0.72, 0.9)
    for x0, y0, x1, y1 in sheet.erasures or []:
        cx, cy = 0.5 * (x0 + x1) / sheet.ppmm, 0.5 * (y0 + y1) / sheet.ppmm
        for _ in range(int(rng.integers(0, 5))):
            r = rng.uniform(0.25, 0.45)
            d = 0.5 * (x1 - x0) / sheet.ppmm + 2.0 + abs(rng.normal(0, 10))  # brushed off the words
            a = rng.uniform(0, 2 * np.pi)
            px, py = cx + d * np.cos(a), cy + 0.5 * d * np.sin(a)
            if not (0 < px < sheet.width_mm and 0 < py < sheet.height_mm):
                continue
            n = int(rng.integers(3, 6))
            step = rng.uniform(0.3, 0.8)
            heading = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.6, n))
            xy = np.column_stack([px + np.cumsum(step * np.cos(heading)), py + np.cumsum(step * np.sin(heading))])
            z = top(sheet, xy[:, 0], xy[:, 1]) + r * 0.85
            path = _resample(np.column_stack([xy, z]), 0.1)
            rad = r * (0.7 + 0.3 * np.sin(np.linspace(0, np.pi, len(path)))) * rng.uniform(0.8, 1.1)
            v, f = _tube(path, rad.astype(np.float32), 10)
            things.append({"verts": v, "faces": f, "uv": None, "material": {"kind": "matte", "colour": list(colour * rng.uniform(0.85, 1.1))}})
    return things


# Tape -------------------------------------------------------------------------

def _tape(sheet, rng, ground):
    """A strip of tape over a corner or across an edge, holding the sheet down."""
    wmm, hmm = sheet.width_mm, sheet.height_mm
    clear = rng.random() < 0.65
    colour = np.array(rng.choice([(0.72, 0.62, 0.42), (0.75, 0.68, 0.5), (0.25, 0.48, 0.75)]), np.float32)
    width = rng.uniform(18, 20) if clear else rng.uniform(18, 26)
    length = rng.uniform(35, 70)
    if rng.random() < 0.55:
        # Over a corner, on the diagonal, about half on the paper.
        cx, cy = rng.choice([0.0, wmm]), rng.choice([0.0, hmm])
        inward = np.array([1.0 if cx == 0 else -1.0, 1.0 if cy == 0 else -1.0]) / np.sqrt(2)
        centre = np.array([cx, cy]) + inward * rng.uniform(2, 10)
        angle = np.arctan2(inward[1], inward[0]) + np.pi / 2 + rng.normal(0, 0.2)
    else:
        # Across an edge, square to it.
        side = rng.integers(4)
        t = rng.uniform(0.2, 0.8) * (wmm if side < 2 else hmm)
        centre = np.array([(t, 0.0), (t, hmm), (0.0, t), (wmm, t)][side])
        angle = (np.pi / 2 if side < 2 else 0.0) + rng.normal(0, 0.12)
        centre = centre + np.array([np.cos(angle), np.sin(angle)]) * rng.normal(0, 4)
    ca, sa = np.cos(angle), np.sin(angle)

    def ends(v):
        """How far each end is cut back across the strip: a dispenser's teeth, or a tear."""
        if clear:
            return 0.35 * np.abs(((v / 1.2) % 2) - 1)
        return 0.6 * np.abs(np.sin(v * 1.7 + rng.uniform(0, 6))) + 0.4 * np.abs(np.sin(v * 4.3))

    # On the paper: paint.
    ppmm = sheet.ppmm
    h, w = sheet.mask.shape
    pad = 0.5 * np.hypot(length, width) + 2
    i0, i1 = max(0, int((centre[1] - pad) * ppmm)), min(h, int((centre[1] + pad) * ppmm) + 1)
    j0, j1 = max(0, int((centre[0] - pad) * ppmm)), min(w, int((centre[0] + pad) * ppmm) + 1)
    if i0 < i1 and j0 < j1:
        ys, xs = np.mgrid[i0:i1, j0:j1].astype(np.float32) / ppmm
        u = (xs - centre[0]) * ca + (ys - centre[1]) * sa
        v = -(xs - centre[0]) * sa + (ys - centre[1]) * ca
        cut = ends(v)
        inside = (smoothstep(width / 2 + 0.05, width / 2 - 0.05, np.abs(v))
                  * smoothstep(length / 2 - cut + 0.05, length / 2 - cut - 0.05, np.abs(u)))
        where = (slice(i0, i1), slice(j0, j1))
        if clear:
            # Clear tape: a faint yellowish haze, a glassy shine, a step at its edges,
            # and here and there a trapped bubble.
            sheet.albedo[where] *= (1 - inside[..., None] * np.array([0.02, 0.025, 0.06], np.float32))
            sheet.sheen[where] = np.maximum(sheet.sheen[where], inside * rng.uniform(0.2, 0.35))
            bubbles = smoothstep(2.2, 3.0, band(inside.shape, rng, 1.5 * ppmm, 6 * ppmm))
            sheet.relief[where] += inside * (0.03 + 0.04 * bubbles)
        else:
            # Masking tape: opaque crepe paper, creped across its length.
            crepe = band(inside.shape, rng, 0.3 * ppmm, 1.2 * ppmm, stretch=6, angle=angle + np.pi / 2)
            tape = colour * (1 + 0.04 * crepe)[..., None]
            sheet.albedo[where] = sheet.albedo[where] * (1 - inside[..., None]) + tape * inside[..., None]
            sheet.sheen[where] *= 1 - inside
            sheet.relief[where] += inside * (0.1 + 0.008 * crepe)

    # Past the paper: a thin layer on the desk.
    nu, nv = max(2, int(length / 1.5)), max(2, int(width / 1.5))
    vv = np.linspace(-width / 2, width / 2, nv)
    s = np.linspace(-1, 1, nu)[:, None]
    half = length / 2 - ends(vv)[None, :]
    uu = s * half
    X = centre[0] + uu * ca - vv[None, :] * sa
    Y = centre[1] + uu * sa + vv[None, :] * ca
    Z = ground(X, Y) + 0.04
    verts, faces = _grid(X, Y, Z)
    if clear:
        material = {"kind": "clear tape"}
    else:
        material = {"kind": "masking tape", "colour": colour.tolist()}
    return [{"verts": verts, "faces": faces, "uv": None, "material": material, "on_desk": True}]


# Paper clip -------------------------------------------------------------------

def _clip_path(length, width, wire, steps=20):
    """A gem clip's wire, flat: (N, 2) mm with y = 0 at the tip of its single big bend and
    y = `length` at the far end, and which face each point is on (0 the inner loop, 1
    the outer loop, between the two along the big bend, where the wire crosses over).

    Measured from real clips: the long outer loop; an inner loop inside it, ending about
    six-sevenths of the way down; one big bend joining the inner loop's right leg to the
    outer loop's left leg; both wire ends a third of the way down.
    """
    pts, face = [], []

    def leg(x, y0, y1, k):
        pts.extend([(x, y) for y in np.linspace(y0, y1, steps)][1:])
        face.extend([k] * (steps - 1))

    def bend(cx, cy, r, a0, a1, k0, k1):
        n = 3 * steps
        pts.extend([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(a0, a1, n)][1:])
        face.extend(list(np.linspace(k0, k1, n))[1:])

    W, L = width, length
    gap = 2.6 * wire  # between the inner and outer legs on the right: they nearly touch
    xo1, xi1, xi2, xo2 = 0.0, gap, W - gap, W
    ri, rt, ro = (xi2 - xi1) / 2, (xi2 - xo1) / 2, (xo2 - xo1) / 2
    end = 0.32 * L
    pts.append((xi1, end))
    face.append(0)
    leg(xi1, end, 0.86 * L - ri, 0)
    bend((xi1 + xi2) / 2, 0.86 * L - ri, ri, np.pi, 0, 0, 0)  # the inner loop's bottom
    leg(xi2, 0.86 * L - ri, rt, 0)
    bend((xo1 + xi2) / 2, rt, rt, 0, -np.pi, 0, 1)  # the big bend, over the paper's edge
    leg(xo1, rt, L - ro, 1)
    bend(W / 2, L - ro, ro, np.pi, 0, 1, 1)  # the outer loop's bottom
    leg(xo2, L - ro, end, 1)
    p = np.array(pts, np.float32)
    p[:, 0] -= W / 2
    return p, np.array(face, np.float32)


def _clip(sheet, rng, ground):
    """A paper clip pushed onto the top edge: one loop on the paper, one under it, the
    big bend standing out past the edge, the paper riding up over the wire beneath."""
    wmm = sheet.width_mm
    length = rng.choice([rng.uniform(26, 30), rng.uniform(31, 35), rng.uniform(44, 50)])
    width = length * rng.uniform(0.24, 0.27)
    wire = rng.uniform(0.38, 0.45) * (length / 33) ** 0.5  # radius
    flat, face = _clip_path(length, width, wire)
    if rng.random() < 0.5:
        flat[:, 0] = -flat[:, 0]  # turned over
    if rng.random() < 0.5:
        face = 1 - face  # the outer loop on top instead
    rt = (width - 2.6 * wire) / 2
    # The paper's edge is just past where the big bend meets the straight legs.
    flat[:, 1] -= rt + rng.uniform(0.5, 3.0)
    angle = rng.normal(0, 0.12)
    c, s = np.cos(angle), np.sin(angle)
    x = rng.uniform(0.2, 0.8) * wmm + flat[:, 0] * c - flat[:, 1] * s
    y = flat[:, 0] * s + flat[:, 1] * c
    # The paper rides up over the wire beneath it, all the way to the wire's end
    # (where it didn't, the end poked through the paper as a dark spot).
    under = face > 0.5
    cell = sheet.BEND_MM
    bh, bw = sheet.bend.shape
    by, bx = np.mgrid[0:bh, 0:bw].astype(np.float32) * cell
    d = np.full(bx.shape, np.inf, np.float32)
    for px, py in zip(x[under], y[under]):
        d = np.minimum(d, np.hypot(bx - px, by - py))
    lift = (ground(bx, by) + 2 * wire + 0.25) * smoothstep(5.0, 1.5, d)
    np.maximum(sheet.bend, lift, out=sheet.bend)
    on_top = top(sheet, x, y) + wire
    below = ground(x, y) + wire
    # Past the paper's edge both faces come down to the paper's thickness apart.
    z = (1 - face) * on_top + face * below
    coated = rng.random() < 0.25
    # Smooth enough that the bends stay round up close; a plastic coat makes the wire thicker.
    path = _resample(np.column_stack([x, y, z]), 0.25)
    # A plastic coat makes the wire thicker and softer looking: rounder, a satin gloss.
    verts, faces = _tube(path, np.full(len(path), wire + (0.17 if coated else 0.0)), 20)
    material = ({"kind": "pplastic", "colour": list(rng.choice([(0.5, 0.05, 0.05), (0.08, 0.22, 0.5), (0.08, 0.36, 0.16), (0.7, 0.6, 0.08), (0.35, 0.1, 0.42), (0.75, 0.75, 0.72)])),
                 "rough": rng.uniform(0.38, 0.5)} if coated else {"kind": "steel", "rough": rng.uniform(0.12, 0.22)})
    return {"verts": verts, "faces": faces, "uv": None, "material": material}


# Binder clip ------------------------------------------------------------------

def _binder(sheet, rng, ground, at=None):
    """A binder clip on the top edge, as real ones are made: a sheet of spring steel folded
    into a triangle (open at its ends, so they show hollow), its two lips rolled outward
    into tubes, the wire handles pivoting inside those rolls. It lies on its lower jaw (and
    on the lower handle, when that is folded back under it), the paper clamped between the jaws."""
    wmm = sheet.width_mm
    width = float(rng.choice([19.0, 25.0, 32.0, 41.0], p=[0.3, 0.35, 0.25, 0.1]))
    depth, height = 0.5 * width, 0.36 * width  # from the spine to the lips; the spine's height
    th = 0.3 + 0.005 * width  # the steel
    rr = 0.04 * width + 0.2  # the rolled lips' radius
    rw = 0.018 * width + 0.15  # the handles' wire
    cx = (rng.uniform(0.25, 0.75) * wmm) if at is None else float(at)
    angle = rng.normal(0, 0.08)
    along = np.array([np.cos(angle), np.sin(angle)])  # the paper's edge
    into = np.array([-along[1], along[0]])  # into the page
    spine = np.array([cx, 0.0]) - into * rng.uniform(0.5, 2.0)
    things = []
    # A rigid clip rests on the highest point of what is under it.
    gu, gb = np.meshgrid(np.linspace(-width / 2, width / 2, 9), np.linspace(0, depth, 5))
    gx, gy = spine[0] + gu * along[0] + gb * into[0], spine[1] + gu * along[1] + gb * into[1]
    base_z = float(np.max(ground(gx.astype(np.float32), gy.astype(np.float32))))
    lower_back = rng.random() < 0.6  # the lower handle folded back, under the clip
    lift = 2 * rw + th if lower_back else th
    # The section, (into, up): lower lip, spine corners, upper lip. The paper rests on the
    # lower jaw and the upper jaw presses down on it at the lip.
    lip_low = (depth - 0.2 * rr, 2 * rr)
    paper_top = lip_low[1] + th + 0.15
    lip_up = (depth - 0.2 * rr, paper_top + 0.5 * th + 0.05)
    keys = np.array([lip_low, (depth * 0.5, 0.5 * (lip_low[1] + lift)), (1.2, lift), (0.0, lift + 1.2), (0.0, height - 1.2),
                     (1.2, height), (depth * 0.5, 0.5 * (height + lip_up[1]) + 0.03 * width), lip_up], np.float64)
    path = _spline(keys, 14).astype(np.float64)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.stack([tang[:, 1], -tang[:, 0]], 1)
    outer, inner = path + 0.5 * th * normal, path - 0.5 * th * normal
    n = 30
    u = np.linspace(-width / 2, width / 2, n)
    m = len(path)

    def place(sec):  # (m, 2) -> (n, m, 3)
        xy = spine[None, None, :] + u[:, None, None] * along + sec[None, :, :1] * into
        return np.concatenate([xy, base_z + np.repeat(sec[None, :, 1:], n, 0)], -1)

    O, I = place(outer), place(inner)
    verts = np.concatenate([O.reshape(-1, 3), I.reshape(-1, 3)]).astype(np.float32)
    off = n * m
    faces = []
    for i in range(n - 1):
        for j in range(m - 1):
            a, b, c, d = i * m + j, i * m + j + 1, (i + 1) * m + j, (i + 1) * m + j + 1
            faces += [(a, c, b), (b, c, d), (off + a, off + b, off + c), (off + b, off + d, off + c)]
        for j in (0, m - 1):  # the strip's long edges (at the lips)
            a, c = i * m + j, (i + 1) * m + j
            faces += [(a, off + a, c), (c, off + a, off + c)] if j == 0 else [(a, c, off + a), (c, off + c, off + a)]
    for i in (0, n - 1):  # the ends: just the steel's thickness, so the clip shows hollow
        for j in range(m - 1):
            a, b = i * m + j, i * m + j + 1
            faces += [(a, b, off + a), (b, off + b, off + a)] if i == 0 else [(a, off + a, b), (b, off + a, off + b)]
    from . import pens
    faces = pens.orient(verts, np.array(faces, np.uint32))
    paint = (0.02, 0.02, 0.025) if rng.random() < 0.8 else tuple(rng.choice([(0.5, 0.04, 0.04), (0.04, 0.12, 0.42), (0.06, 0.3, 0.12), (0.55, 0.55, 0.57)]))
    finish = {"kind": "pplastic", "colour": list(paint), "rough": rng.uniform(0.22, 0.32)}
    things.append({"verts": verts, "faces": faces, "uv": None, "material": finish, "twosided": True})
    # The rolled lips: tubes along the edge, the handles' ends inside.
    rolls = {}
    for name, (b_, z_), up in (("low", lip_low, -1), ("up", lip_up, 1)):
        c = np.array([b_ + 0.15 * rr, z_ + up * rr])
        rolls[name] = c
        line = np.column_stack([spine[0] + u * along[0] + c[0] * into[0], spine[1] + u * along[1] + c[0] * into[1], np.full(n, base_z + c[1])])
        v, f = _tube(line, np.full(n, rr), 18)
        things.append({"verts": v, "faces": f, "uv": None, "material": finish})
    # The handles: each end bent in along the roll, then the two arms out to a loop.
    steel = {"kind": "chrome", "rough": 0.1}
    for which in ("up", "low"):
        if which == "up":
            folded = str(rng.choice(["back", "forward"], p=[0.55, 0.45]))
        else:
            if not lower_back:
                continue
            folded = "back"
        span = width * rng.uniform(0.95, 1.1)
        arm = width / 2 - 2.0
        flat = _handle(2 * arm, span).astype(np.float64)
        c = rolls[which]
        dirn = -into if folded == "back" else into
        y = flat[:, 1]
        if which == "up" and folded == "back":
            z = c[1] + (height + rw - c[1]) * y / max(c[0], 1.0)  # resting on the upper jaw, out past the spine
        elif which == "up":
            z = c[1] + (paper_top + rw - c[1]) * (1 - np.exp(-y / 2.5))  # down onto the paper
        else:
            z = c[1] + (rw - c[1]) * np.clip(y / 4.0, 0, 1)  # down to the desk, under the clip
        X = spine[0] + c[0] * into[0] + flat[:, 0] * along[0] + y * dirn[0]
        Y = spine[1] + c[0] * into[1] + flat[:, 0] * along[1] + y * dirn[1]
        Z = base_z + z
        if which == "up" and folded == "forward":
            Z = np.maximum(Z, top(sheet, X, Y) + rw)
        # The ends tucked into the roll, pointing outward along it.
        ends = []
        for sgn, k in ((-1, 0), (1, -1)):
            ends.append(np.array([X[k], Y[k], Z[k]]) + np.array([*(sgn * 2.0 * along), 0.0]))
        pts = np.vstack([ends[0], np.column_stack([X, Y, Z]), ends[1]])
        pts = _resample(pts, 0.25)
        v, f = _tube(pts, np.full(len(pts), rw), 12)
        things.append({"verts": v, "faces": f, "uv": None, "material": steel})
    # The paper rests on the lower jaw, lifted by it, and pinched at the lips.
    cell = sheet.BEND_MM
    bh, bw = sheet.bend.shape
    by, bx = np.mgrid[0:bh, 0:bw].astype(np.float32) * cell
    rel = np.stack([bx - spine[0], by - spine[1]], -1)
    a, b = rel @ along, rel @ into
    inside = smoothstep(width / 2 + 1, width / 2 - 1, np.abs(a)) * smoothstep(depth + 3, depth, b)
    low = path[: m // 2]
    jaw = np.interp(b, low[::-1, 0], low[::-1, 1]) + 0.5 * th
    # Clamped between the jaws, the paper lies exactly on the lower one; past the lips it
    # comes back down to where it was over a few millimetres.
    held = base_z + jaw
    w_ = inside * smoothstep(depth + 4, depth - 0.5, b)
    sheet.bend[:] = (sheet.bend * (1 - w_) + held * w_).astype(sheet.bend.dtype)
    return things


def _handle(base, span, steps=60):
    """A binder clip's wire handle, flat: (N, 2) mm, x across, y away from the pivot.

    Its two arms run in from the lip's ends, close to a narrow neck and open into a
    rounded loop at the end.
    """
    a, neck, loop = base / 2, 0.11 * span, 0.2 * span
    keys = np.array([(-a, 0.0), (-neck * 1.4, 0.45 * span), (-neck, 0.62 * span), (-loop, 0.8 * span), (-loop * 0.7, 0.96 * span),
                     (0.0, span), (loop * 0.7, 0.96 * span), (loop, 0.8 * span), (neck, 0.62 * span), (neck * 1.4, 0.45 * span), (a, 0.0)])
    return _spline(keys, steps)


def _spline(keys, steps):
    """A Catmull-Rom curve through `keys` (K, 2), `steps` points per span."""
    k = np.concatenate([keys[:1], keys, keys[-1:]])
    out = []
    t = np.linspace(0, 1, steps, endpoint=False)[:, None]
    for i in range(1, len(k) - 2):
        p0, p1, p2, p3 = k[i - 1], k[i], k[i + 1], k[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(keys[-1:])
    return np.concatenate(out).astype(np.float32)


def _sweep(origin, along, into, u, section, ground):
    """A closed section (S, 2) of (distance into the page, height) swept along `u` mm of
    the edge, with its two ends closed: (verts, faces)."""
    n, m = len(u), len(section)
    P = origin[None, None, :] + u[:, None, None] * along + section[None, :, :1] * into  # n x m x 2
    Z = ground(P[..., 0], P[..., 1]) * 0 + ground(origin[0], origin[1]) + section[None, :, 1]
    verts = np.concatenate([np.concatenate([P, Z[..., None]], -1).reshape(-1, 3),
                            np.array([[*(origin + u[0] * along + into * section[:, 0].mean()), float(Z[0].mean())],
                                      [*(origin + u[-1] * along + into * section[:, 0].mean()), float(Z[-1].mean())]])]).astype(np.float32)
    faces = []
    k = np.arange(m)
    for i in range(n - 1):
        p, q = i * m + k, i * m + (k + 1) % m
        faces += [np.stack([p, p + m, q], 1), np.stack([q, p + m, q + m], 1)]
    c0, c1 = n * m, n * m + 1
    faces.append(np.stack([np.full(m, c0), (k + 1) % m, k], 1))
    faces.append(np.stack([np.full(m, c1), (n - 1) * m + k, (n - 1) * m + (k + 1) % m], 1))
    return verts, np.concatenate(faces).astype(np.uint32)


# Staple -----------------------------------------------------------------------

def _staple(sheet, rng):
    """A staple through the top-left corner, with the dents its legs pressed in."""
    length = rng.uniform(11.5, 12.8)
    angle = rng.choice([np.pi / 4, 0.0, np.pi / 2]) + rng.normal(0, 0.12)
    c, s = np.cos(angle), np.sin(angle)
    cx, cy = rng.uniform(6, 14), rng.uniform(6, 14)
    if abs(angle) < 0.3:
        cy = rng.uniform(5, 9)
    elif abs(angle - np.pi / 2) < 0.3:
        cx = rng.uniform(5, 9)
    ends = [(cx - c * length / 2, cy - s * length / 2), (cx + c * length / 2, cy + s * length / 2)]
    ppmm = sheet.ppmm
    h, w = sheet.mask.shape
    for ex, ey in ends:
        i0, j0 = int((ey - 1.5) * ppmm), int((ex - 1.5) * ppmm)
        ys, xs = np.mgrid[i0 : i0 + int(3 * ppmm), j0 : j0 + int(3 * ppmm)].astype(np.float32) / ppmm
        if i0 < 0 or j0 < 0 or ys.shape[0] == 0 or i0 + ys.shape[0] > h or j0 + ys.shape[1] > w:
            continue
        d = np.hypot(xs - ex, ys - ey)
        where = (slice(i0, i0 + ys.shape[0]), slice(j0, j0 + ys.shape[1]))
        sheet.relief[where] -= 0.08 * np.exp(-((d / 0.6) ** 2))
        sheet.albedo[where] *= (1 - 0.35 * smoothstep(0.35, 0.15, d))[..., None]
    # The crown: a flat wire band on the paper, its ends bent down into the holes.
    t = np.linspace(-0.5, 0.5, 24) * length
    z = top(sheet, cx + c * t, cy + s * t) + 0.25
    z[0], z[-1] = z[0] - 0.2, z[-1] - 0.2
    path = np.column_stack([cx + c * t, cy + s * t, z])
    verts, faces = _tube(path, np.full(len(t), 0.28), 6, squash=0.6)
    return {"verts": verts, "faces": faces, "uv": None, "material": {"kind": "steel", "rough": 0.25}}


# Pen --------------------------------------------------------------------------

DRAFTING = [(0.025, 0.025, 0.03), (0.025, 0.025, 0.03), (0.04, 0.08, 0.3), (0.55, 0.56, 0.58), (0.3, 0.04, 0.05)]
TIKKY = [(0.02, 0.02, 0.025), (0.85, 0.85, 0.83), (0.75, 0.06, 0.05), (0.06, 0.2, 0.55), (0.1, 0.55, 0.75), (0.45, 0.1, 0.5),
         (0.9, 0.4, 0.05), (0.4, 0.7, 0.1), (0.9, 0.8, 0.2)]


def _hexagon(a, r_in):
    """The radius of a hexagonal barrel at angles `a`, its faces `r_in` from the axis, one
    face on top and one beneath (a hexagonal pencil lies on a face, not an edge)."""
    return r_in / np.cos(((a - np.pi / 2 + np.pi / 6) % (np.pi / 3)) - np.pi / 6)


def _tube_tex(path, radius, sides):
    """A tube like _tube, with texture coordinates: u along it (0 to 1 by length), v round
    it (0 to 1, starting from the side, through the top at 0.25). The seam has its own
    vertices, so a texture wraps without a smear."""
    path = np.asarray(path, np.float32)
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    side = np.cross(t, [0.0, 0.0, 1.0])
    bad = np.linalg.norm(side, axis=1) < 1e-3
    side[bad] = np.cross(t[bad], [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    up = np.cross(side, t)
    radius = np.asarray(radius, np.float32)
    if radius.ndim == 1:
        radius = np.repeat(radius[:, None], sides, 1)
    radius = np.concatenate([radius, radius[:, :1]], 1)  # the seam column repeats the first
    a = 2 * np.pi * (np.arange(sides + 1) + 0.5) / sides
    ring = path[:, None, :] + radius[:, :, None] * (np.cos(a)[None, :, None] * side[:, None, :] + np.sin(a)[None, :, None] * up[:, None, :])
    n, m = len(path), sides + 1
    verts = np.concatenate([ring.reshape(-1, 3), path[:1], path[-1:]]).astype(np.float32)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    U = np.repeat((s / max(s[-1], 1e-6))[:, None], m, 1)
    V = np.repeat(((np.arange(m) + 0.5) / sides)[None, :], n, 0)
    uv = np.concatenate([np.stack([U, V], -1).reshape(-1, 2), [[0.0, 0.0], [1.0, 0.0]]]).astype(np.float32)
    k = np.arange(sides)
    faces = []
    for i in range(n - 1):
        p, q = i * m + k, i * m + k + 1
        faces += [np.stack([p, q, p + m], 1), np.stack([q, q + m, p + m], 1)]
    first, last = n * m, n * m + 1
    faces.append(np.stack([np.full(sides, first), k + 1, k], 1))
    faces.append(np.stack([np.full(sides, last), (n - 1) * m + k, (n - 1) * m + k + 1], 1))
    # Wound so the faces look outward (as _tube's are).
    return verts, np.concatenate(faces)[:, ::-1].astype(np.uint32).copy(), uv


def _knurl(t, a, pitch, depth, count):
    """Diamond knurling: two sets of fine helical grooves crossing."""
    return -depth * np.clip(np.cos(count * a[None, :] + 2 * np.pi * t[:, None] / pitch) * np.cos(count * a[None, :] - 2 * np.pi * t[:, None] / pitch), 0, 1)


def _lay(sheet, rng, ground, length, R, t, place=None):
    """Where a pencil of radius R (per point and side) lies: along an edge, partly on the
    page, resting on its widest part. Returns the axis path (N, 3) and the axis direction."""
    if place is not None:
        centre, angle = np.asarray(place[0], np.float64), float(place[1])
        axis = np.array([np.cos(angle), np.sin(angle)])
        x, y = centre[0] + (t - length / 2) * axis[0], centre[1] + (t - length / 2) * axis[1]
        z = _rest(sheet, ground, x, y, R.max(1) if R.ndim == 2 else R)
        return np.column_stack([x, y, z]), axis
    wmm, hmm = sheet.width_mm, sheet.height_mm
    side = rng.integers(4)
    along = rng.normal(0, 0.25)
    if side < 2:
        angle = along + (0 if rng.random() < 0.5 else np.pi)
        centre = np.array([rng.uniform(0.3, 0.7) * wmm, (hmm + rng.uniform(-30, 10)) if side else rng.uniform(-10, 30)])
    else:
        angle = np.pi / 2 + along + (0 if rng.random() < 0.5 else np.pi)
        centre = np.array([(wmm + rng.uniform(-30, 10)) if side == 3 else rng.uniform(-10, 30), rng.uniform(0.3, 0.7) * hmm])
    axis = np.array([np.cos(angle), np.sin(angle)])
    x, y = centre[0] + (t - length / 2) * axis[0], centre[1] + (t - length / 2) * axis[1]
    z = _rest(sheet, ground, x, y, R.max(1) if R.ndim == 2 else R)
    return np.column_stack([x, y, z]), axis


def _pencil(sheet, rng, ground, kind=None, place=None):
    """A pencil by a page written in pencil: a wooden one (hexagonal, sharpened, painted or
    bare, eraser or not, new or worn to a stub), or a mechanical one (a hexagonal metal
    drafting pencil with a knurled grip, a plastic one with grey rubber grip pads, or a
    matte silver one), sometimes with its case of leads beside it."""
    kind = kind or str(rng.choice(["wood", "drafting", "tikky", "silver"], p=[0.5, 0.2, 0.18, 0.12]))
    if kind == "wood":
        from . import pens
        return pens.pencil(sheet, rng, ground, place)
    sides = 48
    a = 2 * np.pi * (np.arange(sides) + 0.5) / sides
    things = []
    graphite = {"kind": "plastic", "colour": [0.05, 0.05, 0.055], "rough": 0.3}

    def tube(path, k, ring, material, n=sides):
        rr = ring[k]
        if rr.ndim == 2 and rr.shape[1] != n:
            rr = rr.mean(1)
        v, f = _tube(path[k], rr, n)
        return {"verts": v, "faces": f, "uv": None, "material": material}

    # Mechanical pencils.
    length = rng.uniform(138, 150)
    t = np.linspace(0, length, 900)
    lead_out = rng.uniform(0.4, 1.4)
    sleeve = 4.0 if kind == "drafting" else rng.uniform(2.5, 4.0)
    r = rng.uniform(4.0, 4.4) if kind != "tikky" else rng.uniform(4.2, 4.6)
    R = np.full((len(t), sides), r)
    # The point: lead, a thin sleeve, then a cone up to the grip.
    tip = lead_out + sleeve
    cone_end = tip + rng.uniform(8, 12)
    R[t < lead_out] = 0.27
    sl = (t >= lead_out) & (t < tip)
    R[sl] = 0.55
    cn = (t >= tip) & (t < cone_end)
    R[cn] = np.repeat((1.1 + (r * 0.92 - 1.1) * ((t[cn] - tip) / (cone_end - tip)) ** 0.9)[:, None], sides, 1)
    grip0, grip1 = cone_end, cone_end + rng.uniform(28, 36)
    grip = (t >= grip0) & (t < grip1)
    body = (t >= grip1) & (t < length - 12)
    if kind == "drafting":
        R[grip] = r * 0.94 + _knurl(t[grip], a, 0.7, 0.08, 24)
        R[body] = _hexagon(a, r)[None, :] * np.ones((int(body.sum()), 1))
    elif kind == "tikky":
        # Grey rubber pads in rings round the grip, notched into blocks.
        pitch = rng.uniform(3.6, 4.4)
        ph = ((t[grip] - grip0) / pitch) % 1
        ring = (ph > 0.15) & (ph < 0.85)
        block = np.cos(4 * a)[None, :] > -0.6
        R[grip] = r * 0.92 + 0.35 * (ring[:, None] & block)
    else:
        R[grip] = r * 0.97 + np.where((((t[grip] - grip0) / 1.6) % 1 < 0.5)[:, None], -0.12, 0.0) + _knurl(t[grip], a, 0.8, 0.05, 20)
    # The back: a cap and push button.
    back = t >= length - 12
    cap_r = r * (0.98 if kind != "drafting" else 0.8)
    R[back] = cap_r
    button = t > length - 4
    R[button] = np.repeat((cap_r * 0.75 * np.sqrt(np.clip(1 - ((t[button] - (length - 1.2)) / 1.3).clip(0) ** 2, 0.1, 1)))[:, None], sides, 1)
    support = R.copy()
    support[(t > length - 11.5) & (t < length - 7.0)] += 0.3  # the clip's collar, proud of the barrel
    path, axis = _lay(sheet, rng, ground, length, support, t, place)
    if kind == "drafting":
        colour = DRAFTING[int(rng.integers(len(DRAFTING)))]
        finish = {"kind": "metalpaint", "colour": list(colour)}
        grip_mat = finish
    elif kind == "tikky":
        colour = TIKKY[int(rng.integers(len(TIKKY)))]
        finish = {"kind": "plastic", "colour": list(colour), "rough": 0.12}
        grip_mat = {"kind": "rubber", "colour": [0.22, 0.22, 0.23]}
    else:
        finish = {"kind": "metalpaint", "colour": [0.62, 0.62, 0.64]}
        grip_mat = {"kind": "steel", "rough": 0.3}
    chrome = {"kind": "steel", "rough": 0.08}
    things.append(tube(path, t < lead_out + 0.1, R, graphite, n=12))
    things.append(tube(path, (t >= lead_out) & (t < tip + 0.1), R, chrome, n=16))
    things.append(tube(path, (t >= tip) & (t < cone_end + 0.1), R, chrome if kind != "drafting" else finish, n=sides))
    if kind == "tikky":
        things.append(tube(path, (t >= grip0) & (t < grip1), np.full_like(R, r * 0.92), finish))
        things.append(tube(path, (t >= grip0) & (t < grip1), R, grip_mat))
    else:
        things.append(tube(path, (t >= grip0 - 0.1) & (t < grip1 + 0.1), R, grip_mat))
    things.append(tube(path, (t >= grip1) & (t < length - 11.9), R, finish))
    things.append(tube(path, (t >= length - 12) & (t < length - 3.9), R, finish if kind == "drafting" else chrome))
    things.append(tube(path, t > length - 4.1, R, chrome if kind != "drafting" else finish, n=24))
    if kind == "drafting":
        # The red ring by the cap.
        k = (t > length - 12.6) & (t < length - 11.4)
        things.append(tube(path, k, np.full_like(R, cap_r + 0.05), {"kind": "plastic", "colour": [0.7, 0.05, 0.04], "rough": 0.2}, n=24))
    # The clip: a stamped strip from a collar by the cap, standing off the barrel.
    from . import pens
    ax = pens.Axis(path[:, 0], path[:, 1], path[:, 2], t, roll=float(np.clip(rng.normal(0, 0.6), -1.15, 1.15)))
    under = lambda tt: np.interp(tt, t, R.max(1))
    clip_len = rng.uniform(36, 44)
    clip = pens.Clip("steel", length - 9.0, length - 9.0 - clip_len, width=rng.uniform(3.0, 3.8), thick=rng.uniform(0.6, 0.8),
                     gap=rng.uniform(0.7, 1.0), material=finish if kind == "drafting" else chrome,
                     collar=(length - 12.0, length - 6.5, 0.3), rib=0.1)
    things += pens.clip_meshes(ax, clip, under)
    if rng.random() < 0.35:
        things += _lead_case(sheet, rng, ground, path, axis)
    return things


def _lead_case(sheet, rng, ground, path, axis):
    """A small case of pencil leads beside the pencil: a dark glossy box with a pale label."""
    across = np.array([-axis[1], axis[0]])
    mid = path[len(path) // 2, :2] + across * rng.uniform(12, 22) * rng.choice([-1, 1]) + axis * rng.uniform(-30, 30)
    turn = rng.normal(0, 0.25)
    c, s_ = np.cos(turn), np.sin(turn)
    ax = np.array([axis[0] * c - axis[1] * s_, axis[0] * s_ + axis[1] * c])
    ac = np.array([-ax[1], ax[0]])
    L, W, H = rng.uniform(70, 80), rng.uniform(17, 20), rng.uniform(6.5, 8)
    mx, my = np.array([mid[0]]), np.array([mid[1]])
    on = sample(sheet.mask, mx * sheet.ppmm, my * sheet.ppmm, fill=0.0)[0] > 0.5
    z0 = float((top(sheet, mx, my) if on else ground(mx, my))[0])

    def box(cx, l, w, h, zb, rounding):
        # A rounded box: a stadium in plan swept up, as a tube along its length.
        u = np.linspace(-l / 2, l / 2, 30)
        pts = np.column_stack([cx[0] + u * ax[0], cx[1] + u * ax[1], np.full(len(u), zb + h / 2)])
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        prof = np.stack([w / 2 * np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** rounding, h / 2 * np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** rounding], 1)
        verts, faces = [], []
        for i, p in enumerate(pts):
            ring = p[None, :] + prof[:, :1] * np.array([ac[0], ac[1], 0.0]) + prof[:, 1:] * np.array([0.0, 0.0, 1.0])
            verts.append(ring)
        verts = np.concatenate(verts + [pts[:1], pts[-1:]]).astype(np.float32)
        m, n = len(prof), len(pts)
        k = np.arange(m)
        for i in range(n - 1):
            p_, q = i * m + k, i * m + (k + 1) % m
            faces += [np.stack([p_, p_ + m, q], 1), np.stack([q, p_ + m, q + m], 1)]
        faces.append(np.stack([np.full(m, n * m), (k + 1) % m, k], 1))
        faces.append(np.stack([np.full(m, n * m + 1), (n - 1) * m + k, (n - 1) * m + (k + 1) % m], 1))
        return verts, np.concatenate(faces).astype(np.uint32)

    v, f = box(mid, L, W, H, z0, 0.25)
    body = {"verts": v, "faces": f, "uv": None, "material": {"kind": "plastic", "colour": [0.06, 0.035, 0.025], "rough": 0.06}}
    v2, f2 = box(mid - ax * L * 0.06, L * 0.7, W * 0.8, 0.3, z0 + H - 0.05, 0.5)
    label = {"verts": v2, "faces": f2, "uv": None, "material": {"kind": "matte", "colour": [0.72, 0.72, 0.72]}}
    v3, f3 = box(mid + ax * L * 0.12, L * 0.12, W * 0.8, 0.32, z0 + H - 0.04, 0.5)
    band = {"verts": v3, "faces": f3, "uv": None, "material": {"kind": "matte", "colour": [0.55, 0.25, 0.12]}}
    return [body, label, band]


def _pen(sheet, rng, ground):
    """A pen (or, when the page is in pencil, a pencil) lying along an edge, partly on the
    page, resting on what's highest under it. The pens themselves are in pens.py."""
    if sheet.ink == "pencil":
        return _pencil(sheet, rng, ground)
    from . import pens
    return pens.build(sheet, rng, ground)


# Another sheet ----------------------------------------------------------------

def _other_sheet(sheet, rng, ground):
    """Another sheet under this one, turned and shifted so that part of it shows."""
    wmm, hmm = sheet.width_mm, sheet.height_mm
    angle = rng.normal(0, 0.18)
    shift = rng.uniform(-1, 1, 2) * np.array([wmm, hmm]) * rng.uniform(0.12, 0.3)
    c, s = np.cos(angle), np.sin(angle)
    n = 120  # fine enough to follow cloth without its wrinkles poking through
    u, v = np.meshgrid(np.linspace(0, wmm, n), np.linspace(0, hmm, int(n * hmm / wmm)))
    X = wmm / 2 + shift[0] + (u - wmm / 2) * c - (v - hmm / 2) * s
    Y = hmm / 2 + shift[1] + (u - wmm / 2) * s + (v - hmm / 2) * c
    # On cloth it lies over the wrinkles' crests rather than into every hollow.
    Z = ground(X, Y)
    Z = np.maximum(Z, blur(Z, 3)) + (0.05 if np.ptp(Z) < 0.01 else 0.4)
    verts, faces = _grid(X, Y, Z)
    uv = np.column_stack([(u / wmm).ravel(), (v / hmm).ravel()]).astype(np.float32)
    # Plain or ruled paper, a little greyer than this one, perhaps written on in places.
    res = 3.0
    th, tw = int(hmm * res), int(wmm * res)
    paper = np.array(rng.choice([(0.88, 0.88, 0.85), (0.9, 0.89, 0.84), (0.93, 0.87, 0.6)]), np.float32)
    albedo = paper * (1 + 0.012 * band((th, tw), rng, 2 * res, 30 * res))[..., None]
    if rng.random() < 0.6:
        step = rng.choice([7.1, 8.7]) * res
        rows = np.arange(th)[:, None]
        line = np.exp(-((((rows - 30 * res) % step) / (0.25 * res)) ** 2)) * (rows > 25 * res)
        albedo *= 1 - 0.35 * line[..., None] * np.array([0.6, 0.35, 0.05], np.float32)
    return {"verts": verts, "faces": faces, "uv": uv, "material": {"kind": "paper", "albedo": albedo.astype(np.float32)}}


# Meshes -----------------------------------------------------------------------

def _grid(X, Y, Z):
    ny, nx = X.shape
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    i = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    return verts, faces.astype(np.uint32)


def _resample(path, step):
    """Points every `step` mm along a polyline, so a tube along it bends smoothly."""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    at = np.linspace(0, seg[-1], max(2, int(seg[-1] / step) + 1))
    return np.column_stack([np.interp(at, seg, path[:, k]) for k in range(path.shape[1])]).astype(np.float32)


def _tube(path, radius, sides, squash=1.0):
    """A tube of `radius` (per point) along `path` (N, 3), closed at both ends.

    `squash` flattens it vertically, for flat wire. `radius` may vary round the tube too,
    one per point and side (N, sides), for ribs, dimples and dents.
    """
    path = np.asarray(path, np.float32)
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    side = np.cross(t, [0.0, 0.0, 1.0])
    bad = np.linalg.norm(side, axis=1) < 1e-3
    side[bad] = np.cross(t[bad], [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    up = np.cross(side, t)
    a = 2 * np.pi * (np.arange(sides) + 0.5) / sides
    radius = np.asarray(radius, np.float32)
    rr = radius[:, :, None] if radius.ndim == 2 else radius[:, None, None]  # per point, or per point and side
    ring = (path[:, None, :] + rr * (np.cos(a)[None, :, None] * side[:, None, :]
                                     + squash * np.sin(a)[None, :, None] * up[:, None, :]))
    n = len(path)
    verts = np.concatenate([ring.reshape(-1, 3), path[:1], path[-1:]]).astype(np.float32)
    k = np.arange(sides)
    faces = []
    for i in range(n - 1):
        p, q = i * sides + k, i * sides + (k + 1) % sides
        faces += [np.stack([p, q, p + sides], 1), np.stack([q, q + sides, p + sides], 1)]
    first, last = n * sides, n * sides + 1
    faces.append(np.stack([np.full(sides, first), (k + 1) % sides, k], 1))
    faces.append(np.stack([np.full(sides, last), (n - 1) * sides + k, (n - 1) * sides + (k + 1) % sides], 1))
    # Wound so the faces look outward.
    return verts, np.concatenate(faces)[:, ::-1].astype(np.uint32)

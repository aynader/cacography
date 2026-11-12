"""Pens lying by the sheet: shapes measured from real ones, real clips, printed barrels.

A pen is built in its own frame: `t` mm from the point back along its axis, and an angle
round it (0 at its side, pi/2 where its clip is). It first rests on what is under it, as a
rigid rod (props._rest), then every part is a surface of revolution round that resting
axis, or swept along it (the clip). The whole pen is turned round its axis by `roll`, so
the clip and the printing face any way up; a hexagonal barrel always lies on a face.

Measured from real pens (and the user's reference photos):

- The point: a ball in a metal socket 1.6-2.6 mm long, then a nose easing into the
  barrel over 8-14 mm, straight or slightly concave, never a stubby 3.5 mm cone.
- The clip is a part of its own: a collar round the barrel (or moulded into the cap or
  the back end), a strip 2.8-6 mm wide and 0.5-1.6 mm thick standing 0.6-1.2 mm off the
  barrel, ending in a rounded tip that touches it. It casts its own shadow.
- A soft grip starts at the end of the nose, where the fingers go, and runs 30-40 mm.
- Plastic is glossy, its gloss a little uneven (mould lines, scuffs, finger marks).
"""
from dataclasses import dataclass, field

import numpy as np

from . import props as P
from .imaging import band, blur, smoothstep

STYLES = {"stick": 0.2, "clear": 0.18, "grip": 0.18, "gel": 0.16, "click": 0.16, "capped": 0.12}
COLOURS = {
    # solid barrels: colour, roughness (principled, alpha = roughness squared)
    "black": ((0.025, 0.025, 0.03), 0.25), "blue": ((0.05, 0.12, 0.42), 0.25), "white": ((0.8, 0.8, 0.77), 0.29),
    "red": ((0.52, 0.04, 0.05), 0.25), "grey": ((0.25, 0.26, 0.28), 0.31), "green": ((0.06, 0.3, 0.15), 0.27),
    "yellow": ((0.86, 0.66, 0.06), 0.25), "pink": ((0.8, 0.33, 0.48), 0.27), "orange": ((0.88, 0.36, 0.04), 0.27),
    "sky": ((0.25, 0.5, 0.78), 0.27), "cream": ((0.78, 0.74, 0.62), 0.33),
}
TINTS = [(0.97, 0.97, 0.97)] * 3 + [(0.6, 0.75, 0.98), (0.98, 0.65, 0.7), (0.7, 0.95, 0.75), (0.98, 0.9, 0.6), (0.8, 0.7, 0.95)]
RUBBER = [(0.03, 0.03, 0.035), (0.03, 0.03, 0.035), (0.03, 0.03, 0.035), (0.12, 0.12, 0.13), (0.05, 0.1, 0.3),
          (0.35, 0.04, 0.05), (0.05, 0.22, 0.12), (0.4, 0.4, 0.42)]
INK = {"blue": (0.04, 0.06, 0.3), "black": (0.02, 0.02, 0.025), "red": (0.35, 0.03, 0.04), "green": (0.03, 0.2, 0.06)}
GEL_TINT = {"blue": (0.2, 0.36, 0.95), "black": (0.16, 0.16, 0.18), "red": (0.95, 0.22, 0.2), "green": (0.2, 0.8, 0.36)}
# Made-up names for the printing on a barrel: never a real brand.
NAMES = ["Scriva", "Velto", "Tessa", "Corvo", "Nibli", "Strito", "Arka", "Linea", "Kalo", "Drevo", "Plumo", "Ondi"]
SUFFIXES = ["0.7", "1.0 M", "FINE", "Medium", "0.5", "GEL 0.5", "ball 0.7", "F", "M", "0.38", "fine", "1.0"]
STEEL = {"kind": "steel", "rough": 0.12}
CHROME = {"kind": "chrome", "rough": 0.06}

FONTS = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arial.ttf"]


# Geometry --------------------------------------------------------------------------

def pchip(xk, yk, x):
    """Monotone cubic through the knots (xk, yk): no overshoot between them, so a profile
    measured as a few (t, radius) pairs stays as smooth and as monotone as the real one."""
    xk, yk = np.asarray(xk, np.float64), np.asarray(yk, np.float64)
    x = np.asarray(x, np.float64)
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(xk) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    i = np.clip(np.searchsorted(xk, x) - 1, 0, len(h) - 1)
    s = np.clip((x - xk[i]) / h[i], 0, 1)
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    out = h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1]
    return np.where(x < xk[0], yk[0], np.where(x > xk[-1], yk[-1], out))


class Axis:
    """A straight resting axis: where a pen lies, and its frame (side, up)."""

    def __init__(self, x, y, z, t, roll=0.0):
        self.t = np.asarray(t, np.float64)
        self.p = np.column_stack([x, y, z]).astype(np.float64)
        d = self.p[-1] - self.p[0]
        self.d = d / np.linalg.norm(d)
        side = np.cross(self.d, [0.0, 0.0, 1.0])
        self.side = side / np.linalg.norm(side)
        self.up = np.cross(self.side, self.d)
        self.roll = float(roll)

    def at(self, t):
        t = np.asarray(t, np.float64)
        return np.stack([np.interp(t, self.t, self.p[:, k], left=None, right=None) +
                         self.d[k] * (np.minimum(t - self.t[0], 0) + np.maximum(t - self.t[-1], 0)) for k in range(3)], -1)

    def point(self, t, a, rad):
        """Points at `t` (N,), angles `a` round the pen in its own frame, radius `rad`: (N, M, 3)."""
        t = np.asarray(t, np.float64)
        a = np.asarray(a, np.float64) + self.roll
        rad = np.asarray(rad, np.float64)
        c = np.cos(a)[..., None]
        s = np.sin(a)[..., None]
        if c.ndim == 2:
            c, s = c[None], s[None]
        return self.at(t)[:, None, :] + rad[..., None] * (c * self.side + s * self.up)


_SIGN = None


def _volume(verts, faces):
    v = verts[faces.astype(np.int64)]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6


def orient(verts, faces):
    """Faces wound the way props._tube winds them (looking outward), whatever the loop order."""
    global _SIGN
    if _SIGN is None:
        v, f = P._tube(np.array([[0, 0, 0], [10, 0, 0.0]], np.float32), np.array([1.0, 1.0], np.float32), 12)
        _SIGN = np.sign(_volume(v, f))
    if np.sign(_volume(verts, faces)) != _SIGN:
        faces = faces[:, ::-1].copy()
    return faces


def rings(pts, caps=True):
    """A closed tube through rings of points (N, M, 3), with texture coordinates (u along by
    the rings' centres, v round): (verts, faces, uv). The seam has its own vertices."""
    n, m, _ = pts.shape
    ring = np.concatenate([pts, pts[:, :1]], 1)
    centre = pts.mean(1)
    verts = np.concatenate([ring.reshape(-1, 3), centre[:1], centre[-1:]]).astype(np.float32)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    U = np.repeat((s / max(s[-1], 1e-6))[:, None], m + 1, 1)
    V = np.repeat(((np.arange(m + 1) + 0.5) / m)[None, :], n, 0)
    uv = np.concatenate([np.stack([U, V], -1).reshape(-1, 2), [[0.0, 0.0], [1.0, 0.0]]]).astype(np.float32)
    k = np.arange(m)
    w = m + 1
    faces = []
    for i in range(n - 1):
        p, q = i * w + k, i * w + k + 1
        faces += [np.stack([p, q, p + w], 1), np.stack([q, q + w, p + w], 1)]
    if caps:
        first, last = n * w, n * w + 1
        faces.append(np.stack([np.full(m, first), k + 1, k], 1))
        faces.append(np.stack([np.full(m, last), (n - 1) * w + k, (n - 1) * w + k + 1], 1))
    faces = np.concatenate(faces).astype(np.uint32)
    return verts, orient(verts, faces), uv


@dataclass
class Part:
    """One piece of a pen: from t0 to t1 mm, its radius per t and angle, and its surface."""
    t0: float
    t1: float
    radius: object  # f(t (N,), a (M,)) -> (N, M)
    material: dict
    sides: int = 64
    step: float = 0.25  # mm between rings
    maps: object = None  # f(span_mm, around_mm) -> material extras with textures, for printed barrels
    flat: bool = False

    def mesh(self, axis):
        n = max(4, int(np.ceil((self.t1 - self.t0) / self.step)) + 1)
        t = np.linspace(self.t0, self.t1, n)
        a = 2 * np.pi * (np.arange(self.sides) + 0.5) / self.sides
        R = np.broadcast_to(self.radius(t, a), (n, self.sides))
        verts, faces, uv = rings(axis.point(t, a, R))
        material = dict(self.material)
        if self.maps is not None:
            material.update(self.maps(self.t1 - self.t0, 2 * np.pi * float(R.mean())))
        return {"verts": verts, "faces": faces, "uv": uv, "material": material, "flat": self.flat}


def round_(f):
    """A radius that depends only on t."""
    return lambda t, a: np.asarray(f(t))[:, None] * np.ones((1, len(a)))


def hexagon(r_in):
    return lambda t, a: P._hexagon(np.asarray(a), r_in)[None, :] * np.ones((len(t), 1))


def capped_end(r, t_end, rounding):
    """Radius scale near a rounded end at t_end (1 well before it, 0 at it)."""
    return lambda t: np.sqrt(np.clip(1 - (np.clip(np.asarray(t) - (t_end - rounding), 0, None) / rounding) ** 2, 0.0025, 1))


def front_end(t0, rounding):
    """The same for an end facing the point, at t0."""
    return lambda t: np.sqrt(np.clip(1 - (np.clip(t0 + rounding - np.asarray(t), 0, None) / rounding) ** 2, 0.0025, 1))


# Surfaces --------------------------------------------------------------------------

def _font(px):
    from PIL import ImageFont
    for name in FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def barrel_maps(rng, colour, rough, words="", ink=(0.95, 0.95, 0.93), at=(0.4, 0.75), v=0.5, metal=None,
                grip=None, worn=0.0, clear=False):
    """Textures for a plastic barrel, as a function of its size: the colour with printing on
    it, and an uneven gloss. Rows run round the barrel, columns along it (from the point).

    The gloss: a faint low-contrast unevenness, two mould seams along it, a few scuffs and
    finger marks (denser where it is held, `grip` = (t0, t1) mm from the part's start).
    """
    from PIL import Image, ImageDraw

    seed = int(rng.integers(2**31))

    def make(span, around):
        r = np.random.default_rng(seed)
        ppmm = 8.0
        nu, nv = max(32, int(span * ppmm)), max(32, int(around * ppmm))
        rgh = np.full((nv, nu), rough, np.float32)
        rgh += 0.025 * band((nv, nu), r, 6, 120)
        # Two mould seams along the barrel, opposite each other.
        rows = np.arange(nv)[:, None]
        for c in (0.0, 0.5):
            d = np.abs(((rows / nv - c + 0.5) % 1.0) - 0.5) * around
            rgh += 0.06 * np.exp(-(d / 0.12) ** 2)
        # Scuffs and finger marks: soft blotches of duller gloss.
        blot = np.clip(band((nv, nu), r, 12, 80) - 1.2, 0, None)
        weight = np.full(nu, 0.4 + worn, np.float32)
        if grip is not None:
            cols = np.arange(nu) / ppmm
            weight += 1.2 * smoothstep(grip[0] - 8, grip[0], cols) * smoothstep(grip[1] + 8, grip[1], cols)
        rgh += 0.12 * blot * weight[None, :]
        if worn > 0:
            # Fine scratches: short lines, mostly along the barrel.
            for _ in range(int(40 * worn * span / 100)):
                y0, x0 = r.uniform(0, nv), r.uniform(0, nu)
                ang = r.normal(0, 0.35)
                L = r.uniform(4, 30)
                ts = np.linspace(0, L, int(L * 2))
                xs = (x0 + ts * np.cos(ang)).astype(int) % nu
                ys = (y0 + ts * np.sin(ang)).astype(int) % nv
                rgh[ys, xs] += r.uniform(0.08, 0.2)
        albedo = np.repeat(np.repeat(np.asarray(colour, np.float32)[None, None, :], nv, 0), nu, 1)
        albedo *= (1 + 0.015 * band((nv, nu), r, 8, 100))[..., None]
        mask = np.zeros((nv, nu), np.float32)
        if words:
            px = int(2.3 * ppmm)
            img = Image.new("L", (nu, px + 6), 0)
            d = ImageDraw.Draw(img)
            x = int(at[0] * nu)
            d.text((x, 2), words, fill=255, font=_font(px))
            m = np.asarray(img, np.float32) / 255
            m[:, int(at[1] * nu):] = 0
            m *= np.clip(0.9 + 0.2 * band(m.shape, r, 2, 10), 0, 1)  # printing is never quite even
            row0 = int(v * nv - m.shape[0] / 2)
            rr = (row0 + np.arange(m.shape[0])) % nv
            mask[rr] = np.maximum(mask[rr], m)
            if metal is None:
                albedo = albedo * (1 - mask[..., None]) + np.asarray(ink, np.float32) * mask[..., None]
                rgh = rgh * (1 - mask) + (rough + 0.12) * mask  # printing is a little duller than the plastic
        out = {"albedo": albedo.clip(0, 1).astype(np.float32), "rough_map": rgh.clip(0.05, 0.95).astype(np.float32)}
        if metal is not None and words:
            out["foil"], out["metal"] = mask, metal
        return out

    return make


# Clips -----------------------------------------------------------------------------

@dataclass
class Clip:
    """A pocket clip. Its root is at `t_root` (in a collar, or moulded into the barrel or
    cap), its free end at `t_tip`, either way along the pen."""
    kind: str  # "steel" (stamped strip, with a collar), "plastic" (moulded), "wire" (bent round wire)
    t_root: float
    t_tip: float
    width: float
    thick: float
    gap: float  # stand-off along its length, mm
    material: dict
    collar: tuple = None  # (t0, t1, proud mm) or None
    rib: float = 0.0  # a faint embossed rib along a stamped strip
    angle: float = np.pi / 2  # round the pen, in its own frame
    extra: dict = field(default_factory=dict)


def _section(w, th, p=0.32, m=28):
    """A rounded rectangle, w across and th thick, as (x, y) round its loop (y from 0 to th)."""
    q = 2 * np.pi * np.arange(m) / m
    x = 0.5 * w * np.sign(np.cos(q)) * np.abs(np.cos(q)) ** p
    y = 0.5 * th + 0.5 * th * np.sign(np.sin(q)) * np.abs(np.sin(q)) ** p
    return x, y


def clip_meshes(axis, clip, barrel):
    """The clip as meshes: strip (or wire) and collar. `barrel(t)` is the barrel's radius
    under the clip, mm."""
    out = []
    sign = 1.0 if clip.t_tip > clip.t_root else -1.0
    span = abs(clip.t_tip - clip.t_root)
    rb = lambda t: np.asarray(barrel(np.atleast_1d(np.asarray(t, np.float64))), np.float64)
    if clip.collar is not None:
        c0, c1, proud = clip.collar
        rc = lambda t: (rb(t) + proud) * np.minimum(front_end(c0, 0.5)(t), capped_end(1, c1, 0.5)(t))
        part = Part(c0, c1, round_(rc), clip.material, sides=64, step=0.1)
        out.append(part.mesh(axis))
    if clip.kind == "wire":
        # A round wire: from the collar along one side, round a U at the end, back along the other.
        r_w = clip.thick / 2
        n = 120
        s = np.linspace(0, 1, n)
        t_leg = clip.t_root + sign * s * (span - clip.width / 2)
        half = clip.width / 2 - r_w
        g = clip.gap + r_w
        lift = lambda t: g * smoothstep(0.0, 3.0, np.abs(t - clip.t_root)) + (clip.collar[2] if clip.collar else 0) * (1 - smoothstep(0.0, 3.0, np.abs(t - clip.t_root)))
        legs = []
        for side_ in (-1, 1):
            tt = t_leg if side_ < 0 else t_leg[::-1]
            rad = rb(tt) + lift(tt) * (1 - 0.45 * smoothstep(0.6, 1.0, np.abs(tt - clip.t_root) / span))
            ang = clip.angle + side_ * (half * (1 - 0.15 * np.abs(tt - clip.t_root) / span)) / rad
            legs.append((tt, ang, rad))
        # The U at the free end, bending down to touch the barrel.
        q = np.linspace(-np.pi / 2, np.pi / 2, 40)
        t_end = clip.t_root + sign * (span - clip.width / 2)
        tu = t_end + sign * (clip.width / 2 - r_w) * np.cos(q) * 0.9
        hw = half * 0.85
        leg_end = float(legs[0][2][-1] - rb(legs[0][0][-1:])[0])
        rad_u = rb(tu) + r_w + 0.05 + (leg_end - r_w - 0.05) * (1 - np.cos(q))
        ang_u = clip.angle + hw * np.sin(q) / rad_u
        tt = np.concatenate([legs[0][0], tu, legs[1][0]])
        aa = np.concatenate([legs[0][1], ang_u, legs[1][1]])
        rr = np.concatenate([legs[0][2], rad_u, legs[1][2]])
        pts = axis.point(tt, aa[:, None], rr[:, None])[:, 0, :]
        pts = P._resample(pts, 0.15)
        v, f = P._tube(pts, np.full(len(pts), r_w, np.float32), 14)
        out.append({"verts": v, "faces": f, "uv": None, "material": clip.material})
        return out
    # A strip, its section a rounded rectangle bent round the barrel.
    n = max(60, int(span / 0.2))
    s = np.linspace(0, 1, n)
    t = clip.t_root + sign * s * span
    d_root = s * span  # distance from the root
    d_tip = (1 - s) * span  # distance from the free end
    g = np.full(n, clip.gap)
    if clip.kind == "plastic":
        # Moulded in: the root sinks into the barrel, then rises in a fillet to the stand-off.
        bend = clip.extra.get("bend", 6.0)
        g = -0.6 * clip.thick + (clip.gap + 0.6 * clip.thick) * smoothstep(0.0, bend, d_root)
    else:
        # From the collar's top, a rounded bend up to the stand-off.
        base = (clip.collar[2] - 0.05) if clip.collar else 0.0
        g = base + (clip.gap - base) * smoothstep(0.5, 3.5, d_root)
    g = g - 0.35 * clip.gap * smoothstep(0.35, 1.0, s)  # it leans in towards its tip
    w = np.full(n, clip.width)
    hw = clip.width / 2
    w = w * np.sqrt(np.clip(1 - (np.clip(hw - d_tip, 0, None) / hw) ** 2, 0.0, 1))  # a rounded end, in plan
    th = np.full(n, clip.thick)
    th = th * (0.55 + 0.45 * np.sqrt(np.clip(d_tip / 0.8, 0, 1)))  # thinner at the very end
    # The tip: a nub that comes down to touch the barrel.
    nub = 1 - smoothstep(0.4, 2.8, d_tip)
    drop = (g - 0.03) * nub
    q = 2 * np.pi * np.arange(28) / 28
    pts = np.zeros((n, 28, 3))
    rbt = rb(t)
    for i in range(n):
        x, y = _section(max(w[i], 0.02), th[i] + drop[i])
        if clip.rib > 0:
            y = y + clip.rib * np.exp(-(x / (0.18 * clip.width)) ** 2) * (y > 0.5 * (th[i] + drop[i]))
        r0 = rbt[i] + g[i] - drop[i]
        rad = r0 + y
        ang = clip.angle + x / (r0 + 0.5 * th[i])
        pts[i] = axis.point(np.array([t[i]]), ang[None, :], rad[None, :])[0]
    v, f, uv = rings(pts)
    out.append({"verts": v, "faces": f, "uv": None, "material": clip.material})
    return out


# Pens ------------------------------------------------------------------------------

def ball_part(ball, ink):
    """The ball, a sphere half out of its socket, the ink's colour on it."""
    c = 0.45 * ball
    return Part(c - ball, c + 0.6 * ball, round_(lambda t: np.sqrt(np.clip(ball**2 - (np.asarray(t) - c) ** 2, 1e-4, None))),
                {"kind": "pplastic", "colour": list(ink), "rough": 0.18}, sides=32, step=0.02)


def _point(rng, ball=None):
    """The ball and its metal socket: (knots t, knots r, socket length, socket end radius)."""
    ball = rng.uniform(0.28, 0.4) if ball is None else ball
    tl = rng.uniform(1.6, 2.6)
    re = rng.uniform(0.7, 0.95)
    return [0.0, 0.35, tl], [ball, ball + 0.12, re], tl, re


def build(sheet, rng, ground, style=None, place=None):
    """A pen by the page: things for props.dress. `style` and `place` ((x, y) of its middle,
    angle) are drawn at random unless given."""
    style = style or str(rng.choice(list(STYLES), p=list(STYLES.values())))
    ink_name = sheet.ink if sheet.ink in INK else "blue"
    ink = list(INK[ink_name])
    worn = rng.random() < 0.3
    parts, clips = [], []
    hexa = False
    roll = 0.0

    def colour(prefer_ink=0.4):
        name = ink_name if rng.random() < prefer_ink else str(rng.choice(list(COLOURS)))
        return COLOURS[name]

    def label():
        if rng.random() < 0.25:
            return ""
        name = str(rng.choice(NAMES))
        return name + ("  " + str(rng.choice(SUFFIXES)) if rng.random() < 0.7 else "")

    def printed(col, rough, light_ink=None, grip=None, metal=None):
        dark = float(np.mean(col)) < 0.35
        ink_col = light_ink if light_ink is not None else ((0.92, 0.92, 0.9) if dark else (0.03, 0.03, 0.035))
        if rng.random() < 0.3 and dark:
            ink_col = (0.75, 0.6, 0.25)  # gold-coloured print
        v = float(rng.choice([0.5, 0.0, 0.75]))
        return barrel_maps(rng, col, rough, label(), ink_col, at=(rng.uniform(0.35, 0.5), 0.9), v=v, metal=metal,
                           grip=grip, worn=0.6 if worn else 0.1)

    def plastic(col, rough, maps=None):
        return {"kind": "pplastic", "colour": list(col), "rough": rough}

    tk, rk, tl, re = _point(rng)
    socket = {"kind": "steel", "rough": rng.uniform(0.1, 0.25)}
    length = rng.uniform(136, 148)
    extra = 0.0  # what sticks out past the barrel's end (plunger, posted cap)
    cap_front = None

    if style == "stick":
        r = rng.uniform(3.7, 4.2)
        hexa = rng.random() < 0.2
        nose_end = tl + rng.uniform(9, 13)
        gam = rng.uniform(0.9, 1.3)
        col, rough = colour()
        prof_t = tk + [tl + 0.01, tl + 0.4] + list(tl + 0.4 + (nose_end - tl - 0.4) * np.linspace(0.15, 1, 7)) + [nose_end + 1.8]
        n0 = re + 0.15
        u = np.linspace(0.15, 1, 7)
        prof_r = rk + [n0, n0 + 0.03] + list(n0 + (0.9 * r - n0) * u**gam) + [r * (1.16 if hexa else 1.0)]
        prof = lambda t, pt=prof_t, pr=prof_r: pchip(pt, pr, t)
        if hexa:
            body_r = lambda t, a, pr=prof: np.minimum(P._hexagon(np.asarray(a), r)[None, :], pr(t)[:, None] * 1.16)
        else:
            body_r = round_(prof)
        end = capped_end(1, length, 1.2)
        parts.append(Part(0.0, tl + 0.02, round_(prof), {"kind": "steel", "rough": 0.15}, sides=48, step=0.05))
        parts.append(ball_part(rk[0], ink))
        bm = printed(col, rough)
        parts.append(Part(tl, length - 5.5, body_r, plastic(col, rough), sides=6 if hexa else 72, maps=bm, flat=hexa))
        # The plug at the back: the ink's colour, or the barrel's.
        plug = ink if rng.random() < 0.6 else list(col)
        parts.append(Part(length - 6.0, length, round_(lambda t: 0.93 * r * end(t)), plastic(plug, 0.3), sides=48, step=0.1))
        # A cap: on the point, posted on the back, or lost.
        cap = rng.choice(["front", "back", "none"], p=[0.35, 0.35, 0.3])
        cap_col = ink if rng.random() < 0.7 else list(col)
        cap_col = [min(1.0, c * 1.6 + 0.02) for c in cap_col]
        if cap != "none":
            cl = rng.uniform(44, 52)
            rc = r * (1.16 if hexa else 1.0) + 0.55
            if cap == "front":
                c0, c1 = -2.0, cl - 2.0
                cap_front = (c0, c1)
                rcap = lambda t, c0=c0, c1=c1: rc * front_end(c0, 2.0)(t) * (1 - 0.06 * smoothstep(c1 - 3, c1, t))
                root, tip_end = c0 + 3.0, c0 + 3.0 + rng.uniform(32, 40)
            else:
                c0, c1 = length - 20, length + cl - 20
                extra = cl - 20
                rcap = lambda t, c0=c0, c1=c1: rc * capped_end(1, c1, 2.0)(t) * (1 - 0.06 * (1 - smoothstep(c0, c0 + 3, t)))
                root, tip_end = c1 - 3.0, c1 - 3.0 - rng.uniform(32, 40)
            parts.append(Part(c0, c1, round_(rcap), plastic(cap_col, 0.28), sides=64, step=0.15))
            clips.append(Clip("plastic", root, tip_end, width=rng.uniform(0.4, 0.55) * 2 * rc, thick=rng.uniform(1.0, 1.4),
                              gap=rng.uniform(0.5, 0.9), material=plastic(cap_col, 0.28), extra={"barrel": rcap}))

    elif style == "clear":
        # A clear hexagonal barrel (a Cristal-like stick): a brass point, a white tip holder,
        # the barrel's cone cut into its hexagon, the ink tube inside, a plug at the back.
        r = rng.uniform(3.6, 3.85)
        hexa = rng.random() < 0.8
        holder = rng.uniform(2.5, 3.5)
        cone0, cone1 = tl + holder, tl + holder + rng.uniform(12, 16)
        rh = rng.uniform(1.2, 1.35)
        cone = lambda t: pchip([cone0, cone0 + 1, cone1], [rh + 0.35, rh + 0.6, r * 1.16], t)
        shape = (lambda t, a: np.minimum(P._hexagon(np.asarray(a), r)[None, :], cone(t)[:, None])) if hexa else round_(lambda t: np.minimum(r, cone(t)))
        brass = rng.random() < 0.6
        parts.append(Part(0.0, tl + 0.05, round_(lambda t: pchip(tk, rk, t)), {"kind": "brass", "rough": 0.2} if brass else socket, sides=48, step=0.05))
        parts.append(ball_part(rk[0], ink))
        parts.append(Part(tl, cone0 + 0.6, round_(lambda t: rh * front_end(tl, 0.3)(t)), plastic((0.85, 0.85, 0.82), 0.35), sides=48, step=0.1))
        tint = list(TINTS[int(rng.integers(len(TINTS)))])
        parts.append(Part(cone0, length - 0.2, shape, {"kind": "clear", "tint": tint, "alpha": 0.04}, sides=6 if hexa else 64, step=0.5, flat=hexa))
        tube_r = rng.uniform(1.0, 1.2)
        level = rng.uniform(0.25, 0.9) * (length - 20)
        parts.append(Part(cone0 + 0.5, cone0 + level, round_(lambda t: np.full(len(t), tube_r)), plastic(ink, 0.25), sides=24, step=1.0))
        parts.append(Part(cone0 + level - 0.2, length - 9, round_(lambda t: np.full(len(t), tube_r * 0.98)), plastic((0.82, 0.82, 0.8), 0.3), sides=24, step=1.0))
        plug = ink if rng.random() < 0.7 else [0.05, 0.05, 0.05]
        end = capped_end(1, length + 1.5, 1.0)
        parts.append(Part(length - 8, length + 1.5, round_(lambda t: (r * 0.92) * end(t)), plastic([min(1, c * 1.5) for c in plug], 0.3), sides=48, step=0.1))
        cap = rng.choice(["front", "back", "none"], p=[0.35, 0.25, 0.4])
        cap_col = [min(1.0, c * 1.6 + 0.02) for c in ink]
        if cap != "none":
            cl = rng.uniform(46, 52)
            rc = r * 1.16 + 0.5
            if cap == "front":
                c0, c1 = -2.5, cl - 2.5
                cap_front = (c0, c1)
                rcap = lambda t, c0=c0: rc * front_end(c0, 1.5)(t)
                root, tip_end = c0 + 2.5, c0 + 2.5 + rng.uniform(32, 38)
            else:
                c0, c1 = length - 18, length - 18 + cl
                extra = cl - 18
                rcap = lambda t, c1=c1: rc * capped_end(1, c1, 1.5)(t)
                root, tip_end = c1 - 2.5, c1 - 2.5 - rng.uniform(32, 38)
            parts.append(Part(c0, c1, round_(rcap), plastic(cap_col, 0.28), sides=64, step=0.15))
            clips.append(Clip("plastic", root, tip_end, width=rng.uniform(3.2, 4.0), thick=rng.uniform(1.1, 1.5),
                              gap=rng.uniform(0.6, 1.0), material=plastic(cap_col, 0.28), extra={"barrel": rcap}))

    elif style in ("grip", "gel", "click"):
        kind = {"grip": str(rng.choice(["soft", "rubber", "dimpled"], p=[0.4, 0.25, 0.35])),
                "gel": str(rng.choice(["rings", "dimpled"], p=[0.6, 0.4])), "click": "plain"}[style]
        r = rng.uniform(4.3, 5.0) if style != "click" else rng.uniform(4.0, 4.7)
        # The nose: metal (or metal-look plastic) easing from the socket into the barrel.
        nose_end = tl + rng.uniform(8, 12)
        gam = rng.uniform(1.0, 1.4)
        n0 = re + 0.12
        u = np.linspace(0.12, 1, 8)
        r_front = 0.82 * r if kind != "plain" else 0.97 * r
        stepped = rng.random() < 0.5 and kind in ("rings", "dimpled", "plain")
        nose_r = n0 + (r_front - n0) * u**gam
        if stepped:  # a couple of little steps on the cone, as on many gel pens
            nose_r = nose_r - 0.12 * ((u < 0.35) + (u < 0.65))
        prof_t = tk + [tl + 0.01] + list(tl + (nose_end - tl) * u)
        prof_r = rk + [n0 - 0.05] + list(np.maximum.accumulate(nose_r))
        nose = lambda t, pt=prof_t, pr=prof_r: pchip(pt, pr, t)
        nose_mat = CHROME if kind in ("soft", "rings") or rng.random() < 0.5 else {"kind": "steel", "rough": 0.25}
        if style == "click" and rng.random() < 0.4:
            nose_mat = None  # moulded in the barrel's own plastic (only on a solid barrel, below)
        g0 = nose_end + (rng.uniform(0.0, 1.5) if kind != "soft" else 0.0)
        g1 = g0 + rng.uniform(30, 40)
        if kind == "rubber":
            g1 = g0 + rng.uniform(18, 24)
        body_kind = "clear" if kind in ("rings", "dimpled") else "solid"
        if style == "click" and rng.random() < 0.25:
            body_kind = "clear"
        if body_kind == "clear" and nose_mat is None:
            nose_mat = CHROME
        col, rough = colour(0.3)
        if kind == "soft":
            body_mat = {"kind": "metalpaint", "colour": list(rng.choice([(0.18, 0.2, 0.22), (0.05, 0.12, 0.4), (0.35, 0.05, 0.06), (0.05, 0.05, 0.06), (0.5, 0.5, 0.52)]))}
            body_maps = None
        elif kind == "rubber":
            body_mat = {"kind": "rubber", "colour": list(rng.choice([(0.05, 0.05, 0.055), (0.12, 0.12, 0.13), (0.06, 0.1, 0.25)]))}
            body_maps = None
        elif body_kind == "clear":
            body_mat = {"kind": "clear", "tint": list(TINTS[int(rng.integers(len(TINTS)))]) if style != "gel" else [0.97, 0.97, 0.97], "alpha": 0.04}
            body_maps = None
        else:
            body_mat = plastic(col, rough)
            body_maps = printed(col, rough, grip=(0, 40))
        end_t = length
        endr = capped_end(1, end_t, 1.5)
        barrel = lambda t: np.full(len(np.atleast_1d(t)), r)
        parts.append(ball_part(rk[0], ink))
        parts.append(Part(0.0, tl + 0.02, round_(lambda t: pchip(tk, rk, t)), socket, sides=48, step=0.05))
        if nose_mat is not None:
            parts.append(Part(tl, nose_end + 0.3, round_(nose), nose_mat, sides=72, step=0.1))
        else:
            parts.append(Part(tl, nose_end + 0.3, round_(nose), plastic(col, rough), sides=72, step=0.1))
        # The barrel, with its grip.
        if kind == "soft":
            bulge = lambda t: r * (1.01 + 0.09 * np.sin(np.pi * np.clip((t - g0) / (g1 - g0), 0, 1)) ** 0.8)
            parts.append(Part(nose_end, length - 7, round_(lambda t: np.full(len(t), r)), body_mat, sides=72, step=0.5))
            rub = list(RUBBER[int(rng.integers(len(RUBBER)))])
            parts.append(Part(g0 + 0.4, g1 - 0.4, round_(bulge), {"kind": "rubber", "colour": rub}, sides=72, step=0.25))
            for at in (g0, g1):  # chrome rings at both ends of the grip
                rr = float(bulge(np.array([at]))[0]) + 0.15
                parts.append(Part(at - 0.7, at + 0.7, round_(lambda t, rr=rr, at=at: rr * np.minimum(front_end(at - 0.7, 0.3)(t), capped_end(1, at + 0.7, 0.3)(t))), CHROME, sides=64, step=0.05))
            # The top: a chrome cap with the clip's collar.
            parts.append(Part(length - 7.2, length, round_(lambda t: r * 0.99 * endr(t)), CHROME, sides=64, step=0.1))
        elif kind == "rubber":
            pitch = rng.uniform(2.2, 2.8)
            groove = lambda t: r - 0.35 * ((t > nose_end + 2) & (t < nose_end + 2 + 6 * pitch) & (np.cos(2 * np.pi * (t - nose_end - 2) / pitch) > 0.75))
            parts.append(Part(nose_end, length, round_(lambda t: groove(t) * endr(t)), body_mat, sides=72, step=0.15))
        else:
            ease_in, ease_out = 1.8, 3.0
            back0 = length - rng.uniform(14, 20) if (style in ("gel", "click") or kind == "dimpled") else length - 0.4
            rg = r * rng.uniform(1.04, 1.08)
            if kind == "rings":
                n_r = int(rng.integers(8, 13))
                gr = lambda t: (r + (rg - r) * smoothstep(g0, g0 + ease_in, t) * smoothstep(g1, g1 - ease_out, t)
                                + 0.07 * ((t > g0 + 1.2) & (t < g0 + 1.2 + n_r * 0.8)) * np.maximum(np.cos(2 * np.pi * (t - g0) / 0.8), 0) ** 2)
                grip_r = round_(gr)
                grip_mat = {"kind": "clear", "tint": list(GEL_TINT.get(ink_name, GEL_TINT["blue"])), "alpha": 0.35, "cloud": 0.7}
                sides = 72
            elif kind == "dimpled":
                pitch = rng.uniform(1.5, 1.8)
                n_a = int(round(2 * np.pi * rg / rng.uniform(1.5, 1.8)))
                depth, size = rng.uniform(0.1, 0.16), rng.uniform(0.45, 0.6)

                def grip_r(t, a, pitch=pitch, n_a=n_a, depth=depth, size=size, rg=rg):
                    base = r + (rg - r) * smoothstep(g0, g0 + ease_in, t) * smoothstep(g1, g1 - ease_out, t)
                    out = np.repeat(base[:, None], len(a), 1)
                    zone = smoothstep(g0 + 2.0, g0 + 3.0, t) * smoothstep(g1 - 3.5, g1 - 4.5, t)
                    u = (t - g0) / pitch
                    for k in (np.floor(u), np.ceil(u)):
                        dt = (u - k)[:, None] * pitch
                        ph = (k % 2)[:, None] * 0.5
                        da = ((a[None, :] * n_a / (2 * np.pi) - ph + 0.5) % 1.0 - 0.5) * (2 * np.pi * rg / n_a)
                        out -= depth * zone[:, None] * np.exp(-(dt**2 + da**2) / size**2)
                    return out
                grip_mat = {"kind": "rubber", "colour": list(RUBBER[int(rng.integers(len(RUBBER)))])}
                sides = 160
            else:
                grip_r, grip_mat, sides = None, None, 72
            if body_kind == "clear":
                parts.append(Part(nose_end - 0.2, back0 + 0.6, round_(lambda t: np.full(len(t), r)), body_mat, sides=72, step=0.5))
                tube_r = rng.uniform(1.0, 1.2)
                level = rng.uniform(0.3, 0.9) * (length - 30)
                parts.append(Part(tl + 1.5, tl + 1.5 + level, round_(lambda t: np.full(len(t), tube_r)), plastic(ink, 0.25), sides=24, step=1.0))
                parts.append(Part(tl + 1.5 + level - 0.2, length - 8, round_(lambda t: np.full(len(t), tube_r * 0.98)), plastic((0.85, 0.85, 0.82), 0.3), sides=24, step=1.0))
                spring = Part(length - 20, length - 8, round_(lambda t: 1.3 + 0.25 * (np.cos(2 * np.pi * t / 1.1) > 0.6)), STEEL, sides=16, step=0.1)
                parts.append(spring)
            else:
                parts.append(Part(nose_end - 0.2, back0 + 0.6, round_(lambda t: np.full(len(t), r)), body_mat, sides=72, step=0.5, maps=body_maps))
            if grip_r is not None:
                lift = lambda t, a, f=grip_r: f(t, a) + 0.02
                parts.append(Part(g0, g1, lift, grip_mat, sides=sides, step=0.12 if kind == "dimpled" else 0.2))
        # The back: a plunger out of the end (click pens), or the end itself.
        bright = [min(1.0, c * 1.7 + 0.02) for c in ink]
        trim_col = bright if style == "gel" else (list(col) if body_kind == "solid" else list(rng.choice([(0.03, 0.03, 0.035), bright])))
        if style in ("gel", "click") or kind == "dimpled":
            extra = rng.uniform(6, 9)
            # The back end piece the clip is moulded into (the clear pens' coloured top).
            parts.append(Part(back0, length + 0.4, round_(lambda t: r * 1.015 * capped_end(1, length + 0.4, 1.0)(t) * front_end(back0, 0.4)(t)),
                              plastic(trim_col, 0.28), sides=72, step=0.15))
            pr = rng.uniform(1.9, 2.5)
            plunger_mat = CHROME if rng.random() < 0.35 else plastic(trim_col if rng.random() < 0.6 else (0.75, 0.75, 0.76), 0.25)
            parts.append(Part(length - 1.0, length + extra, round_(lambda t: pr * capped_end(1, length + extra, 1.0)(t)), plunger_mat, sides=48, step=0.1))
            clip_kind = "steel" if rng.random() < 0.3 else "plastic"
            if clip_kind == "plastic":
                w = rng.uniform(0.45, 0.65) * 2 * r
                clips.append(Clip("plastic", length - 1.5, length - 1.5 - rng.uniform(34, 42), width=w, thick=rng.uniform(1.2, 1.7),
                                  gap=rng.uniform(0.6, 1.0), material=plastic(trim_col, 0.26), extra={"bend": rng.uniform(5, 9)}))
            else:
                clips.append(Clip("steel", length - 3.0, length - 3.0 - rng.uniform(34, 42), width=rng.uniform(2.8, 3.6), thick=rng.uniform(0.5, 0.7),
                                  gap=rng.uniform(0.7, 1.1), material={"kind": "steel", "rough": rng.uniform(0.1, 0.2)},
                                  collar=(length - 7.0, length - 1.0, 0.3), rib=0.1 if rng.random() < 0.5 else 0.0))
        elif kind == "soft":
            clips.append(Clip("steel", length - 4.0, length - 4.0 - rng.uniform(36, 44), width=rng.uniform(3.0, 3.8), thick=rng.uniform(0.6, 0.8),
                              gap=rng.uniform(0.8, 1.1), material=CHROME, collar=(length - 7.6, length - 1.6, 0.3), rib=0.12))
        else:  # rubber-coated
            clips.append(Clip("steel", length - 3.0, length - 3.0 - rng.uniform(34, 40), width=rng.uniform(3.0, 3.6), thick=rng.uniform(0.6, 0.8),
                              gap=rng.uniform(0.8, 1.1), material={"kind": "steel", "rough": 0.15}, collar=(length - 6.5, length - 0.5, 0.35), rib=0.1))

    else:  # capped: a fineliner or felt pen with its cap on the point or posted on the back
        r = rng.uniform(3.6, 4.3)
        hexa = rng.random() < 0.3
        col, rough = colour(0.3)
        fibre = rng.uniform(0.25, 0.35)
        sleeve_l, sleeve_r = rng.uniform(1.8, 3.0), rng.uniform(0.45, 0.6)
        nose_end = sleeve_l + 0.8 + rng.uniform(7, 10)
        nose = lambda t: pchip([sleeve_l + 0.8, sleeve_l + 1.5, nose_end - 1.5, nose_end], [sleeve_r + 0.5, sleeve_r + 0.9, 0.93 * r, r], t)
        length = rng.uniform(132, 142)
        shape = (lambda t, a: np.minimum(P._hexagon(np.asarray(a), r)[None, :], nose(t)[:, None] * 1.16)) if hexa else round_(lambda t: np.minimum(r, nose(t)))
        parts.append(Part(-0.2, 0.9, round_(lambda t: fibre * front_end(-0.2, 0.25)(t)), plastic(ink, 0.6), sides=24, step=0.05))
        parts.append(Part(0.8, sleeve_l + 0.9, round_(lambda t: np.full(len(t), sleeve_r)), {"kind": "steel", "rough": 0.15}, sides=32, step=0.1))
        parts.append(Part(sleeve_l + 0.8, length, (lambda t, a, s=shape: s(t, a) * capped_end(1, length, 1.0)(t)[:, None]),
                          plastic(col, rough), sides=6 if hexa else 72, step=0.2, maps=None if hexa else printed(col, rough), flat=hexa))
        cap_col = list(col) if rng.random() < 0.6 else [0.03, 0.03, 0.035]
        cl = rng.uniform(42, 50)
        rc = r * (1.16 if hexa else 1.0) + 0.5
        posted = rng.random() < 0.4
        if posted:
            c0, c1 = length - 16, length - 16 + cl
            extra = cl - 16
            rcap = lambda t, c1=c1: rc * capped_end(1, c1, 1.8)(t)
            root, tip_end = c1 - 2.0, c1 - 2.0 - rng.uniform(30, 38)
        else:
            c0, c1 = -3.0, cl - 3.0
            cap_front = (c0, c1)
            rcap = lambda t, c0=c0: rc * front_end(c0, 1.8)(t)
            root, tip_end = c0 + 2.0, c0 + 2.0 + rng.uniform(30, 38)
        parts.append(Part(c0, c1, round_(rcap), plastic(cap_col, 0.3), sides=72, step=0.15))
        clip_kind = str(rng.choice(["plastic", "wire", "none"], p=[0.55, 0.3, 0.15]))
        if clip_kind == "plastic":
            clips.append(Clip("plastic", root, tip_end, width=rng.uniform(0.45, 0.6) * 2 * rc, thick=rng.uniform(1.1, 1.5),
                              gap=rng.uniform(0.6, 1.0), material=plastic(cap_col, 0.3), extra={"barrel": rcap}))
        elif clip_kind == "wire":
            clips.append(Clip("wire", root + (1.5 if not posted else -1.5), tip_end, width=rng.uniform(2.4, 3.0), thick=rng.uniform(0.85, 1.0),
                              gap=rng.uniform(0.6, 0.9), material={"kind": "steel", "rough": 0.1},
                              collar=((c0 + 0.5, c0 + 4.5, 0.25) if not posted else (c1 - 4.5, c1 - 0.5, 0.25)), extra={"barrel": rcap}))

    if hexa:
        roll = float(rng.choice([0, 0, 0, 1, -1])) * np.pi / 3
    elif clips:
        roll = float(np.clip(rng.normal(0, 0.6), -1.15, 1.15))  # clip on the upper half, as pens are put down
    else:
        roll = float(rng.uniform(-np.pi, np.pi))

    # Where it lies: along an edge, partly on the page.
    wmm, hmm = sheet.width_mm, sheet.height_mm
    if place is None:
        side = rng.integers(4)
        along = rng.normal(0, 0.2)
        if side < 2:
            angle = along + (0 if rng.random() < 0.5 else np.pi)
            centre = np.array([rng.uniform(0.35, 0.65) * wmm, (hmm + rng.uniform(-25, 10)) if side else rng.uniform(-10, 25)])
        else:
            angle = np.pi / 2 + along + (0 if rng.random() < 0.5 else np.pi)
            centre = np.array([(wmm + rng.uniform(-25, 10)) if side == 3 else rng.uniform(-10, 25), rng.uniform(0.35, 0.65) * hmm])
    else:
        centre, angle = np.asarray(place[0], np.float64), float(place[1])
    t0 = min(-3.5, cap_front[0] - 0.5) if cap_front else -0.5
    t1 = length + extra + 0.5
    tg = np.linspace(t0, t1, 360)
    rmax = np.zeros(len(tg))
    a8 = 2 * np.pi * (np.arange(12) + 0.5) / 12
    for part in parts:
        k = (tg >= part.t0) & (tg <= part.t1)
        if k.any():
            rmax[k] = np.maximum(rmax[k], np.asarray(part.radius(tg[k], a8)).max(1))
    rmax = np.maximum(rmax, 0.2)
    for clip in clips:  # a collar stands proud of the barrel: the pen rests on it too
        if clip.collar is not None:
            c0, c1, proud = clip.collar
            k = (tg >= c0 + 0.5) & (tg <= c1 - 0.5)
            rmax[k] += proud
    mid = 0.5 * (t0 + t1)
    axis2 = np.array([np.cos(angle), np.sin(angle)])
    x, y = centre[0] + (tg - mid) * axis2[0], centre[1] + (tg - mid) * axis2[1]
    z = P._rest(sheet, ground, x, y, rmax)
    axis = Axis(x, y, z, tg, roll)
    things = [part.mesh(axis) for part in parts]
    for clip in clips:
        under = clip.extra.get("barrel")
        if under is None:
            def under(t, parts=parts):
                out = np.zeros(len(t))
                for part in parts:
                    if part.material.get("kind") in ("clear",) and part.radius(np.array([0.0]), a8).max() < 2:
                        continue
                    k = (t >= part.t0) & (t <= part.t1)
                    if k.any():
                        out[k] = np.maximum(out[k], np.asarray(part.radius(t[k], np.array([np.pi / 2]))).max(1))
                return out
        things += clip_meshes(axis, clip, under)
    if rng.random() < 0.35 and style not in ("capped",) and cap_front is None:
        # Ink dried round the point: a dark gloss on the socket just behind the ball.
        e = rng.uniform(0.6, 1.2)
        things.append(Part(0.3, e, round_(lambda t: pchip(tk, rk, t) + 0.03), plastic(ink, 0.15), sides=32, step=0.05).mesh(axis))
    return things


# Pencils ---------------------------------------------------------------------------

PAINTS = [(0.85, 0.62, 0.08), (0.85, 0.62, 0.08), (0.88, 0.7, 0.1), (0.32, 0.04, 0.06), (0.06, 0.22, 0.12), (0.03, 0.03, 0.035),
          (0.03, 0.03, 0.035), (0.55, 0.06, 0.05), (0.08, 0.16, 0.4), (0.04, 0.3, 0.32), None, None]  # None: bare varnished wood
ERASERS = [(0.86, 0.5, 0.5), (0.86, 0.5, 0.5), (0.88, 0.85, 0.76), (0.82, 0.74, 0.55), (0.78, 0.33, 0.28), (0.25, 0.55, 0.4)]
GRAPHITE = {"kind": "graphite"}


def lay(sheet, rng, place, length):
    """Where a pencil lies: as given, or along an edge, partly on the page. (centre, angle)."""
    if place is not None:
        return np.asarray(place[0], np.float64), float(place[1])
    wmm, hmm = sheet.width_mm, sheet.height_mm
    side = rng.integers(4)
    along = rng.normal(0, 0.25)
    if side < 2:
        angle = along + (0 if rng.random() < 0.5 else np.pi)
        centre = np.array([rng.uniform(0.3, 0.7) * wmm, (hmm + rng.uniform(-30, 10)) if side else rng.uniform(-10, 30)])
    else:
        angle = np.pi / 2 + along + (0 if rng.random() < 0.5 else np.pi)
        centre = np.array([(wmm + rng.uniform(-30, 10)) if side == 3 else rng.uniform(-10, 30), rng.uniform(0.3, 0.7) * hmm])
    return centre, angle


def rest_axis(sheet, ground, parts, t0, t1, centre, angle, roll=0.0):
    """Lay a rigid rod made of `parts` (from t0 to t1) at centre/angle: its resting Axis."""
    tg = np.linspace(t0, t1, 360)
    a12 = 2 * np.pi * (np.arange(12) + 0.5) / 12
    rmax = np.full(len(tg), 0.2)
    for part in parts:
        k = (tg >= part.t0) & (tg <= part.t1)
        if k.any():
            rmax[k] = np.maximum(rmax[k], np.asarray(part.radius(tg[k], a12)).max(1))
    mid = 0.5 * (t0 + t1)
    d = np.array([np.cos(angle), np.sin(angle)])
    x, y = centre[0] + (tg - mid) * d[0], centre[1] + (tg - mid) * d[1]
    z = P._rest(sheet, ground, x, y, rmax)
    return Axis(x, y, z, tg, roll)


def _pencil_maps(rng, cone, r_in, t0, paint, words, face, worn, foil_metal):
    """The textures of a sharpened pencil, wood and paint as one surface: wood wherever the
    sharpener cut into the hexagon (its scalloped edge a little chipped), lacquer elsewhere,
    a line of stamped text on one face. Rows run round the pencil (row / rows is the angle
    over 2 pi, as Part lays them), columns along it from t0."""
    from PIL import Image, ImageDraw

    seed = int(rng.integers(2**31))
    cedar = np.array([0.8, 0.6, 0.4]) * rng.uniform(0.88, 1.05)

    def make(span, around):
        r = np.random.default_rng(seed)
        ppmm = 8.0
        nu, nv = max(64, int(span * ppmm)), 6 * 32
        t = t0 + (np.arange(nu) + 0.5) / ppmm
        a = 2 * np.pi * (np.arange(nv) + 0.5) / nv
        hexr = P._hexagon(a, r_in)
        chip = 0.03 * np.clip(band((nv, nu), r, 1.5, 6), -1, 3)
        cut = ((cone(t)[None, :] < hexr[:, None] - 0.005 + chip) & (cone(t)[None, :] < 1.0 * hexr.max())).astype(np.float32)
        cut = np.clip(blur(cut, 0.6), 0, 1)
        # Cedar: fine straight growth lines along the pencil, irregular, a little fibrous.
        lines = np.clip(np.sin(np.linspace(0, r.uniform(28, 44) * np.pi, nv)[:, None] + 2.2 * band((nv, nu), r, 30, 300, stretch=8.0)), 0, 1) ** 8
        wood = cedar[None, None, :] * (1 + 0.06 * band((nv, nu), r, 2, 30, stretch=6.0) - 0.16 * lines + 0.05 * band((nv, nu), r, 1, 4))[..., None]
        wood *= (1 + 0.04 * band((1, nu), r, 40, 400))[0][None, :, None]  # a little variation along it
        if paint is None:
            coat = wood * np.array([0.92, 0.86, 0.78])  # under clear varnish: a touch darker and warmer
            coat_rough = 0.4
        else:
            coat = np.asarray(paint, np.float32)[None, None, :] * (1 + 0.02 * band((nv, nu), r, 4, 60, stretch=3.0))[..., None]
            coat_rough = r.uniform(0.38, 0.5)
            if worn:
                # Paint thinned on the corners of the hexagon, where fingers rub it.
                corner = np.abs(((a - np.pi / 2 + np.pi / 6) % (np.pi / 3)) - np.pi / 6)
                rub = np.exp(-((np.pi / 6 - corner) / 0.035) ** 2)[:, None] * np.clip(band((nv, nu), r, 20, 200, stretch=3.0) - 0.8, 0, 1)
                coat = coat * (1 - 0.3 * rub[..., None]) + wood * 0.3 * rub[..., None]
        albedo = wood * cut[..., None] + coat * (1 - cut[..., None])
        rough = (0.85 + 0.04 * band((nv, nu), r, 2, 20)) * cut + (coat_rough + 0.05 * band((nv, nu), r, 1, 6)) * (1 - cut)
        out = {"albedo": albedo.clip(0, 1).astype(np.float32), "rough_map": rough.clip(0.05, 0.95).astype(np.float32)}
        if words:
            px = int(2.4 * ppmm)
            img = Image.new("L", (nu, px + 6), 0)
            d = ImageDraw.Draw(img)
            x0 = int(r.uniform(0.35, 0.55) * nu)
            d.text((x0, 2), words, fill=255, font=_font(px))
            m = np.asarray(img, np.float32) / 255
            m[:, int(0.95 * nu):] = 0
            m *= np.clip(0.85 + 0.3 * band(m.shape, r, 2, 12), 0, 1)
            row0 = int((0.25 + face / 6) * nv - m.shape[0] / 2)
            rows = (row0 + np.arange(m.shape[0])) % nv
            mask = np.zeros((nv, nu), np.float32)
            mask[rows] = m
            mask *= 1 - cut
            if foil_metal:
                out["foil"], out["metal"] = mask, foil_metal
            else:
                light = paint is None or float(np.mean(paint)) > 0.3
                ink = np.array([0.04, 0.04, 0.045]) if light else np.array([0.85, 0.85, 0.82])
                out["albedo"] = (out["albedo"] * (1 - mask[..., None]) + ink * mask[..., None]).astype(np.float32)
        return out

    return make


def pencil(sheet, rng, ground, place=None):
    """A wooden pencil: hexagonal, sharpened (to a fine point, or worn blunt), painted or bare,
    with a crimped ferrule and an eraser worn at a slant, or without; new or worn to a stub."""
    length = float(rng.choice([rng.uniform(150, 185), rng.uniform(95, 150), rng.uniform(60, 95)], p=[0.45, 0.4, 0.15]))
    r_in = rng.uniform(3.4, 3.6)
    corner = r_in / np.cos(np.pi / 6)
    eraser = rng.random() < 0.55
    sharp = rng.random() < 0.7
    r_tip = rng.uniform(0.08, 0.16) if sharp else rng.uniform(0.3, 0.55)
    cone_len = rng.uniform(15, 22) if sharp else rng.uniform(12, 18)
    gam = rng.uniform(0.95, 1.1)
    r_lead = rng.uniform(0.95, 1.08)
    worn = rng.random() < 0.45
    roll = float(rng.choice([0, 0, 1, -1, 2])) * np.pi / 3

    def cone(t):
        t = np.asarray(t, np.float64)
        c = r_tip + (corner * 1.02 - r_tip) * np.clip(t / cone_len, 0, 1) ** gam
        return np.where(t < 0, np.sqrt(np.clip(r_tip**2 - t**2, 1e-5, None)), c)

    # Where the cone reaches the edge of the lead: graphite in front, wood behind.
    tt = np.linspace(0, cone_len, 2000)
    t_lead = float(tt[np.argmax(cone(tt) >= r_lead)])
    sp = rng.uniform(0, 2 * np.pi)

    def facets(t, a):  # the faint spiral the sharpener blade leaves on the cone
        t = np.asarray(t, np.float64)
        on = (smoothstep(t_lead, t_lead + 2, t) * (t < cone_len))[:, None]
        return 1 + 0.012 * np.cos(6 * np.asarray(a)[None, :] + sp + 2 * np.pi * t[:, None] / 7.0) * on

    paint = PAINTS[int(rng.integers(len(PAINTS)))]
    if paint is None:
        words = str(rng.choice(["HB", "2B", "B", "No. 2", ""]))
    else:
        grade = str(rng.choice(["HB", "2B", "B", "F", "H", "2H", "4B"], p=[0.4, 0.15, 0.15, 0.08, 0.08, 0.07, 0.07]))
        words = str(rng.choice([grade, f"{str(rng.choice(NAMES)).upper()}   {grade}", f"{rng.integers(1000, 9999)}   {grade}", f"DRAWING  {grade}"]))
        if rng.random() < 0.15:
            words = ""
    foil = (str(rng.choice(["Au", "Ag"])) if rng.random() < 0.7 else None) if paint is not None else None
    parts = [Part(-r_tip, t_lead + 0.05, round_(cone), GRAPHITE, sides=48, step=0.04)]
    fer0, fer1 = (length - rng.uniform(9, 11), length - rng.uniform(2.5, 6.5)) if eraser else (length, length)
    body_end = fer0 + 0.6 if eraser else length

    def shape(t, a):
        out = np.minimum(P._hexagon(np.asarray(a), r_in)[None, :], cone(t)[:, None]) * facets(t, a)
        if not eraser:
            out = out * capped_end(1, length, 0.4)(t)[:, None]
        return out

    maps = _pencil_maps(rng, cone, r_in, t_lead, paint, words, int(rng.choice([0, 0, 1, -1])), worn, foil)
    parts.append(Part(t_lead, body_end, shape, {"kind": "pplastic", "colour": [0.7, 0.55, 0.35], "rough": 0.5}, sides=6 * 32, step=0.12, maps=maps))
    things = []
    if eraser:
        # The ferrule: crimped round the wood and the eraser, grooved bands, crimp dents.
        rf = r_in * rng.uniform(1.06, 1.1)
        n_g = int(rng.integers(3, 7))
        g_at = fer0 + (fer1 - fer0) * rng.uniform(0.3, 0.45)
        dents = rng.random() < 0.7
        d_at = fer0 + (fer1 - fer0) * 0.22

        def ferrule(t, a):
            t = np.asarray(t, np.float64)
            g = np.zeros_like(t)
            for k in range(n_g):
                g += np.exp(-((t - (g_at + 0.55 * k)) / 0.13) ** 2)
            base = rf - 0.12 * g - 0.08 * np.exp(-((t - fer1 + 0.8) / 0.25) ** 2)
            out = (base * np.minimum(front_end(fer0, 0.25)(t), capped_end(1, fer1, 0.25)(t)))[:, None] * np.ones((1, len(a)))
            if dents:
                out -= 0.18 * np.exp(-((t[:, None] - d_at) / 0.6) ** 2) * np.clip(np.cos(4 * (np.asarray(a)[None, :] - 0.3)), 0, None) ** 6
            return out

        metal = {"kind": "steel", "rough": float(rng.uniform(0.18, 0.3))} if rng.random() < 0.7 else {"kind": "brass", "rough": 0.2}
        parts.append(Part(fer0, fer1, ferrule, metal, sides=96, step=0.05))
        # The eraser: out of the ferrule, worn to a slanted face with a rounded rim.
        re_ = r_in * 0.98
        out_len = rng.uniform(2.0, 6.5) if worn else rng.uniform(5.0, 7.0)
        slant = np.radians(rng.uniform(12, 35)) if rng.random() < 0.85 else 0.0
        a0 = rng.uniform(0, 2 * np.pi)
        m = 64
        ang = 2 * np.pi * (np.arange(m) + 0.5) / m
        t_end = np.maximum(fer1 - 0.5 + out_len - np.tan(slant) * re_ * np.cos(ang - a0), fer1 + 0.3)
        rho = min(1.0, 0.25 * out_len)
        s = np.linspace(0, 1, 24)
        T = (fer1 - 0.5) + s[:, None] * (t_end[None, :] - rho - (fer1 - 0.5))
        Rr = np.full_like(T, re_)
        q = np.linspace(0, np.pi / 2, 10)[1:]
        T = np.vstack([T, t_end[None, :] - rho * (1 - np.sin(q))[:, None]])
        Rr = np.vstack([Rr, re_ - rho * (1 - np.cos(q))[:, None] * np.ones((1, m))])
    centre, angle = lay(sheet, rng, place, length)
    support = parts + ([Part(fer1, fer1 + 6, round_(lambda t: np.full(len(t), r_in)), {})] if eraser else [])
    axis = rest_axis(sheet, ground, support, -r_tip, (length + 2) if eraser else length, centre, angle, roll)
    things += [p.mesh(axis) for p in parts]
    if eraser:
        pts = np.stack([axis.point(T[:, j], ang[j:j + 1], Rr[:, j:j + 1])[:, 0] for j in range(m)], 1)
        v, f, uv = rings(pts)
        er = ERASERS[int(rng.integers(len(ERASERS)))]
        nu = 64
        col = (np.arange(nu) + 0.5) / nu
        r_ = np.random.default_rng(int(rng.integers(2**31)))
        tex = np.asarray(er, np.float32)[None, None, :] * (1 + 0.05 * band((32, nu), r_, 1, 6))[..., None]
        grey = (0.35 if worn else 0.6) * np.asarray(er, np.float32) + 0.04  # rubbed graphite on the worn face
        dirt = smoothstep(0.7, 1.0, col)
        tex = tex * (1 - dirt[None, :, None]) + grey * dirt[None, :, None]
        tex *= (1 - 0.1 * (1 - smoothstep(0.0, 0.12, col)))[None, :, None]  # grimy where it leaves the ferrule
        things.append({"verts": v, "faces": f, "uv": uv, "material": {"kind": "pplastic", "colour": list(er), "rough": 0.75,
                       "albedo": tex.clip(0, 1).astype(np.float32), "rough_map": np.full((32, nu), 0.8, np.float32)}})
    return things

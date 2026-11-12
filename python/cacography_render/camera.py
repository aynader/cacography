"""Photograph a sheet: lay it on a desk, light the room, point a camera at it.

The scene is modelled in millimetres. The sheet lies on the desk at z = 0 with its
corner at the origin, x to the right and y down the page, z up off the desk. One
point light (a lamp, a ceiling light) and an even ambient light (the rest of the room)
light it; the photographer's phone and hand may shade part of it. Paper, ink, graphite
and a varnished desk each reflect some of the lamp back as glare. The camera is a
pinhole with a tilt, a roll and a distance, then the things a phone does to a picture:
lens blur, vignetting, colour fringes, sensor noise, white balance, a tone curve,
sharpening and JPEG.

A document is shot in one room (`Studio`): the desk, the lights and the phone stay the
same from page to page, while how each page lies and how it was framed changes.
"""
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from . import surfaces
from .imaging import band, blur, rgb, sample, smoothstep, to_linear, to_srgb

CAMERAS = ("phone", "scan", "none")
DESKS = {
    # light and dark wood (or the surface's two tones), grain wavelength in mm, varnish gloss
    "oak": ((198, 166, 124), (140, 104, 68), 9.0, 1.0),
    "walnut": ((120, 86, 62), (62, 42, 30), 7.0, 1.0),
    "birch": ((226, 203, 170), (186, 156, 118), 12.0, 1.0),
    "laminate": ((226, 223, 216), (210, 206, 199), 0, 1.6),
    "felt": ((92, 96, 101), (70, 73, 78), 0, 0.0),
    "white": ((238, 238, 234), (229, 229, 225), 0, 0.8),
    "black": ((34, 34, 36), (26, 26, 28), 0, 1.4),
    "glass": ((20, 22, 24), (15, 16, 17), 0, 4.0),  # dark glass over a desk: mostly reflection
}


@dataclass
class Studio:
    desk: str
    desk_angle: float
    light: np.ndarray  # lamp position relative to the page centre, mm
    key: float  # share of the light that comes from the lamp
    key_tint: np.ndarray
    sheen: float  # how much of the lamp the paper reflects as glare
    sheen_power: float  # how tight that glare is: low is a broad haze
    desk_gloss: float
    ambient_tint: np.ndarray
    ambient_slope: np.ndarray
    room: str  # the HDRI of the room around the desk (surfaces.ROOMS), when tracing in 3D
    white: np.ndarray  # what the camera's white balance leaves behind
    brightness: float  # sRGB level the camera exposes the paper to
    grain: float  # sensor noise, from the light level
    vignette: float
    fringe: float
    sharpen: float
    quality: int
    kind: str = "lamp"  # which of LIGHTS


LIGHTS = {
    # where the main light sits relative to the page (mm), and its share of all the light
    "lamp": lambda r: (r.uniform(150, 450), r.uniform(450, 800), r.uniform(0.6, 0.8)),  # a desk lamp nearby
    "overhead": lambda r: (r.uniform(0, 200), r.uniform(1200, 1800), r.uniform(0.45, 0.6)),  # a ceiling light
    "window": lambda r: (r.uniform(800, 1500), r.uniform(500, 900), r.uniform(0.45, 0.6)),  # daylight from one side
    "soft": lambda r: (r.uniform(0, 600), r.uniform(1500, 2500), r.uniform(0.15, 0.3)),  # an evenly lit room
    # Low sun or a lamp set down beside the page: every crease and groove shows.
    # Kept to about two thirds of the light: a room always fills the shade in some.
    "raking": lambda r: (r.uniform(700, 1000), r.uniform(180, 320), r.uniform(0.62, 0.74)),
    # The phone's own flash, beside the lens (placed once the shot is framed: with_flash).
    "flash": lambda r: (0.0, 300.0, r.uniform(0.82, 0.92)),
    # A room lit by one weak lamp in the evening: the phone pushes its exposure.
    "dim": lambda r: (r.uniform(400, 1000), r.uniform(500, 1200), r.uniform(0.4, 0.6)),
}


def studio(aspects):
    """The room a document is shot in: the desk, the lights and the phone."""
    bg, lit, cam = aspects.rng("background"), aspects.rng("light"), aspects.rng("camera")
    desk = surfaces.choose(aspects.value("background"), bg)
    kind = aspects.value("light")
    if kind == "random":
        kind = str(lit.choice(list(LIGHTS), p=[0.33, 0.2, 0.18, 0.12, 0.04, 0.05, 0.08]))
    lamp_angle = lit.uniform(0, 2 * np.pi)
    reach, height, key = LIGHTS[kind](lit)
    warm = lit.uniform(-1, 1)  # lamp warmer than the room, or cooler
    toward = np.array([np.cos(lamp_angle), np.sin(lamp_angle)])
    # The room is brighter towards a window; elsewhere it just varies a little.
    slope = toward * lit.uniform(0.15, 0.35) if kind == "window" else lit.normal(0, 0.1, 2)
    if kind == "dim":
        warm = lit.uniform(1.0, 2.0)  # tungsten, which the phone balances only partly
    st = Studio(
        desk=desk,
        desk_angle=bg.uniform(0, np.pi),
        light=np.array([reach * toward[0], reach * toward[1], height]),
        key=key,
        key_tint=np.array([1 + 0.05 * warm, 1.0, 1 - 0.07 * warm], np.float32),
        sheen=lit.uniform(0.02, 0.07),
        sheen_power=lit.uniform(5, 18),
        desk_gloss=bg.uniform(0.05, 0.3),
        ambient_tint=np.array([1 - 0.03 * warm, 1.0, 1 + 0.04 * warm], np.float32),
        ambient_slope=slope,
        room=surfaces.pick_room(kind, lit),
        white=np.array([1 + cam.normal(0.018, 0.018), 1.0, 1 + cam.normal(-0.035, 0.025)], np.float32),  # auto white balance leaves it a little warm
        brightness=float(np.clip(cam.normal(214, 10), 188, 236)),
        grain=cam.uniform(0.5, 1.5),
        vignette=cam.uniform(0.06, 0.22),
        fringe=cam.uniform(0.0003, 0.0009),
        sharpen=cam.uniform(0.35, 0.75),
        quality=int(cam.integers(80, 91)),
        kind=kind,
    )
    if kind == "flash":
        # A small bright source right beside the lens: a hot spot of glare, flat light,
        # the desk dropping into the dark around the sheet.
        st = replace(st, sheen=lit.uniform(0.05, 0.12), sheen_power=lit.uniform(25, 70), key_tint=np.array([0.98, 1.0, 1.04], np.float32),
                     vignette=cam.uniform(0.25, 0.4),
                     white=st.white * np.array([0.99, 1.0, 1.02], np.float32))
    elif kind == "dim":
        # Little light: the phone exposes the paper darker, its noise rises, and its
        # denoiser smears the fine detail away.
        st = replace(st, brightness=float(np.clip(cam.normal(170, 12), 140, 195)), grain=cam.uniform(2.2, 3.6),
                     sharpen=cam.uniform(0.1, 0.3), quality=int(cam.integers(72, 85)),
                     white=st.white * np.array([1 + cam.uniform(0.03, 0.08), 1.0, 1 - cam.uniform(0.06, 0.12)], np.float32))
    return st


def with_flash(st, shot, wmm, hmm):
    """With light=flash, the flash where the phone is, about a centimetre from the lens."""
    if st.kind != "flash":
        return st
    beside = shot.centre + 10.0 * shot.right - 4.0 * shot.down
    return replace(st, light=beside - np.array([wmm / 2, hmm / 2, 0.0]))


@dataclass
class Shot:
    """Where the camera is and how it frames the page."""
    centre: np.ndarray
    right: np.ndarray
    down: np.ndarray
    forward: np.ndarray
    scale: float  # output pixels per unit of the image plane
    offset: np.ndarray

    def project(self, X):
        p = X - self.centre
        z = p @ self.forward
        return np.stack([p @ self.right / z, p @ self.down / z], -1) * self.scale + self.offset

    def rays(self, u, v):
        a = (u - self.offset[0]) / self.scale
        b = (v - self.offset[1]) / self.scale
        return [self.right[i] * a + self.down[i] * b + self.forward[i] for i in range(3)]


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


TILTS = {"flat": (0, 2), "slight": (3, 7), "steep": (15, 32)}  # degrees off straight down
# Where the cut-off margins sit: one edge, or two meeting at a corner (left, right, top, bottom).
BESIDE = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


def _frame(sheet, out_w, out_h, rng, kind, angle="random", framing="tight", frng=None):
    """Pick the camera's pose, then zoom so the page fills the frame with a little desk round it
    (`framing` tight), more of the desk (loose), or part of the sheet cut off (cropped).

    `frng` is the framing aspect's own random numbers; without it the shot is framed tight.
    """
    wmm, hmm = sheet.width_mm, sheet.height_mm
    if kind == "scan":
        tilt, roll, distance = 0.0, rng.normal(0, np.radians(0.4)), 5000.0
        target = np.array([wmm / 2 + rng.normal(0, 0.8), hmm / 2 + rng.normal(0, 0.8), 0.0])
        margins = rng.uniform(-0.004, 0.004, 4)
    else:
        if angle == "random":
            tilt = np.radians(np.clip(2 + abs(rng.normal(0, 5)), 0, 14))
            if frng is not None and frng.random() < 0.2:
                tilt = np.radians(frng.uniform(14, 28))  # now and then a hurried shot from the side
        elif angle in TILTS:
            tilt = np.radians(rng.uniform(*TILTS[angle]))
        else:
            tilt = np.radians(float(angle))
        roll = rng.normal(0, np.radians(1.2))
        distance = rng.uniform(230, 340)
        target = np.array([wmm / 2 + rng.normal(0, 6), hmm / 2 + rng.normal(0, 8), 0.0])
        margins = rng.uniform(0.012, 0.06, 4)
        if rng.random() < 0.25:
            margins[rng.integers(4)] = rng.uniform(-0.03, 0.0)  # one edge cut off
        if frng is not None:
            if framing == "random":
                framing = str(frng.choice(["tight", "loose", "cropped"], p=[0.45, 0.3, 0.25]))
            if framing == "loose":
                # Held further back: the sheet fills two-thirds of the frame or less.
                margins = frng.uniform(0.06, 0.22, 4)
            elif framing == "cropped":
                # Held too close: an edge, or a corner, falls outside the frame.
                margins = frng.uniform(0.02, 0.07, 4)
                k = int(frng.integers(4))
                margins[k] = frng.uniform(-0.18, -0.05)
                if frng.random() < 0.5:
                    margins[frng.choice(BESIDE[k])] = frng.uniform(-0.12, -0.03)
    # Most people stand at the bottom of the page and lean over it.
    side = rng.normal(0, np.radians(25))
    turn = _rotation([np.cos(side), np.sin(side), 0], -tilt) @ _rotation([0, 0, 1], roll)
    right, down, forward = turn @ [1.0, 0, 0], turn @ [0, 1.0, 0], turn @ [0, 0, -1.0]
    centre = target - forward * distance
    shot = Shot(centre, right, down, forward, 1.0, np.zeros(2))
    t = np.linspace(0, 1, 40)
    edge = np.concatenate([np.stack([t * wmm, 0 * t], 1), np.stack([wmm + 0 * t, t * hmm], 1),
                           np.stack([t * wmm, hmm + 0 * t], 1), np.stack([0 * t, t * hmm], 1)])
    z = sheet.lift(edge[:, 0], edge[:, 1])
    uv = shot.project(np.column_stack([edge, z]))
    lo, hi = uv.min(0), uv.max(0)
    left, right_m, top, bottom = margins
    scale = min(out_w * (1 - left - right_m) / (hi[0] - lo[0]), out_h * (1 - top - bottom) / (hi[1] - lo[1]))
    room = np.array([out_w, out_h]) - (hi - lo) * scale
    share = np.array([left / max(left + right_m, 1e-6), top / max(top + bottom, 1e-6)])
    shot.scale = scale
    shot.offset = -lo * scale + room * np.clip(share, -0.2, 1.2)
    return shot


def _hand(rng):
    """The hand holding the phone, as ellipses in the phone's own frame (its half width and
    half length are 1, +y towards the photographer): (x, y, rx, ry, turn, z mm, rz mm).
    The palm and wrist below the phone's near end, three fingers wrapped round one long
    edge, the thumb along the other, the forearm going off towards the photographer."""
    s = 1.0 if rng.random() < 0.5 else -1.0  # which side the fingers are on
    up = rng.uniform(-0.15, 0.25)  # how far up the phone the hand holds it
    parts = [(rng.normal(0, 0.1), 0.95 + up, rng.uniform(1.0, 1.2), rng.uniform(0.5, 0.65), 0.0, -12.0, 14.0)]  # palm
    for k in range(3):
        parts.append((s * rng.uniform(1.0, 1.1), 0.05 + up + 0.24 * k + rng.normal(0, 0.03), rng.uniform(0.24, 0.32), rng.uniform(0.1, 0.13),
                      0.0, -4.0, 8.0))
    parts.append((-s * rng.uniform(0.9, 1.0), 0.4 + up, rng.uniform(0.26, 0.34), rng.uniform(0.32, 0.42), s * 0.3, 2.0, 8.0))  # thumb
    parts.append((s * rng.uniform(0.05, 0.3), 2.1 + up, rng.uniform(0.75, 0.9), rng.uniform(1.3, 1.6), s * rng.uniform(0.05, 0.25), 25.0, 30.0))
    return parts


def _occluder(rng, wmm, hmm, choice="random"):
    """The phone and the hand holding it, between the lamp and the page."""
    if choice == "none" or (rng.random() > 0.35 and choice == "random"):
        return None
    occ = dict(cx=rng.uniform(0.15, 0.85) * wmm, cy=hmm + rng.uniform(-110, 30), half=(rng.uniform(40, 90), rng.uniform(70, 140)),
               angle=rng.normal(0, 0.4), radius=rng.uniform(10, 35), soft=rng.uniform(15, 45), strength=rng.uniform(0.5, 0.9))
    occ["half"] = (occ["half"][0], max(occ["half"][1], 1.6 * occ["half"][0]))  # a phone is about twice as long as wide
    occ["hand"] = _hand(rng)
    return occ


def _shade_of(occ, x, y):
    if occ is None:
        return 0.0
    c, s = np.cos(occ["angle"]), np.sin(occ["angle"])
    px, py = x - occ["cx"], y - occ["cy"]
    qx = np.abs(px * c + py * s) - (occ["half"][0] - occ["radius"])
    qy = np.abs(-px * s + py * c) - (occ["half"][1] - occ["radius"])
    sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - occ["radius"]
    hx, hy = occ["half"]
    u, v = px * c + py * s, -px * s + py * c
    for ex, ey, rx, ry, _, _, _ in occ.get("hand", []):
        # The hand's shadow: each part an ellipse (its distance roughly, which the blur hides).
        rx, ry = rx * hx, ry * hy
        d = (np.hypot((u - ex * hx) / rx, (v - ey * hy) / ry) - 1) * min(rx, ry)
        sd = np.minimum(sd, d)
    return occ["strength"] * smoothstep(occ["soft"], -occ["soft"], sd)


def _light(st, X, Y, Z, n, wmm, hmm, occ, shadow=0.0, ambient_block=0.0):
    """Irradiance (RGB, about 1 for flat paper at the page centre) at points with normals `n`."""
    lamp = st.light + [wmm / 2, hmm / 2, 0.0]
    vx, vy, vz = lamp[0] - X, lamp[1] - Y, lamp[2] - Z
    r2 = vx * vx + vy * vy + vz * vz
    cos = (n[0] * vx + n[1] * vy + n[2] * vz) / np.sqrt(r2)
    flat = lamp[2] / np.linalg.norm(st.light) ** 3  # the same at the page centre, flat
    shade = _shade_of(occ, X, Y)
    direct = np.maximum(cos, 0) / r2 / flat * (1 - shade) * (1 - shadow)
    ambient = (1 + st.ambient_slope[0] * (X / wmm - 0.5) + st.ambient_slope[1] * (Y / hmm - 0.5)) * (1 - 0.25 * shade)
    ambient = ambient * (1 - ambient_block) * (0.85 + 0.15 * n[2])
    return (st.key * direct)[..., None] * st.key_tint + ((1 - st.key) * ambient)[..., None] * st.ambient_tint


def _glare(st, eye, X, Y, Z, n, wmm, hmm, occ):
    """How well the lamp's reflection lines up with the camera (n.h), and the lamp's strength.

    Raise n.h to a power for a surface's gloss: low for paper's broad haze, high for
    varnish or the oily shine of ink and graphite.
    """
    lamp = st.light + [wmm / 2, hmm / 2, 0.0]
    lx, ly, lz = lamp[0] - X, lamp[1] - Y, lamp[2] - Z
    r2 = lx * lx + ly * ly + lz * lz
    r = np.sqrt(r2)
    vx, vy, vz = eye[0] - X, eye[1] - Y, eye[2] - Z
    v = np.sqrt(vx * vx + vy * vy + vz * vz)
    hx, hy, hz = lx / r + vx / v, ly / r + vy / v, lz / r + vz / v
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    ndh = np.clip((n[0] * hx + n[1] * hy + n[2] * hz) / hn, 0, 1)
    flat = lamp[2] / np.linalg.norm(st.light) ** 3
    strength = st.key * np.maximum((n[0] * lx + n[1] * ly + n[2] * lz) / r, 0) / r2 / flat * (1 - _shade_of(occ, X, Y))
    return ndh, strength


def _normals(height, ppmm):
    gy, gx = np.gradient(height * ppmm)
    norm = np.sqrt(gx * gx + gy * gy + 1)
    return -gx / norm, -gy / norm, 1 / norm


def _desk(st, kind, x0, y0, x1, y1, res, rng, surface=None):
    """The desk's colour over [x0, x1] x [y0, y1] mm, `res` pixels per mm."""
    h, w = max(2, int((y1 - y0) * res)), max(2, int((x1 - x0) * res))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) / res
    xs += x0
    ys += y0
    if surface is not None and surface.textured and kind != "scan":
        return surface.colour_at(xs, ys).astype(np.float32), xs, ys
    if surface is not None and kind != "scan":
        kind = surface.procedural
    if kind == "scan":
        lid = rgb((232, 232, 228)) * (1 + 0.004 * band((h, w), rng, 2 * res, 40 * res))[..., None]
        return lid.astype(np.float32), xs, ys
    light, dark, period, _ = DESKS[kind]
    light, dark = rgb(light), rgb(dark)
    if period:
        a = st.desk_angle
        across = -xs * np.sin(a) + ys * np.cos(a)
        # Growth rings cut lengthwise: long stripes that drift slowly.
        warp = 1.2 * band((h, w), rng, 25 * res, 250 * res, stretch=14, angle=a) + 0.08 * band((h, w), rng, 6 * res, 40 * res, stretch=12, angle=a)
        year = across / period + warp
        f = year - np.floor(year)
        # Each year's wood is pale where it grew in spring and darkens into the late wood,
        # which ends sharply where the next spring begins.
        late = smoothstep(0.5, 0.93, f) * (1 - smoothstep(0.93, 1.0, f))
        fibres = band((h, w), rng, 0.15 * res, 1.5 * res, stretch=25, angle=a)
        pores = smoothstep(1.3, 2.4, band((h, w), rng, 0.25 * res, 1.2 * res, stretch=10, angle=a))
        plank = np.floor(across / rng.uniform(90, 160))
        tone = 0.06 * np.sin(plank * 12.9898 + 4.1) + 0.04 * band((h, w), rng, 20 * res, 200 * res)
        mix = np.clip(0.22 + 0.42 * late + 0.06 * fibres + 0.25 * pores * (0.4 + late) + tone, 0, 1)
        colour = light * (1 - mix[..., None]) + dark * mix[..., None]
    else:
        mix = np.clip(0.5 + 0.25 * band((h, w), rng, 0.3 * res, 3 * res) + 0.15 * band((h, w), rng, 10 * res, 100 * res), 0, 1)
        colour = light * (1 - mix[..., None]) + dark * mix[..., None]
    return colour.astype(np.float32), xs, ys


def photograph(sheet, st, kind, aspects, page):
    """The finished photo (or scan) of `sheet`, as an RGB image the size of the page."""
    rng = aspects.rng("camera", page)
    h, w = sheet.mask.shape
    wmm, hmm, ppmm = sheet.width_mm, sheet.height_mm, sheet.ppmm
    if kind == "scan":
        sheet = _flattened(sheet)
    surface = surfaces.prepare(st, aspects, page, wmm, hmm) if kind == "phone" else None
    if surface is not None:
        sheet = surface.drape(sheet)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    X, Y = xs / ppmm, ys / ppmm
    relief = sheet.lift(X, Y) * (0.05 if kind == "scan" else 1.0) + sheet.relief
    n = _normals(relief, ppmm)

    if kind == "none":
        light = (0.8 + 0.2 * (n[2] * 0.95 - n[1] * 0.31) / 0.95)[..., None]
        radiance = sheet.albedo * light
        radiance = radiance * sheet.mask[..., None] + (1 - sheet.mask[..., None])
        plain = replace(st, white=np.ones(3, np.float32), brightness=246.0, vignette=0.0, fringe=0.0, sharpen=0.0)
        return _develop(radiance, sheet.mask, plain, kind, rng, w, h, None, np.ones(3))

    occ = _occluder(aspects.rng("shadow", page), wmm, hmm, aspects.value("shadow")) if kind == "phone" else None
    if kind == "scan":
        st = _scanner(st)
    shot = _frame(sheet, w, h, aspects.rng("angle", page), kind, aspects.value("angle"),
                  aspects.value("framing"), aspects.rng("framing", page))
    if kind == "phone" and st.kind == "flash":
        st, occ = with_flash(st, shot, wmm, hmm), None
    lit = _light(st, X, Y, relief, n, wmm, hmm, occ)
    cast = lit.reshape(-1, 3).mean(0)  # the colour of the light, which white balance takes out
    paper = sheet.albedo * lit
    if kind == "phone":
        # Glare: a broad haze off the paper, and a tighter shine off ink and graphite.
        ndh, strength = _glare(st, shot.centre, X, Y, relief, n, wmm, hmm, occ)
        g = np.clip(sheet.sheen, 0, 1) if sheet.sheen is not None else 0.0
        spec = strength * (st.sheen * ndh**st.sheen_power * (1 - g) + 0.5 * g * ndh**60)
        paper = paper + spec[..., None] * st.key_tint
    paper = blur(paper, 0.4)

    sx, sy, ex, ey, focus = footprint(sheet, shot, kind, w, h)
    on_paper = sample(sheet.mask, sx * ppmm, sy * ppmm, fill=0.0)
    image = sample(paper, sx * ppmm, sy * ppmm)

    # The desk, shaded by the room and by the sheet lying on it.
    res = 6.0
    x0, x1, y0, y1 = desk_bounds(ex, ey, wmm, hmm)
    # The same desk on every page: its grain comes from the background's own seed.
    desk, DX, DY = _desk(st, "scan" if kind == "scan" else st.desk, x0, y0, x1, y1, res, aspects.rng("background"), surface)
    DZ = surface.height_at(DX, DY) if surface is not None else 0 * DX
    under = sample(sheet.mask, DX * ppmm, DY * ppmm, fill=0.0)
    lamp = st.light + [wmm / 2, hmm / 2, 0.0]
    qx, qy = DX.copy(), DY.copy()
    for _ in range(2):
        lift = sheet.lift(qx, qy) + 0.1 if kind == "phone" else 0.1
        qx = DX + lift * (lamp[0] - DX) / lamp[2]
        qy = DY + lift * (lamp[1] - DY) / lamp[2]
    dropped = blur(sample(sheet.mask, qx * ppmm, qy * ppmm, fill=0.0), 0.6 * res)
    contact = blur(under, 1.5 * res)
    flat = _normals(DZ, res) if surface is not None and surface.height is not None else (np.zeros_like(DX), np.zeros_like(DX), np.ones_like(DX))
    desk = desk * _light(st, DX, DY, DZ, flat, wmm, hmm, occ, shadow=0.85 * dropped, ambient_block=0.55 * contact)
    if kind == "phone":
        ndh, strength = _glare(st, shot.centre, DX, DY, DZ, flat, wmm, hmm, occ)
        gloss = desk_gloss(st, surface)
        desk = desk + (gloss * strength * ndh**40 * (1 - 0.85 * dropped))[..., None] * st.key_tint
    behind = sample(desk, (ex - x0) * res, (ey - y0) * res)
    m = on_paper[..., None]
    image = image * m + behind * (1 - m)

    return _develop(image, on_paper, st, kind, rng, w, h, focus, cast)


def desk_gloss(st, surface):
    """How much of the lamp the desk mirrors: its clear coat, or a procedural desk's varnish."""
    if surface is not None and surface.textured:
        return 0.35 * surface.coat
    return st.desk_gloss * DESKS[surface.procedural if surface is not None else st.desk][3]


def footprint(sheet, shot, kind, w, h):
    """Where each output pixel's ray meets the sheet and the desk (mm), and how far it went."""
    f = 4
    gh, gw = int(np.ceil((h - 1) / f)) + 1, int(np.ceil((w - 1) / f)) + 1
    gv, gu = np.mgrid[0:gh, 0:gw].astype(np.float64) * f
    dx, dy, dz = shot.rays(gu, gv)
    C = shot.centre
    t = -C[2] / dz
    px, py = C[0] + t * dx, C[1] + t * dy
    desk_x, desk_y = px.copy(), py.copy()
    for _ in range(5):  # walk the ray up to the bent sheet
        zt = sheet.lift(px, py) if kind == "phone" else 0.0
        t = (zt - C[2]) / dz
        px, py = C[0] + t * dx, C[1] + t * dy
    depth = t * (dx * shot.forward[0] + dy * shot.forward[1] + dz * shot.forward[2])
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    U, V = xs / f, ys / f
    grid = [a.astype(np.float32) for a in (px, py, desk_x, desk_y, depth)]
    return tuple(sample(a, U, V) for a in grid)


def desk_bounds(ex, ey, wmm, hmm):
    """The stretch of desk in view, in mm, capped a little way past the sheet."""
    x0, x1 = max(ex.min(), -250) - 2, min(ex.max(), wmm + 250) + 2
    y0, y1 = max(ey.min(), -250) - 2, min(ey.max(), hmm + 250) + 2
    return x0, x1, y0, y1


def _flattened(sheet):
    """Under the scanner's lid, folds press nearly flat."""
    return replace(sheet, relief=sheet.relief * 0.6)


def _scanner(st):
    return replace(st, light=np.array([0.0, -9000.0, 9000.0]), key=0.35, key_tint=np.ones(3, np.float32),
                   ambient_tint=np.ones(3, np.float32), ambient_slope=np.zeros(2), white=np.ones(3, np.float32),
                   brightness=244.0, grain=0.25, vignette=0.0, fringe=0.0, sharpen=0.25, quality=90)


def _develop(image, on_paper, st, kind, rng, w, h, depth, cast):
    """What the phone's camera pipeline makes of the light that reached it."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if depth is not None and kind == "phone":
        # Focus on the middle of the page; the rest is a little soft in proportion to depth.
        mid = np.median(depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3])
        # How fast it softens with depth depends on the phone's lens and how close it is.
        sigma = 0.45 + rng.uniform(7, 14) * np.abs(depth - mid * rng.uniform(0.98, 1.02)) / mid
        soft = blur(image, 1.1)
        softer = blur(image, 2.2)
        a = np.clip((sigma - 0.45) / 0.65, 0, 1)[..., None]
        b = np.clip((sigma - 1.1) / 1.1, 0, 1)[..., None]
        image = blur(image, 0.45) * (1 - a) + soft * a
        image = image * (1 - b) + softer * b
        if rng.random() < 0.1:
            # Now and then the hand moved a little while the shutter was open.
            length, a = rng.uniform(1.0, 3.0), rng.uniform(0, np.pi)
            steps = np.linspace(-0.5, 0.5, 7) * length
            image = sum(sample(image, xs + s * np.cos(a), ys + s * np.sin(a)) for s in steps) / len(steps)
    # White balance takes out the light's colour (not the paper's), leaving a small cast.
    image = image / (cast / cast.mean()) * st.white
    paper = on_paper > 0.99
    if paper.sum() > 100:
        level = np.percentile(image[paper].mean(-1), 60)
    else:
        level = np.percentile(image.mean(-1), 60)
    image = image * (to_linear(st.brightness / 255.0) / max(level, 1e-6))
    r = np.hypot((xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)) / np.sqrt(2)
    if st.vignette:
        image *= (1 - st.vignette * r**2)[..., None]
    if st.fringe:
        cx, cy = (w - 1) / 2, (h - 1) / 2
        for c, k in ((0, 1 + st.fringe), (2, 1 - st.fringe)):
            image[..., c] = sample(image[..., c], cx + (xs - cx) * k, cy + (ys - cy) * k)
    if kind != "none":
        # Shot noise grows with brightness; the phone's denoiser leaves it fine and soft.
        luma = blur(rng.standard_normal((h, w)).astype(np.float32), 0.6)
        luma /= luma.std() + 1e-9
        chroma = blur(rng.standard_normal((h, w, 3)).astype(np.float32), 2.0)
        chroma /= chroma.std() + 1e-9
        sd = st.grain * (0.0015 + 0.008 * np.sqrt(np.clip(image, 0, None)))
        image = image + sd * (luma[..., None] + 0.35 * chroma)
        if kind == "scan":
            image *= (1 + 0.002 * band((1, w), rng, 1, 40)[0])[None, :, None]
    if kind == "phone":
        # Phones lift what is darker than its surroundings (their HDR tone mapping), and
        # take the colour out of dark parts, where it would be chroma noise.
        weights = np.array([0.2126, 0.7152, 0.0722], np.float32)
        Y = np.clip(image @ weights, 1e-5, None)
        logY = np.log(Y)
        near, around = blur(logY, 0.012 * w), blur(logY, 0.1 * w)
        lift = rng.uniform(0.2, 0.35)
        # Only what is itself darker than its surroundings is lifted, so the paper beside
        # a dark pen doesn't glow.
        image = image * np.exp(lift * np.clip(around - np.maximum(near, logY), 0, None))[..., None]
        Y = np.clip(image @ weights, 1e-5, None)
        keep = 0.55 + 0.45 * smoothstep(0.01, 0.2, Y)
        image = Y[..., None] + (image - Y[..., None]) * keep[..., None]
    if kind == "phone":
        # Phones keep the brightest parts of a page from burning out (their HDR): above
        # about the paper's own level, brightness rolls off towards white instead of clipping.
        # Only the broad brightness is compressed (local tone mapping): the folds of a white
        # blanket or the creases on a sunlit page keep their contrast.
        knee_lin = 1.1 * float(to_linear(st.brightness / 255.0))
        knee = float(np.log(knee_lin))
        Y = np.clip(image @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-6, None)
        logY = np.log(Y)
        base = blur(logY, 0.02 * w)
        room = float(np.log(max(0.93 / knee_lin, 1.02)))  # the broad level stays just under white
        over = np.maximum(base - knee, 0)
        pressed = np.where(base > knee, knee + room * np.tanh(over / room), base)
        image = image * np.exp(pressed - base)[..., None]
    out = to_srgb(image)
    if kind == "phone":
        # The phone's contrast curve, on brightness alone, so it adds no colour.
        l = np.clip(out @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-4, 1)
        out = out * ((l + 0.18 * l * (1 - l) * (2 * l - 1)) / l)[..., None]
    if st.sharpen:
        out = out + st.sharpen * (out - blur(out, 1.2))
    out = np.clip(out * 255 + rng.uniform(-0.5, 0.5, out.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(out), (95 if kind == "none" else st.quality)

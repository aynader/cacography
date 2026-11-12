"""Cloth made thread by thread: woven, looped and printed fabrics for the backgrounds.

Scanned cloth textures are close-ups of one bolt each, in one colour, with no pattern.
These are built from the threads up, so thread count, weave, yarn, dye and print can all
vary, and each comes out as one repeating tile of colour, normal and roughness maps:

- a weave (plain, twill, basket) of warp and weft threads, each thread a round yarn that
  rises over the threads it crosses and dips under the next, so the light picks out the
  weave as it does on real cloth; yarns vary in thickness along their length (slubs),
  wander a little, carry their twist, and differ a little in dye from one to the next;
- terry: piled yarn loops (towels), optionally sheared into a raised pattern (jacquard);
- woven patterns where each thread carries its own colour (stripes, plaid, gingham), and
  printed ones dyed over the threads (dots, a small print, a broad painted grid);
- fuzz: loose fibres lying on top, catching the light, more on towels and linen than on
  smooth cotton.

Stray threads and lint lying on the cloth are geometry (`strays`), so they never repeat
with the tile. Everything is periodic over the tile, so tiling shows no seam.
"""
import numpy as np

from .imaging import blur, rgb

MAX_PX = 2048  # the largest tile, per side

# Dyes for cloth, sRGB: naturals and the colours bedding, towels and shirts come in.
NATURALS = [(222, 214, 196), (206, 192, 166), (186, 168, 140), (238, 234, 224), (170, 160, 146), (214, 206, 190)]
DYED = [(70, 72, 78), (150, 150, 152), (40, 52, 88), (110, 36, 48), (46, 82, 60), (196, 152, 60), (226, 216, 196),
        (196, 150, 150), (235, 235, 232), (32, 32, 34), (40, 110, 116), (190, 170, 140), (120, 150, 170), (88, 110, 84),
        (150, 100, 160), (200, 120, 70), (230, 196, 120), (130, 170, 190), (176, 196, 160), (240, 180, 180)]
PRINT_INKS = [(200, 60, 40), (220, 150, 40), (40, 70, 130), (30, 30, 34), (60, 120, 90), (120, 50, 90), (90, 130, 170),
              (210, 110, 120), (160, 120, 60), (70, 60, 50)]


# -- periodic noise -----------------------------------------------------------

def _window(f, lo, hi):
    """A smooth band pass between wavelengths lo and hi (pixels)."""
    a, b = 1.0 / hi, 1.0 / lo
    return 1 / (1 + (a / np.maximum(f, 1e-9)) ** 4) / (1 + (f / b) ** 4)


def _pn1(n, length, rng, lo, hi):
    """`n` rows of noise along `length` pixels, wrapping round, wavelengths lo..hi."""
    spec = np.fft.rfft(rng.standard_normal((n, length)).astype(np.float32), axis=1)
    f = np.fft.rfftfreq(length)[None, :]
    w = _window(f, lo, hi)
    w[:, 0] = 0
    out = np.fft.irfft(spec * w, n=length, axis=1).astype(np.float32)
    return out / (out.std() + 1e-9)


def _pn2(shape, rng, lo, hi, sx=1.0, sy=1.0):
    """2D noise wrapping round the tile; sy > 1 stretches its features along x, sx along y."""
    h, w = shape
    spec = np.fft.rfft2(rng.standard_normal(shape).astype(np.float32))
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
    win = _window(np.hypot(fx * sx, fy * sy), lo, hi)
    win[0, 0] = 0
    out = np.fft.irfft2(spec * win, s=shape).astype(np.float32)
    return out / (out.std() + 1e-9)


def _wrap_blur(a, sigma):
    """A blur that wraps round the tile, so the tile still repeats seamlessly."""
    if sigma <= 0:
        return a
    h, w = a.shape[:2]
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2)).astype(np.float32)
    if a.ndim == 3:
        return np.stack([np.real(np.fft.ifft2(np.fft.fft2(a[..., c]) * g)) for c in range(a.shape[2])], -1).astype(np.float32)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g)).astype(np.float32)


# -- weaves -------------------------------------------------------------------

def _repeat(weave, rng):
    """The weave's repeat: True where the warp thread lies over the weft."""
    if weave == "plain":
        return np.array([[1, 0], [0, 1]], bool)
    if weave.startswith("twill"):
        up, n = {"twill21": (2, 3), "twill22": (2, 4), "twill31": (3, 4)}[weave]
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        return ((i - j) % n) < up  # each row one thread on: the diagonal
    if weave.startswith("basket"):
        g = int(weave[-1])
        i, j = np.meshgrid(np.arange(2 * g), np.arange(2 * g))
        return ((i // g + j // g) % 2) == 0
    raise ValueError(weave)


def _weave(n_warp, n_weft, p, rng, weave="plain", cover=0.9, slub=0.15, wander=0.04, twist=0.0, flat=1.0):
    """The threads' surface over the tile: height (in thread radii), which thread is on
    top (True: warp), each pixel's warp and weft thread, and how far across its thread a
    pixel lies (0 at the middle, 1 at the edge), for the twist and the shading.

    Warp threads run down the tile (along y), weft threads across it."""
    H, W = n_weft * p, n_warp * p
    rep = _repeat(weave, rng)
    S = np.where(rep, 1.0, -1.0).astype(np.float32)
    r = S.shape[0]
    y = (np.arange(H, dtype=np.float32) + 0.5)[:, None]
    x = (np.arange(W, dtype=np.float32) + 0.5)[None, :]
    i = (x // p).astype(np.int32) % n_warp
    j = (y // p).astype(np.int32) % n_weft
    fx, fy = x / p - np.floor(x / p), y / p - np.floor(y / p)

    def thread(n, length, along_pos, idx_cross, frac_across, frac_along, sign, crossing):
        # Thickness along each thread: slubs, thicker stretches of yarn.
        width = cover * (1 + slub * np.maximum(_pn1(n, length, rng, 3 * p, 18 * p), -0.6) ** 2 * np.sign(1)
                         - 0.04 * np.abs(_pn1(n, length, rng, p, 4 * p)))
        width = np.clip(width, 0.35, 1.25)
        off = wander * _pn1(n, length, rng, 4 * p, 30 * p)  # threads aren't quite straight
        w_here = width[idx_cross, along_pos]
        o_here = off[idx_cross, along_pos]
        d = np.abs(frac_across - 0.5 - o_here) / (w_here / 2)
        prof = np.sqrt(np.clip(1 - d**2, 0, 1))
        # Over and under: the thread's middle rises where it lies on top and dips where it
        # passes under, easing between crossings.
        t = frac_along - 0.5
        k0 = np.floor(t).astype(np.int32)
        a = t - k0
        s0 = sign * crossing(k0)
        s1 = sign * crossing(k0 + 1)
        middle = s0 + (s1 - s0) * (1 - np.cos(np.pi * a)) / 2
        hgt = np.where(d < 1, 0.45 * flat * middle + prof * (w_here / cover) ** 0.5, -3.0)
        return hgt.astype(np.float32), np.clip(d, 0, 1).astype(np.float32), prof

    yi = np.broadcast_to(np.arange(H)[:, None], (H, W))
    xi = np.broadcast_to(np.arange(W)[None, :], (H, W))
    I, J = np.broadcast_to(i, (H, W)), np.broadcast_to(j, (H, W))
    warp_h, warp_d, warp_p = thread(n_warp, H, yi, I, np.broadcast_to(fx, (H, W)), y / p + 0 * x, 1.0,
                                    lambda k: S[(k % r), (I % r)])
    weft_h, weft_d, weft_p = thread(n_weft, W, xi, J, np.broadcast_to(fy, (H, W)), x / p + 0 * y, -1.0,
                                    lambda k: S[(J % r), (k % r)])
    top = warp_h >= weft_h
    height = np.maximum(np.maximum(warp_h, weft_h), -1.2)
    across = np.where(top, warp_d, weft_d)
    if twist > 0:
        # The yarn's twist: fine slanted ridges along each thread.
        ph_warp = y / p * twist + np.broadcast_to(fx, (H, W)) * 1.2
        ph_weft = x / p * twist + np.broadcast_to(fy, (H, W)) * 1.2
        ridge = np.where(top, np.cos(2 * np.pi * ph_warp), np.cos(2 * np.pi * ph_weft))
        height = height + np.where(height > -1, 0.12 * ridge * np.where(top, warp_p, weft_p), 0)
    return height.astype(np.float32), top, I, J, across


def _shade(height, across):
    """How much light the weave's own shape lets reach each spot: the tops of threads
    bright, their flanks and the gaps between them in shadow."""
    lo, hi = -1.2, height.max()
    ao = np.clip((height - lo) / (hi - lo + 1e-6), 0, 1) ** 0.7
    return (0.35 + 0.65 * ao) * (1 - 0.18 * across**2)


# -- yarns and colours --------------------------------------------------------

def _thread_colours(n, base, rng, spread=0.05, sett=None):
    """One colour per thread: the base or the sett's colour, each a little off."""
    cols = np.empty((n, 3), np.float32)
    if sett is None:
        cols[:] = base
    else:
        cols[:] = sett[np.arange(n) % len(sett)]
    lum = np.exp(rng.normal(0, spread, (n, 1))).astype(np.float32)
    hue = np.exp(rng.normal(0, spread * 0.3, (n, 3))).astype(np.float32)
    return cols * lum * hue


def _sett(n, rng, colours, min_run, max_run):
    """A woven stripe sequence of `n` threads, mirrored as tartans are: a list of colours per thread."""
    half = []
    while len(half) < n // 2:
        c = colours[int(rng.integers(len(colours)))]
        half += [c] * int(rng.integers(min_run, max_run + 1))
    half = half[: n // 2]
    seq = half + half[::-1]
    seq += [seq[-1]] * (n - len(seq))
    return np.array(seq, np.float32)


def _melange(shape, top, rng, p, amount, light):
    """Heathered yarn: fibres of a lighter shade spun in, streaks along each thread."""
    if amount <= 0:
        return np.zeros(shape, np.float32)
    along_y = _pn2(shape, rng, 0.6 * p, 4 * p, sx=1.0, sy=8.0)
    along_x = _pn2(shape, rng, 0.6 * p, 4 * p, sx=8.0, sy=1.0)
    m = np.where(top, along_y, along_x)
    return (amount * np.clip((m - 0.6) * 1.5, 0, 1) * light).astype(np.float32)


def _fibres(shape, rng, n, length, curl, width=1.0):
    """Loose fibres lying on the cloth: how much of each pixel they cover."""
    h, w = shape
    if n <= 0:
        return np.zeros(shape, np.float32)
    steps = max(4, int(length / 0.7))
    a = rng.uniform(0, 2 * np.pi, (n, 1)) + np.cumsum(rng.normal(0, curl, (n, steps)), 1)
    lens = rng.uniform(0.4, 1.0, (n, 1)) * 0.7
    xs = rng.uniform(0, w, (n, 1)) + np.cumsum(np.cos(a) * lens, 1)
    ys = rng.uniform(0, h, (n, 1)) + np.cumsum(np.sin(a) * lens, 1)
    cov = np.zeros(h * w, np.float32)
    idx = (np.floor(ys).astype(np.int64) % h) * w + (np.floor(xs).astype(np.int64) % w)
    # Fibres thin out towards their ends.
    fade = np.sin(np.linspace(0.15, np.pi - 0.15, steps))[None, :].repeat(n, 0).astype(np.float32)
    np.add.at(cov, idx.ravel(), fade.ravel())
    cov = _wrap_blur(cov.reshape(h, w), 0.35 * width)
    return 1 - np.exp(-1.6 * cov)


def _normals(height_mm, px_mm):
    """A tangent-space normal map (x along u, y along v down the tile) from heights in mm."""
    gy = (np.roll(height_mm, -1, 0) - np.roll(height_mm, 1, 0)) / (2 * px_mm)
    gx = (np.roll(height_mm, -1, 1) - np.roll(height_mm, 1, 1)) / (2 * px_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


# -- prints -------------------------------------------------------------------

def _dots(H, W, rng, px_mm, ink):
    """Polka dots on a slightly irregular staggered grid, their edges soaked into the cloth."""
    spacing = rng.uniform(8, 22) / px_mm
    nx = max(2, int(round(W / spacing)))
    ny = max(2, int(round(H / (spacing * 0.866))) // 2 * 2)
    sx, sy = W / nx, H / ny
    radius = rng.uniform(0.18, 0.3) * sx
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    row = np.floor(y / sy)
    cx = (np.floor((x - (row % 2) * sx / 2) / sx) + 0.5) * sx + (row % 2) * sx / 2
    cy = (row + 0.5) * sy
    jitter = rng.normal(0, 0.03 * sx, (ny, nx + 1, 2))
    ri, ci = row.astype(int) % ny, (np.floor((x - (row % 2) * sx / 2) / sx).astype(int)) % (nx + 1)
    dx = (x - cx - jitter[ri, ci, 0] + W / 2) % W - W / 2
    dy = (y - cy - jitter[ri, ci, 1] + H / 2) % H - H / 2
    d = np.hypot(dx, dy) / radius
    return [(np.clip((1 - d) * radius * px_mm / 0.25, 0, 1), ink)]


def _floral(H, W, rng, px_mm, inks):
    """A small all-over print: soft blobs in one colour, small five-petal flowers and leaves in others."""
    layers = []
    if rng.random() < 0.5:  # soft sprigs behind the flowers, in a paler shade
        blobs = _pn2((H, W), rng, 4 / px_mm, 14 / px_mm)
        layers.append((np.clip((blobs - 1.4) * 3, 0, 1) * 0.4, inks[0]))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    flowers = np.zeros((H, W), np.float32)
    leaves = np.zeros((H, W), np.float32)
    n = int(H * W * px_mm**2 / rng.uniform(250, 600))
    for _ in range(min(n, 400)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(1.5, 3.2) / px_mm
        rot = rng.uniform(0, 2 * np.pi)
        x0, x1, y0, y1 = int(cx - 3 * r), int(cx + 3 * r) + 1, int(cy - 3 * r), int(cy + 3 * r) + 1
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        ang = np.arctan2(dy, dx) - rot
        rr = np.hypot(dx, dy)
        petal = np.clip((r * (0.55 + 0.45 * np.abs(np.cos(2.5 * ang))) - rr) * px_mm / 0.2, 0, 1)
        centre = np.clip((0.3 * r - rr) * px_mm / 0.2, 0, 1)
        la = rot + rng.uniform(0.5, 2.5)
        lx, ly = dx * np.cos(la) + dy * np.sin(la) - 1.9 * r, -dx * np.sin(la) + dy * np.cos(la)
        leaf = np.clip((1 - (lx / (1.1 * r)) ** 2 - (ly / (0.45 * r)) ** 2) * 4, 0, 1)
        ys, xs = np.arange(y0, y1) % H, np.arange(x0, x1) % W
        flowers[np.ix_(ys, xs)] = np.maximum(flowers[np.ix_(ys, xs)], petal * (1 - centre))
        leaves[np.ix_(ys, xs)] = np.maximum(leaves[np.ix_(ys, xs)], leaf * (petal < 0.5))
    layers.append((leaves, inks[2 % len(inks)]))
    layers.append((flowers, inks[1]))
    return layers


def _grid(H, W, rng, px_mm, inks):
    """A broad painted grid: bands of several colours crossing, wobbling a little, see-through where they cross."""
    layers = []
    for axis in (0, 1):
        length, across = (H, W) if axis == 1 else (W, H)
        n = int(rng.integers(3, 6))
        pos = np.sort(rng.uniform(0, across, n))
        for k in range(n):
            width = rng.uniform(8, 18) / px_mm
            wob = _pn1(1, length, rng, 40 / px_mm, 200 / px_mm)[0] * (1.2 / px_mm)
            c = np.arange(across, dtype=np.float32)
            d = np.abs((c[None, :] - pos[k] - wob[:, None] + across / 2) % across - across / 2)
            band = np.clip((width / 2 - d) * px_mm / 0.3, 0, 1)
            if axis == 0:
                band = band.T
            layers.append((band * 0.92, inks[int(rng.integers(len(inks)))]))
    return layers


# -- terry ----------------------------------------------------------------------

def _terry(n, L, rng, jacquard=False):
    """A tile of terry: yarn loops standing up from a woven ground, crowded together.

    Each loop is a ring of twisted yarn seen at some tilt; loops lean together in patches.
    With `jacquard`, the pile is sheared off along a pattern of grooves, leaving the flat
    woven ground showing. Returns height (in loop radii), shade and the pile's mask."""
    size = n * L
    height = np.full(size * size, -1.0, np.float32)
    lean = _pn2((n, n), rng, 3, 12)  # loops stand taller or lie flatter in patches
    # Loops come up from the ground weave in rows, evenly, a few per cell, each pulled a
    # little off its place: no bare patches, no clumps.
    gy, gx = np.mgrid[0:n * 2, 0:n * 2].astype(np.float64) * (L / 2)
    keep = rng.random(gx.size) < 0.9
    gy, gx = gy.ravel()[keep], gx.ravel()[keep]
    count = len(gx)
    cx = (gx + rng.uniform(0, L / 2, count)) % size
    cy = (gy + rng.uniform(0, L / 2, count)) % size
    # Each loop: an arc of yarn most of the way round an ellipse, seen at a tilt.
    m = 40
    r = rng.uniform(0.22, 0.4, (count, 1)) * L
    q = rng.uniform(0.3, 1.0, (count, 1))
    phi = rng.uniform(0, np.pi, (count, 1))
    th0 = rng.uniform(0, 2 * np.pi, (count, 1))
    th = th0 + np.linspace(0, 1, m)[None, :] * rng.uniform(1.5, 2.0, (count, 1)) * np.pi
    eu, ev = r * np.cos(th), r * q * np.sin(th)
    px = cx[:, None] + eu * np.cos(phi) - ev * np.sin(phi)
    py = cy[:, None] + eu * np.sin(phi) + ev * np.cos(phi)
    tall = rng.uniform(0.45, 1.2, (count, 1)) + 0.15 * lean[(cy // L).astype(int) % n, (cx // L).astype(int) % n][:, None]
    # The loop arches up from the ground and back; its twisted yarn ridges along it.
    arch = np.sin(np.linspace(0.1, np.pi - 0.1, m))[None, :]
    ridge = 0.06 * np.cos(np.linspace(0, 1, m)[None, :] * rng.integers(8, 14) * np.pi)
    core = tall * (0.35 + 0.65 * arch) + ridge
    t = 0.13 * L  # the yarn's radius
    T = int(np.ceil(t))
    off = np.arange(-T, T + 1)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    disc = np.sqrt(np.clip(1 - (ox**2 + oy**2) / (t * t), 0, 1)).astype(np.float32)
    keep = disc > 0
    oy, ox, disc = oy[keep], ox[keep], disc[keep]
    for chunk in range(0, count, 4000):
        sl = slice(chunk, chunk + 4000)
        iy = (np.floor(py[sl]).astype(np.int64)[..., None] + oy) % size
        ix = (np.floor(px[sl]).astype(np.int64)[..., None] + ox) % size
        vals = core[sl][..., None] + 0.5 * disc
        np.maximum.at(height, (iy * size + ix).ravel(), vals.astype(np.float32).ravel())
    height = height.reshape(size, size)
    pile = np.ones((size, size), np.float32)
    if jacquard:
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        period = size / int(rng.integers(3, 7))
        gap = rng.uniform(0.12, 0.2) * period / 2.5
        kind = rng.integers(3)
        if kind == 0:  # chevrons
            u = np.abs(((x / period) % 1) - 0.5) * period + y
        elif kind == 1:  # diagonal ribs
            u = x + y
        else:  # nested squares, like a key pattern
            u = np.maximum(np.abs(((x / period) % 1) - 0.5), np.abs(((y / period) % 1) - 0.5)) * period * 2
        groove = np.abs(((u / (period / 2.5)) % 1) - 0.5) * (period / 2.5)
        pile = np.clip((groove - gap) / (0.25 * L), 0, 1)
        ground, _, _, _, _ = _weave(size // 4, size // 4, 4, rng, "plain", cover=0.95, slub=0.05)
        height = np.where(pile > 0.5, height, -0.6 + 0.12 * ground)
    # Each strand stands out by the shadow round it: shade by how far a spot stands above
    # what is close round it, and a little by its height overall.
    local = height - _wrap_blur(height, 0.3 * L)
    shade = 0.72 + 0.55 * np.clip(local, -0.6, 0.4) + 0.12 * np.clip((height + 1.0) / 2.2, 0, 1)
    return height, np.clip(shade, 0.3, 1.0).astype(np.float32), pile


# -- the cloths ---------------------------------------------------------------

def _tile_threads(tpmm, p, want_mm, repeat=1):
    """Threads per tile side: about `want_mm` of cloth, within MAX_PX, a multiple of `repeat`."""
    n = int(round(want_mm * tpmm / repeat)) * repeat
    n = min(n, MAX_PX // p // repeat * repeat)
    return max(n, repeat * 2)


def make(kind, rng, pattern=None):
    """One tile of the cloth `kind`: (colour (linear), normal, roughness, tile in mm, sheen)."""
    k = dict(CLOTHS[kind])
    pattern = pattern or k.get("pattern")
    tpmm = rng.uniform(*k["tpmm"])
    p = k.get("p", 6)
    sheen = k.get("sheen", 0.4)
    if k.get("terry"):
        L = 18
        loop_mm = rng.uniform(1.5, 2.2)
        n = 64
        height, shade, pile = _terry(n, L, rng, jacquard=(pattern == "jacquard") or rng.random() < k.get("jacquard", 0))
        px_mm = loop_mm / L
        size = n * L
        base = rgb(_pick(rng, k["colours"]))
        colour = np.broadcast_to(base, (size, size, 3)) * shade[..., None]
        colour = colour * np.exp(0.06 * _pn2((size, size), rng, 4 * L, 30 * L))[..., None]
        fib = _fibres((size, size), rng, int(size * size / 90), 2.5 * L, 0.5)
        colour = colour * (1 - 0.35 * fib[..., None]) + 0.35 * fib[..., None] * np.minimum(base * 1.25 + 0.02, 1)
        hmm = (height + 0.3 * fib) * (0.14 * loop_mm)
        rough = np.full((size, size), 0.95, np.float32)
        return (np.clip(colour, 0, 1).astype(np.float32), _normals(_wrap_blur(hmm, 1.0), px_mm), rough,
                (size * px_mm, size * px_mm), sheen)

    weave = k["weave"] if isinstance(k["weave"], str) else str(rng.choice(k["weave"]))
    rep = _repeat(weave, rng).shape[0]
    pitch_mm = 1 / tpmm
    sett = None
    if pattern in ("stripes", "plaid", "gingham"):
        inks = [rgb(_pick(rng, DYED)) for _ in range(int(rng.integers(2, 5)))]
        ground = rgb(_pick(rng, NATURALS + [(235, 235, 232)] * 3)) if pattern != "plaid" else inks[0]
        if pattern == "gingham":
            check = int(round(rng.uniform(2.5, 7) * tpmm))
            n = _tile_threads(tpmm, p, 90, 2 * check * rep)
            seq = np.array(([ground] * check + [inks[0]] * check) * (n // (2 * check)), np.float32)
            warp_sett = weft_sett = seq
        else:
            n = _tile_threads(tpmm, p, 120 if pattern == "plaid" else 100, 2 * rep)
            runs = (max(1, int(tpmm * 0.6)), max(2, int(tpmm * 9)))
            warp_sett = _sett(n, rng, [ground] * (2 if pattern == "stripes" else 0) + inks, *runs)
            weft_sett = warp_sett if pattern == "plaid" and rng.random() < 0.7 else (
                _sett(n, rng, inks, *runs) if pattern == "plaid" else np.array([ground] * n, np.float32))
        sett = (warp_sett, weft_sett)
    else:
        n = _tile_threads(tpmm, p, k.get("tile", 100), rep)
    H = W = n * p
    px_mm = pitch_mm / p
    height, top, I, J, across = _weave(n, n, p, rng, weave, cover=rng.uniform(*k.get("cover", (0.85, 0.97))),
                                       slub=rng.uniform(*k.get("slub", (0.05, 0.15))), wander=k.get("wander", 0.04),
                                       twist=k.get("twist", 0.0), flat=k.get("flat", 1.0))
    if k.get("warp_weft"):  # denim: indigo warp, ecru weft
        warp_base, weft_base = rgb(_pick(rng, k["warp_weft"][0])), rgb(_pick(rng, k["warp_weft"][1]))
    else:
        warp_base = weft_base = rgb(_pick(rng, k["colours"]))
    spread = k.get("spread", 0.05)
    if sett is not None:
        cw = _thread_colours(n, None, rng, spread, sett[0])
        cf = _thread_colours(n, None, rng, spread, sett[1])
        warp_base = sett[0].mean(0)
    else:
        cw = _thread_colours(n, warp_base, rng, spread)
        cf = _thread_colours(n, weft_base, rng, spread)
    # Each yarn's dye also drifts along its length a little, more on slubby yarn.
    drift_w = np.exp(spread * 0.6 * _pn1(n, H, rng, 2 * p, 25 * p))
    drift_f = np.exp(spread * 0.6 * _pn1(n, W, rng, 2 * p, 25 * p))
    yi = np.arange(H)[:, None]
    xi = np.arange(W)[None, :]
    base = np.where(top[..., None], cw[I] * drift_w[I, yi][..., None], cf[J] * drift_f[J, xi][..., None])
    mel = _melange((H, W), top, rng, p, k.get("melange", 0.0), 1.0)
    base = base * (1 - mel[..., None]) + mel[..., None] * np.minimum(base * 1.6 + 0.08, 1)
    if pattern in ("dots", "print", "grid"):
        inks = [rgb(_pick(rng, PRINT_INKS)) for _ in range(3)]
        layers = _dots(H, W, rng, px_mm, inks[0]) if pattern == "dots" else (
            _floral(H, W, rng, px_mm, inks) if pattern == "print" else _grid(H, W, rng, px_mm, inks))
        for cover_map, ink in layers:
            # Print dye soaks along the threads a little, and the tops of threads take it
            # less than the hollows, which flatten a print's edges and colour.
            soak = _wrap_blur(cover_map, 0.6 * p / 6)
            soak = soak * (0.85 + 0.15 * (1 - np.clip(height, 0, 1)))
            # Dyes multiply over the cloth's own colour, as a print on white does.
            base = base * (1 - soak[..., None]) + soak[..., None] * base * ink / np.maximum(base.mean(-1, keepdims=True), 0.2)
    shade = _shade(height, across)
    colour = base * shade[..., None]
    fuzz = k.get("fuzz", 0.3) * rng.uniform(0.6, 1.3)
    fib = _fibres((H, W), rng, int(H * W / (p * p) * 0.06 * fuzz), rng.uniform(4, 10) * p, 0.35)
    tint = np.minimum(_wrap_blur(base, 2.0) * 1.3 + 0.03, 1)
    colour = colour * (1 - 0.3 * fib[..., None]) + 0.3 * fib[..., None] * tint
    hmm = (height + 0.4 * fib) * (0.42 * pitch_mm)
    rough = np.clip(0.88 - 0.1 * np.clip(height, 0, 1) + 0.08 * fib, 0.6, 1.0).astype(np.float32)
    return (np.clip(colour, 0, 1).astype(np.float32), _normals(_wrap_blur(hmm, 0.5), px_mm), rough,
            (W * px_mm, H * px_mm), sheen)


def _pick(rng, colours):
    return colours[int(rng.integers(len(colours)))]


# Each cloth: weave, threads per mm, yarn character, colours. tile: mm of cloth per tile.
CLOTHS = {
    # Cotton percale and poplin: fine, even, smooth, a little fuzz.
    "bedsheet": dict(weave="plain", tpmm=(3.0, 4.5), p=4, slub=(0.02, 0.08), cover=(0.9, 0.98), fuzz=0.2, sheen=0.3,
                     colours=[(238, 236, 230)] * 4 + DYED, spread=0.03, tile=90),
    # Linen: coarse, uneven yarn, slubs, open weave, crinkled after washing.
    "linen": dict(weave="plain", tpmm=(1.3, 2.2), p=6, slub=(0.25, 0.5), cover=(0.72, 0.88), wander=0.07, fuzz=0.5,
                  sheen=0.25, colours=NATURALS * 3 + DYED, spread=0.07, melange=0.25, twist=1.2, tile=120),
    # Canvas and cotton duck: thick, round, even threads.
    "canvas": dict(weave=["plain", "basket2"], tpmm=(1.0, 1.7), p=7, slub=(0.05, 0.15), cover=(0.85, 0.95), fuzz=0.3,
                   sheen=0.2, colours=NATURALS + DYED, spread=0.05, twist=1.5, tile=120),
    # Twill suiting and chino: fine diagonal ribs, heathered yarn.
    "twill": dict(weave=["twill21", "twill22", "twill31"], tpmm=(2.2, 3.4), p=5, slub=(0.05, 0.2), cover=(0.9, 1.0),
                  fuzz=0.35, sheen=0.3, colours=NATURALS * 2 + DYED, spread=0.05, melange=0.45, twist=1.0, tile=100),
    # Denim: indigo warp over ecru weft in a 3/1 twill, slubby.
    "denim": dict(weave=["twill31", "twill21"], tpmm=(2.2, 3.2), p=5, slub=(0.15, 0.35), cover=(0.92, 1.02), fuzz=0.35,
                  sheen=0.15, warp_weft=([(38, 52, 92), (30, 40, 70), (52, 74, 120), (70, 96, 140), (24, 30, 46)],
                                         [(220, 214, 200), (200, 196, 186)]), spread=0.08, twist=1.4, tile=110),
    # Woven patterns: each thread its own colour.
    "stripes": dict(weave=["plain", "twill22"], tpmm=(2.0, 3.6), p=5, slub=(0.03, 0.12), fuzz=0.25, sheen=0.3,
                    colours=DYED, spread=0.03, pattern="stripes", twist=1.0),
    "plaid": dict(weave=["twill22", "plain"], tpmm=(1.8, 3.2), p=5, slub=(0.05, 0.15), fuzz=0.35, sheen=0.3,
                  colours=DYED, spread=0.04, pattern="plaid", melange=0.15, twist=1.0),
    "gingham": dict(weave="plain", tpmm=(2.4, 3.6), p=5, slub=(0.02, 0.08), fuzz=0.2, sheen=0.3,
                    colours=DYED, spread=0.03, pattern="gingham"),
    # Printed cotton.
    "dots": dict(weave="plain", tpmm=(2.6, 4.0), p=4, slub=(0.02, 0.08), fuzz=0.2, sheen=0.3,
                 colours=[(238, 236, 230)] * 3 + NATURALS, spread=0.03, pattern="dots", tile=110),
    "print": dict(weave="plain", tpmm=(2.6, 4.0), p=4, slub=(0.02, 0.08), fuzz=0.2, sheen=0.3,
                  colours=[(238, 236, 230)] * 3 + NATURALS, spread=0.03, pattern="print", tile=110),
    "grid": dict(weave=["plain", "basket2"], tpmm=(1.4, 2.2), p=4, slub=(0.04, 0.12), fuzz=0.3, sheen=0.25,
                 colours=[(240, 238, 232)] * 3, spread=0.03, pattern="grid", tile=420),
    # Terry towels: loops; some sheared into a raised pattern.
    "towel": dict(terry=True, tpmm=(1, 1), colours=DYED + NATURALS, jacquard=0.3, sheen=0.5),
    # Sisal and jute: thick twisted yarn in a basket weave, hairy.
    "jute": dict(weave=["basket2", "basket3", "plain"], tpmm=(0.35, 0.6), p=10, slub=(0.15, 0.3), cover=(0.8, 0.95),
                 wander=0.08, fuzz=1.4, sheen=0.1, colours=[(150, 116, 74), (176, 140, 92), (130, 100, 64), (190, 160, 116)],
                 spread=0.1, melange=0.5, twist=2.2, flat=1.3, tile=200),
}


# -- stray threads and lint ---------------------------------------------------

def strays(surface, rng, bounds, colour):
    """A few loose threads, hairs and bits of lint lying on the cloth, as small tubes
    (props' format): never many, and some cloths have none."""
    from .props import _spline, _tube

    x0, x1, y0, y1 = bounds
    area = (x1 - x0) * (y1 - y0) / 1e4  # in 10 cm squares
    things = []
    n_threads = rng.poisson(0.05 * area * (2 if surface.name in ("towel", "jute", "linen") else 1))
    n_lint = rng.poisson(0.12 * area)
    base = np.asarray(colour, np.float32)
    for kind, count in (("thread", min(n_threads, 4)), ("lint", min(n_lint, 10))):
        for _ in range(count):
            if kind == "thread":
                length, r = rng.uniform(12, 70), rng.uniform(0.06, 0.12)
                keys_n = int(rng.integers(4, 8))
            else:
                length, r = rng.uniform(1.5, 6), rng.uniform(0.03, 0.07)
                keys_n = 3
            start = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            a = rng.uniform(0, 2 * np.pi)
            keys = [start]
            for _k in range(keys_n - 1):
                a += rng.normal(0, 0.9)
                keys.append(keys[-1] + length / (keys_n - 1) * np.array([np.cos(a), np.sin(a)]))
            path = _spline(np.array(keys, np.float32), 10)
            z = surface.surface_at(path[:, 0], path[:, 1]) + r * 0.7 + 0.05
            pts = np.column_stack([path, z]).astype(np.float32)
            verts, faces = _tube(pts, np.full(len(pts), r, np.float32), 5)
            if kind == "thread" and rng.random() < 0.6:
                col = np.minimum(base * rng.uniform(0.8, 1.2), 1)  # one of the cloth's own threads
            else:
                col = rgb(_pick(rng, [(40, 36, 34), (200, 196, 190), (150, 40, 50), (60, 70, 120), (120, 96, 70)]))
            things.append(dict(verts=verts, faces=faces, material=dict(kind="plastic", colour=[float(c) for c in col], rough=0.6)))
    return things


if __name__ == "__main__":  # a preview of every cloth: python -m cacography_render.fabric out.jpg
    import sys

    from PIL import Image

    from .imaging import to_srgb

    out = sys.argv[1] if len(sys.argv) > 1 else "fabrics.jpg"
    names = sys.argv[2:] or list(CLOTHS)
    tiles = []
    for idx, name in enumerate(names):
        g = np.random.default_rng(idx + 7)
        colour, normal, rough, tile, sheen = make(name, g)
        # Shade with a light from the upper left, to see the relief, and show 40 mm at 12 px/mm.
        light = np.array([-0.4, -0.5, 0.77])
        lit = colour * (0.25 + 0.85 * np.clip((normal * light).sum(-1), 0, 1))[..., None]
        px = colour.shape[1] / tile[0]
        span = int(min(40 * px, colour.shape[0]))
        crop = lit[:span, :span]
        img = Image.fromarray((to_srgb(crop) * 255).astype(np.uint8)).resize((480, 480), Image.LANCZOS)
        tiles.append((name, img))
        print(name, colour.shape, f"{tile[0]:.0f} mm", flush=True)
    sheet = Image.new("RGB", (4 * 480, -(-len(tiles) // 4) * 500), (250, 250, 250))
    from PIL import ImageDraw

    d = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(tiles):
        x, y = (i % 4) * 480, (i // 4) * 500
        sheet.paste(img, (x, y + 20))
        d.text((x + 6, y + 4), name, fill=(0, 0, 0))
    sheet.save(out, quality=90)

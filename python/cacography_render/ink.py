"""How a pen puts its line down: pressure, flow, blobs and skips, and how ink builds up.

Strokes are drawn as an amount of ink per pixel rather than as a yes-or-no shape. A hand
presses harder going down than coming up and eases off when it moves fast, a ballpoint
lands with ink that pooled on its ball, flicks thin when lifted on the move and now and
then skips. Where strokes cross, the ink adds up. Sheet.lay_ink (paper.py) turns the
amount into colour, so crossings come out darker and more saturated, and thin strokes
lighter, the way ink absorbs light. Pencil uses the same strokes, but its graphite
catches on the paper's high points (also in lay_ink).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PENS = {
    # width: line width per pen width. give: how much the width follows pressure.
    # start: chance of extra ink when the pen lands after moving through the air, and how much.
    # blob: chance of a blob at a sharp turn. skip: skips per mm of fast, light writing.
    # flow: how much the ink's flow wanders along the line.
    "ballpoint": dict(width=1.0, give=0.12, start=(0.35, 0.45), blob=0.025, skip=0.05, flow=0.09, dot=1.25),
    "pencil": dict(width=1.15, give=0.3, start=(0.0, 0.0), blob=0.0, skip=0.0, flow=0.1, dot=1.05),
}
SCALE = 2.0  # most ink one stroke lays down, as an amount
AIR = 3.0  # in mm: a lift this long before landing lets ink pool on the ball
TURN = 0.5  # cosine between the way in and the way out at a sharp turn


def pen_of(ink):
    return "pencil" if ink == "pencil" else "ballpoint"


class Canvas:
    """Ink amount per page pixel: 1 is a single pass at the writer's usual pressure."""

    SS = 3  # strokes are drawn this much finer, then averaged down for smooth edges

    def __init__(self, width, height, ppmm, pen="ballpoint", rng=None, pressure=1.0):
        self.size = (width, height)
        self.amount = np.zeros((height, width), np.float32)
        self.ppmm = ppmm
        self.spec = PENS[pen]
        self.pen = pen
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pressure = pressure
        self.last = None  # where the pen last lifted
        self.typical = []  # per-stroke median step lengths, for what counts as fast
        self.travel = 0.0  # ink line written so far, in mm, so flow stays continuous
        # The ink's flow pulses slowly along the line: a few waves 3 to 20 mm long.
        self.waves = [(2 * np.pi / self.rng.uniform(3, 20), self.rng.uniform(0, 2 * np.pi)) for _ in range(4)]

    # Drawing ---------------------------------------------------------------

    def stroke(self, pts, width):
        p = np.asarray(pts, dtype=np.float64)
        flown = np.inf if self.last is None else float(np.hypot(*(p[0] - self.last))) / self.ppmm
        self.last = p[-1].copy()
        if len(p) >= 3:
            p = _catmull_rom(p)
        seg = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
        length = seg.sum()
        if len(p) == 1 or length < 0.6 * width:
            self._dot(p.mean(0), width)
            return
        if np.any(seg > 0):
            self.typical = (self.typical + [float(np.median(seg[seg > 0]))])[-400:]
        ref = max(float(np.median(self.typical)), 1e-6)
        s = np.r_[0.0, np.cumsum(seg)]  # arc length at each point, px
        mm = self.ppmm

        # Per point: the mean of the steps either side, smoothed along the line.
        speed = _smooth(np.r_[seg[0], (seg[:-1] + seg[1:]) / 2, seg[-1]] / ref, 7)
        dy = np.diff(p[:, 1]) / np.maximum(seg, 1e-9)
        down = _smooth(np.r_[dy[0], (dy[:-1] + dy[1:]) / 2, dy[-1]], 5)
        press = self.pressure * (1 + 0.22 * down) * np.clip(1.15 - 0.22 * speed, 0.6, 1.15)
        # Lifting on the move: the line thins out over a stretch that grows with the speed.
        end_speed = float(speed[-6:].mean())
        fade = max(0.0, end_speed - 0.8) * 0.5 * mm
        if fade > 0:
            press *= 0.4 + 0.6 * np.clip((length - s) / fade, 0, 1)
        press *= 0.8 + 0.2 * np.clip(s / (0.25 * mm), 0, 1)  # landing
        amount = press * self._flow(s / mm)
        if self.spec["start"][0] and flown > AIR and self.rng.random() < self.spec["start"][0]:
            amount *= 1 + self.spec["start"][1] * self.rng.uniform(0.3, 1) * np.exp(-s / (0.5 * mm))
        amount *= self._skips(s / mm, speed, press)
        widths = width * self.spec["width"] * (1 - self.spec["give"] + self.spec["give"] * np.clip(press, 0.3, 1.4))
        self.travel += length / mm

        blobs = []
        if self.spec["blob"]:
            for i in _turns(p, s, 0.4 * mm):
                if self.rng.random() < self.spec["blob"]:
                    blobs.append((p[i], widths[i] * self.rng.uniform(0.55, 0.8), self.rng.uniform(1.4, 1.9)))
        self._paint(p, widths, amount, blobs)

    def _dot(self, c, width):
        # A dot is a moment's pressure in one place: a ballpoint leaves it dark.
        a = self.pressure * self.spec["dot"] * self.rng.uniform(0.9, 1.15)
        r = 0.5 * width * self.spec["width"] * self.rng.uniform(1.0, 1.25)
        self._paint(np.array([c]), np.array([2 * r]), np.array([a]), [])

    def glyph(self, ch, x, y, size, font_path):
        font = ImageFont.truetype(font_path, max(4, int(size * self.SS)))
        s = self.SS
        box = font.getbbox(ch, anchor="ls")
        x0, y0 = int(np.floor(x + box[0] / s)) - 1, int(np.floor(y + box[1] / s)) - 1
        x1, y1 = int(np.ceil(x + box[2] / s)) + 1, int(np.ceil(y + box[3] / s)) + 1
        img = Image.new("L", ((x1 - x0) * s, (y1 - y0) * s), 0)
        ImageDraw.Draw(img).text(((x - x0) * s, (y - y0) * s), ch, font=font, fill=int(255 * self.pressure / SCALE), anchor="ls")
        self._add(img, x0, y0)

    def coverage(self):
        """How much of each pixel is inked, 0..1, for checks that only need the shape."""
        return np.clip(self.amount, 0, 1)

    # Parts -----------------------------------------------------------------

    def _flow(self, at_mm):
        t = self.travel + at_mm
        wave = sum(np.sin(k * t + ph) for k, ph in self.waves) / np.sqrt(len(self.waves) / 2)
        return 1 + self.spec["flow"] * wave

    def _skips(self, at_mm, speed, press):
        """Short gaps where the ball rolled without inking, mostly on fast, light lines."""
        out = np.ones_like(at_mm)
        if not self.spec["skip"]:
            return out
        risk = np.clip(speed - 1.0, 0, None) * np.clip(1.2 - press, 0, None)
        per_point = self.spec["skip"] * risk * np.diff(at_mm, prepend=at_mm[0]) * 4
        for i in np.flatnonzero(self.rng.random(len(at_mm)) < per_point):
            span = self.rng.uniform(0.15, 0.45)
            near = np.abs(at_mm - at_mm[i]) < span / 2
            out[near] *= self.rng.uniform(0.2, 0.55)
        return out

    def _paint(self, p, widths, amount, blobs):
        s = self.SS
        pad = float(widths.max()) + 2
        x0, y0 = int(np.floor(p[:, 0].min() - pad)), int(np.floor(p[:, 1].min() - pad))
        x1, y1 = int(np.ceil(p[:, 0].max() + pad)), int(np.ceil(p[:, 1].max() + pad))
        img = Image.new("L", ((x1 - x0) * s, (y1 - y0) * s), 0)
        d = ImageDraw.Draw(img)
        q = (p - [x0, y0]) * s
        fill = np.clip(np.round(255 * amount / SCALE), 1, 255).astype(int)
        if len(q) == 1:
            r = widths[0] * s / 2
            d.ellipse([q[0, 0] - r, q[0, 1] - r, q[0, 0] + r, q[0, 1] + r], fill=int(fill[0]))
        for i in range(len(q) - 1):
            w = float(widths[i]) * s
            f = int(round((fill[i] + fill[i + 1]) / 2))
            d.line([tuple(q[i]), tuple(q[i + 1])], fill=f, width=max(1, int(round(w))))
            r = w / 2
            d.ellipse([q[i + 1, 0] - r, q[i + 1, 1] - r, q[i + 1, 0] + r, q[i + 1, 1] + r], fill=f)
        for c, r, a in blobs:
            c = (c - [x0, y0]) * s
            r *= s
            # Pooled ink sits a little to one side of the line, not centred on it.
            c = c + self.rng.normal(0, 0.25 * r, 2)
            d.ellipse([c[0] - r, c[1] - 0.8 * r, c[0] + r, c[1] + 0.8 * r], fill=int(min(255, 255 * a * self.pressure / SCALE)))
        self._add(img, x0, y0)

    def _add(self, img, x0, y0):
        s = self.SS
        w, h = img.size[0] // s, img.size[1] // s
        small = np.asarray(img.resize((w, h), Image.BOX), np.float32) * (SCALE / 255.0)
        H, W = self.amount.shape
        a0, b0 = max(0, -y0), max(0, -x0)
        a1, b1 = min(h, H - y0), min(w, W - x0)
        if a1 > a0 and b1 > b0:
            # A stroke crossing another adds its ink: crossings come out darker.
            self.amount[y0 + a0 : y0 + a1, x0 + b0 : x0 + b1] += small[a0:a1, b0:b1]


def _smooth(v, n):
    pad = np.r_[np.full(n // 2, v[0]), v, np.full(n // 2, v[-1])]
    return np.convolve(pad, np.ones(n) / n, "valid")


def _turns(p, s, reach):
    """Indices where the line doubles back sharply, judged over `reach` px each way."""
    out = []
    last = -np.inf
    for i in range(1, len(p) - 1):
        a = np.searchsorted(s, s[i] - reach)
        b = np.searchsorted(s, s[i] + reach)
        if a >= i or b <= i or b >= len(p):
            continue
        u, v = p[i] - p[a], p[b] - p[i]
        n = np.hypot(*u) * np.hypot(*v)
        if n > 0 and u @ v / n < -TURN and s[i] - last > 2 * reach:
            out.append(i)
            last = s[i]
    return out


def _catmull_rom(p, n=4):
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = (np.arange(n) / n)[:, None]
    a, b, c, d = q[:-3, None], q[1:-2, None], q[2:-1, None], q[3:, None]
    out = 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t**2 + (3 * b - a - 3 * c + d) * t**3)
    return np.vstack([out.reshape(-1, 2), p[-1:]])

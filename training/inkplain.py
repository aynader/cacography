"""One plain ink renderer used for both real and generated pen paths.

Real and generated samples must go through identical drawing so that a blind
test judges only the movement, never the renderer.
"""
import numpy as np
from PIL import Image, ImageDraw

SS = 4  # supersampling factor


def _spline(p, n=6):
    """Catmull-Rom through the points, so drawing never shows sampling corners."""
    if len(p) < 3:
        return p
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = (np.arange(n) / n)[:, None]
    a, b, c, d = q[:-3, None], q[1:-2, None], q[2:-1, None], q[3:, None]
    out = 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t**2 + (3 * b - a - 3 * c + d) * t**3)
    return np.vstack([out.reshape(-1, 2), p[-1:]])


def strokes_of(d):
    """Split an (N, 3) x, y, pen-up array into stroke arrays."""
    ends = np.flatnonzero(d[:, 2] > 0.5)
    out, start = [], 0
    for e in ends:
        out.append(d[start : e + 1, :2])
        start = e + 1
    if start < len(d):
        out.append(d[start:, :2])
    return [s for s in out if len(s)]


def render(d, scale=2.0, margin=12, width=1.5, height=None, ink=(28, 34, 92)):
    """Render an (N, 3) path to an RGB image cropped to the writing."""
    xy = d[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    w = int((hi[0] - lo[0]) * scale + 2 * margin)
    h = int((hi[1] - lo[1]) * scale + 2 * margin) if height is None else height
    y_off = margin if height is None else (height - (hi[1] - lo[1]) * scale) / 2
    canvas = Image.new("L", (max(w, 8) * SS, max(h, 8) * SS), 0)
    draw = ImageDraw.Draw(canvas)
    for s in strokes_of(d):
        p = (s - lo) * scale * SS + np.array([margin, y_off]) * SS
        p = _spline(p)
        r = width * SS / 2
        if len(p) == 1:
            x, y = p[0]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            continue
        seg = np.hypot(*np.diff(p, axis=0).T)
        speed = seg / max(np.median(seg[seg > 0]) if np.any(seg > 0) else 1.0, 1e-6)
        kernel = np.ones(7) / 7
        speed = np.convolve(np.r_[speed[0], speed, speed[-1]], kernel, "same")[1:-1]
        wid = np.clip(1.15 - 0.22 * speed, 0.72, 1.12) * width * SS
        for i in range(len(p) - 1):
            ww = float(wid[i])
            draw.line([tuple(p[i]), tuple(p[i + 1])], fill=255, width=max(1, int(round(ww))))
            x, y, rr = p[i + 1][0], p[i + 1][1], ww / 2
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=255)
    alpha = canvas.resize((max(w, 8), max(h, 8)), Image.LANCZOS)
    a = np.asarray(alpha, np.float32)[..., None] / 255
    paper = np.array([248, 247, 241], np.float32)
    rgb = paper * (1 - a) + np.array(ink, np.float32) * a
    return Image.fromarray(rgb.astype(np.uint8))

"""Drawings on a page (TikZ, pgfplots, anything that writes PDF drawing commands), redrawn by hand.

LuaLaTeX records each drawing command with the point on the page where it was written
(cacography.lua). Here they are read back into paths in page points, and page.py traces
every stroked path as one pen stroke, with the slow waver of a hand that draws a line in
one go. Small filled shapes, such as arrow tips and plot marks, are drawn as small marks.

Clipping (`W n`, how pgfplots keeps a plot inside its axes) cuts later paths to the clip
until the matching `Q`. Each path also carries the grey of its stroke and fill, so pale
helper lines (grids) can be drawn lighter and darker fills hatched closer.
"""
from dataclasses import dataclass

import numpy as np

PT_PER_BP = 72.27 / 72


@dataclass
class Path:
    points: np.ndarray  # (N, 2) page pt
    closed: bool
    stroked: bool
    filled: bool
    width: float  # pt
    line: int
    dash: list = None  # pt, or None
    stroke_grey: float = 0.0  # 0 black, 1 white
    fill_grey: float = 0.0

    def __iter__(self):  # the old tuple form, for callers that unpack it
        return iter((self.points, self.closed, self.stroked, self.filled, self.width, self.line, self.dash))


def _bezier(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[1:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _grey(nums, op):
    """The lightness (0 black, 1 white) a colour operator sets."""
    if op in ("g", "G") and nums:
        return float(nums[-1])
    if op in ("rg", "RG") and len(nums) >= 3:
        r, g, b = nums[-3:]
        return float(0.3 * r + 0.59 * g + 0.11 * b)
    if op in ("k", "K") and len(nums) >= 4:
        c, m, y, k = nums[-4:]
        return float((1 - k) * (1 - (0.3 * c + 0.59 * m + 0.11 * y)))
    return None


def _inside(points, poly):
    """Even-odd test of points (N, 2) against a polygon (M, 2)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    a, b = poly[None, :, :], np.roll(poly, -1, axis=0)[None, :, :]
    cross = (a[..., 1] > y) != (b[..., 1] > y)
    xs = a[..., 0] + (y - a[..., 1]) * (b[..., 0] - a[..., 0]) / (b[..., 1] - a[..., 1] + 1e-12)
    return (cross & (x < xs)).sum(1) % 2 == 1


def _split(points, clip, step=0.4):
    """The runs of a polyline that lie inside `clip`, cut where it crosses the edge."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return [points] if _inside(points[:1], clip)[0] else []
    t = np.unique(np.r_[s, np.linspace(0, s[-1], max(2, int(s[-1] / step)))])
    q = np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], 1)
    keep = _inside(q, clip)

    def edge(a, b):
        # Where the line leaves the clip between a (inside) and b (outside), by bisection.
        for _ in range(12):
            m = 0.5 * (a + b)
            if _inside(m[None], clip)[0]:
                a = m
            else:
                b = m
        return a

    runs, start = [], None
    for i, k in enumerate(np.r_[keep, False]):
        if k and start is None:
            start = i
        elif not k and start is not None:
            run = q[start:i]
            if start > 0:
                run = np.vstack([edge(q[start], q[start - 1]), run])
            if i < len(q):
                run = np.vstack([run, edge(q[i - 1], q[i])])
            if len(run) >= 2:
                runs.append(run)
            start = None
    return runs


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman: a polygon cut to a convex clip polygon."""
    out = list(subject)
    area = np.sum(clip[:, 0] * np.roll(clip[:, 1], -1) - np.roll(clip[:, 0], -1) * clip[:, 1])
    sign = 1 if area > 0 else -1
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if np.allclose(a, b):
            continue

        def side(p):
            return sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0

        inp, out = out, []
        for j in range(len(inp)):
            p, q = np.asarray(inp[j - 1]), np.asarray(inp[j])
            if side(q):
                if not side(p):
                    out.append(_meet(p, q, a, b))
                out.append(q)
            elif side(p):
                out.append(_meet(p, q, a, b))
        if not out:
            return None
    return np.array(out)


def _meet(p, q, a, b):
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return q
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    return p + t * d1


def paths(draw):
    """The page's drawing as a list of Path, in the order drawn."""
    out = []
    ctm, saved = np.eye(3), []
    width, dash, array = 0.4, None, []
    stroke_grey, fill_grey = 0.0, 0.0
    clip, pending_clip = None, False
    # pgf writes one command per literal, so a path's state carries from one to the next.
    current, subpaths, start = [], [], None
    for lit in draw:
        ox, oy, line = lit["x"], lit["y"], lit.get("line", 0)

        def page(u, v):
            x, y, _ = ctm @ np.array([u, v, 1.0])
            return np.array([ox + x * PT_PER_BP, oy - y * PT_PER_BP])

        nums = []
        for tok in lit["d"].split():
            try:
                nums.append(float(tok))
                continue
            except ValueError:
                pass
            if tok == "[":
                nums = []
                continue
            if tok == "]":
                array, nums = list(nums), []
                continue
            grey = _grey(nums, tok)
            if grey is not None:
                if tok.isupper():
                    stroke_grey = grey
                else:
                    fill_grey = grey
            elif tok == "q":
                saved.append((ctm.copy(), width, dash, stroke_grey, fill_grey, clip))
            elif tok == "Q":
                ctm, width, dash, stroke_grey, fill_grey, clip = saved.pop() if saved else (np.eye(3), 0.4, None, 0.0, 0.0, None)
            elif tok == "d":
                # "[on off ...] phase d"
                dash = [v * PT_PER_BP for v in array] or None
            elif tok == "cm" and len(nums) >= 6:
                a, b, c, d, e, f = nums[-6:]
                ctm = ctm @ np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])
            elif tok == "w" and nums:
                width = nums[-1] * PT_PER_BP
            elif tok in ("W", "W*"):
                pending_clip = True
            elif tok == "m" and len(nums) >= 2:
                if len(current) > 1:
                    subpaths.append((np.array(current), False))
                current = [page(*nums[-2:])]
                start = current[0]
            elif tok == "l" and len(nums) >= 2 and current:
                current.append(page(*nums[-2:]))
            elif tok in ("c", "v", "y") and current:
                need = {"c": 6, "v": 4, "y": 4}[tok]
                if len(nums) >= need:
                    q = nums[-need:]
                    p0 = current[-1]
                    if tok == "c":
                        p1, p2, p3 = page(q[0], q[1]), page(q[2], q[3]), page(q[4], q[5])
                    elif tok == "v":
                        p1, p2, p3 = p0, page(q[0], q[1]), page(q[2], q[3])
                    else:
                        p1, p2, p3 = page(q[0], q[1]), page(q[2], q[3]), page(q[2], q[3])
                    current.extend(_bezier(p0, p1, p2, p3))
            elif tok == "h" and current:
                if start is not None:
                    current.append(start)
                subpaths.append((np.array(current), True))
                current = []
            elif tok == "re" and len(nums) >= 4:
                x, y, w, h = nums[-4:]
                box = [page(x, y), page(x + w, y), page(x + w, y + h), page(x, y + h), page(x, y)]
                subpaths.append((np.array(box), True))
            elif tok in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"):
                if len(current) > 1:
                    subpaths.append((np.array(current), tok in ("s", "b", "b*")))
                current = []
                if pending_clip and subpaths:
                    # The new clip: this path, inside any clip already set.
                    poly = max((p for p, _ in subpaths), key=len)
                    clip = poly if clip is None else (_clip_polygon(poly, clip) if len(clip) else poly)
                    if clip is None:
                        clip = np.zeros((0, 2))
                    pending_clip = False
                if tok != "n":
                    stroked = tok in ("S", "s", "B", "B*", "b", "b*")
                    filled = tok in ("f", "F", "f*", "B", "B*", "b", "b*")
                    for p, closed in subpaths:
                        out += _clipped(Path(p, closed, stroked, filled, width, line, dash, stroke_grey, fill_grey), clip)
                subpaths = []
            nums = []
    return out


def _clipped(path, clip):
    """The parts of `path` inside the clip: a filled shape cut to it, a line split into runs."""
    if clip is None:
        return [path]
    if len(clip) < 3:
        return []
    if path.filled:
        poly = _clip_polygon(path.points, clip)
        if poly is None or len(poly) < 3:
            return []
        path.points = np.vstack([poly, poly[:1]]) if path.closed else poly
        return [path]
    runs = _split(path.points, clip)
    if len(runs) == 1 and len(runs[0]) >= len(path.points) and np.allclose(runs[0][[0, -1]], path.points[[0, -1]]):
        return [path]  # wholly inside: keep its own points
    return [Path(r, False, path.stroked, False, path.width, path.line, path.dash, path.stroke_grey, path.fill_grey) for r in runs]

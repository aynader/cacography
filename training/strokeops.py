"""Stroke-level operations shared by the BRUSH and IAM-OnDB preparation.

Late marks: people often finish a word, then go back to dot an i or cross a t. Only
the finished ink is visible, so such a stroke can move to just after the letters it
belongs to while keeping its exact position. The model then learns the displaced dots
and long bars of real writing, and its character pointer only ever moves forward.
"""
import numpy as np


def stroke_ranges(pen):
    """(start, end) index pairs, end exclusive, from an (N) pen-up flag array."""
    ends = np.flatnonzero(pen > 0.5)
    out, start = [], 0
    for e in ends:
        out.append((start, int(e) + 1))
        start = int(e) + 1
    if start < len(pen):
        out.append((start, len(pen)))
    return out


def reorder_by_labels(points, labels):
    """Move strokes that return to an earlier character. Returns points, labels, moved."""
    ranges = stroke_ranges(points[:, 2])
    placed = []  # (range, label array)
    reached = -1
    moved = 0
    for start, end in ranges:
        lab = labels[start:end]
        if lab.max() < reached:
            target = lab.max()
            k = next(i for i, (_, l) in enumerate(placed) if l.max() >= target)
            # Keep several marks for one spot in the order they were written.
            while k + 1 < len(placed) and placed[k + 1][0][2]:
                k += 1
            top = placed[k][1].max()
            placed.insert(k + 1, ((start, end, True), np.full(end - start, top, dtype=lab.dtype)))
            moved += 1
        else:
            placed.append(((start, end, False), lab))
            reached = max(reached, int(lab.max()))
    pts = np.concatenate([points[r[0] : r[1]] for r, _ in placed])
    lab = np.concatenate([l for _, l in placed])
    pts = pts.copy()
    cursor = 0
    for r, _ in placed:
        cursor += r[1] - r[0]
        pts[cursor - 1, 2] = 1.0
    return pts, np.maximum.accumulate(lab), moved


def reorder_by_geometry(strokes, unit):
    """Same idea without labels: find small strokes written back over earlier ink.

    strokes is a list of (N, 2) arrays in time order and unit is the rough x-height.
    Long returns are left alone, because they tend to be corrections of whole words
    and the alignment quality check handles those.
    """
    placed = []
    xmax = -np.inf
    moved = 0
    for s in strokes:
        lo, hi = s[:, 0].min(), s[:, 0].max()
        height = np.ptp(s[:, 1])
        center = 0.5 * (lo + hi)
        small = hi - lo < 2.5 * unit and height < 1.3 * unit
        back = center < xmax - 0.4 * unit and xmax - center < 12 * unit
        if placed and small and back:
            k = None
            for i in range(len(placed) - 1, -1, -1):
                if placed[i][0] and placed[i][1][:, 0].min() <= center <= placed[i][1][:, 0].max():
                    k = i
                    break
            if k is None:
                k = int(np.argmin([abs(0.5 * (p[1][:, 0].min() + p[1][:, 0].max()) - center) for p in placed]))
            while k + 1 < len(placed) and not placed[k + 1][0]:
                k += 1
            placed.insert(k + 1, (False, s))
            moved += 1
        else:
            placed.append((True, s))
            xmax = max(xmax, hi)
    return [p[1] for p in placed], moved


def to_points(strokes):
    """Join strokes into an (N, 3) x, y, pen-up array."""
    out = []
    for s in strokes:
        pen = np.zeros((len(s), 1), dtype=np.float32)
        pen[-1] = 1
        out.append(np.concatenate([s.astype(np.float32), pen], 1))
    return np.concatenate(out)

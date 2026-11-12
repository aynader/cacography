"""Turn MathWriting's isolated symbols into the bank the renderer draws from.

Devices differ, so every sample is resampled to a steady pen speed relative to its size,
then centred and scaled so its larger side is 1. Samples far from the usual shape of their
symbol (a slip, a misplaced label) are left out: the renderer picks one sample per symbol
per document, and it should not land on a bad one. Saved under models/, which stays out
of version control because the data allows non-commercial research use only.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_iam import clean_stroke  # noqa: E402

STEP = 0.035  # distance between points as a share of the symbol's larger side
GRID = 24
KEEP = 0.6  # share of each symbol's samples kept, nearest its usual shape first
MIN_KEEP = 8


def to_strokes(ink):
    """Split (N, 4) x, y, t, pen-up rows into strokes."""
    out, start = [], 0
    for e in np.flatnonzero(ink[:, 3] > 0.5):
        out.append(ink[start : e + 1, :3].astype(np.float64))
        start = e + 1
    return out


def normalize(strokes):
    """Strokes at a steady speed, centred, with the larger side 1. None if unusable."""
    length = duration = 0.0
    for s in strokes:
        d = np.diff(s, axis=0)
        ok = d[:, 2] > 0
        length += np.hypot(d[ok, 0], d[ok, 1]).sum()
        duration += d[ok, 2].sum()
    xy = np.concatenate([s[:, :2] for s in strokes])
    size = np.ptp(xy, axis=0).max()
    if size <= 0 or duration <= 0 or length <= 0:
        return None
    dt = STEP * size / (length / duration)
    cleaned = [clean_stroke(s, dt) for s in strokes]
    xy = np.concatenate(cleaned)
    lo, hi = xy.min(0), xy.max(0)
    size = (hi - lo).max()
    if size <= 0:
        return None
    return [(c - 0.5 * (lo + hi)) / size for c in cleaned]


def raster(strokes):
    """A small picture of the ink's shape, for comparing samples of one symbol."""
    pic = np.zeros((GRID, GRID), np.float32)
    for s in strokes:
        if len(s) > 1:
            t = np.linspace(0, len(s) - 1, 3 * len(s))
            p = np.stack([np.interp(t, np.arange(len(s)), s[:, k]) for k in (0, 1)], 1)
        else:
            p = s
        cell = np.clip(((p + 0.5) * (GRID - 1)).round().astype(int), 0, GRID - 1)
        pic[cell[:, 1], cell[:, 0]] = 1.0
    return pic


def slant(strokes):
    """Horizontal drift per unit of downward travel, over pen-down movement heading down."""
    dx = dy = 0.0
    for s in strokes:
        d = np.diff(s, axis=0)
        down = (d[:, 1] > 0) & (np.abs(d[:, 1]) > np.abs(d[:, 0]))
        dx += d[down, 0].sum()
        dy += d[down, 1].sum()
    return dx, dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/cache/symbols.npz"))
    ap.add_argument("--out", type=Path, default=Path("../models/symbols"))
    args = ap.parse_args()
    z = np.load(args.raw)
    labels = json.loads(args.raw.with_suffix(".json").read_text(encoding="utf-8"))["labels"]
    off, pts = z["offsets"], z["points"]
    by_label = defaultdict(list)
    skipped = 0
    for i, label in enumerate(labels):
        made = normalize(to_strokes(pts[off[i] : off[i + 1]]))
        if made is None:
            skipped += 1
        else:
            by_label[label].append(made)
    points, offsets, names = [], [0], []
    drift = np.zeros(2)
    for label in sorted(by_label):
        samples = by_label[label]
        pics = np.stack([raster(s) for s in samples])
        mean = pics.mean(0)
        order = np.argsort(((pics - mean) ** 2).sum((1, 2)))
        keep = max(min(MIN_KEEP, len(samples)), int(len(samples) * KEEP))
        for k in order[:keep]:
            rows = []
            for s in samples[k]:
                pen = np.zeros((len(s), 1))
                pen[-1] = 1
                rows.append(np.concatenate([s, pen], 1))
                drift += slant([s])
            ink = np.concatenate(rows).astype(np.float32)
            points.append(ink)
            offsets.append(offsets[-1] + len(ink))
            names.append(label)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "bank.npz", points=np.concatenate(points), offsets=np.array(offsets, np.int64))
    mean_slant = float(drift[0] / drift[1])
    (args.out / "bank.json").write_text(json.dumps(dict(labels=names, mean_slant=mean_slant)), encoding="utf-8")
    counts = np.array([names.count(k) for k in by_label])
    print(f"kept {len(names)} of {len(labels)} samples, {len(by_label)} symbols, skipped {skipped}")
    print(f"samples per symbol pct 10/50/90 {np.percentile(counts, [10, 50, 90]).astype(int)}  points per sample median {int(np.median(np.diff(offsets)))}")
    print(f"mean slant {mean_slant:.3f}")


if __name__ == "__main__":
    main()

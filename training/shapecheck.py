"""A shape check that catches scribbles the checker still manages to read.

Real words, however messy, spread their ink along the line. A scribble piles ink onto a
small area or writes over the same spot. Three measures, with limits taken from real
writing, flag words outside what real people produce.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

COLUMN = 2.0  # model units, about half an x-height


def measures(steps, n_chars):
    s = steps.detach().cpu().numpy() if torch.is_tensor(steps) else np.asarray(steps)
    xy = np.cumsum(s[:, :2], 0)
    width = max(float(np.ptp(xy[:, 0])), 1e-3)
    down = s[:-1, 2] < 0.5
    seg = np.hypot(*np.diff(xy, axis=0).T) * down
    length = float(seg.sum())
    mid = 0.5 * (xy[1:, 0] + xy[:-1, 0])
    bins = np.floor((mid - xy[:, 0].min()) / COLUMN).astype(int)
    per_col = np.bincount(bins, weights=seg, minlength=bins.max() + 1 if len(bins) else 1)
    used = per_col[per_col > 0]
    peak = float(used.max() / np.median(used)) if len(used) else 0.0
    return {"ink_per_width": length / width, "width_per_char": width / max(n_chars, 1), "peak_column": peak}


def fit_limits(lines, low=1.0, high=99.0):
    from handdata import VOCAB

    rows = [measures(l.steps, len(l.text)) for l in lines]
    limits = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        limits[k] = [float(np.percentile(v, low)), float(np.percentile(v, high))]
    return limits


def passes(steps, n_chars, limits):
    m = measures(steps, n_chars)
    return all(limits[k][0] <= m[k] <= limits[k][1] for k in limits)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wordsdata import load_splits

    data = load_splits(Path("data/cache/words.npz"), Path("writers.json"))
    rng = np.random.default_rng(0)
    pick = [data.train[i] for i in rng.choice(len(data.train), 20000, replace=False)]
    limits = fit_limits(pick)
    Path("runs/shape_limits.json").write_text(json.dumps(limits, indent=1), encoding="utf-8")
    test = data.test_unseen_words
    rate = np.mean([passes(l.steps, len(l.text), limits) for l in test])
    print("limits", {k: [round(a, 3), round(b, 3)] for k, (a, b) in limits.items()})
    print(f"real unseen words passing the shape check: {rate:.3f}")

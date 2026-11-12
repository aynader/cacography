"""Writing with the trained model: words, the check-and-retry step, and simple lines."""
import numpy as np
import torch

from handdata import SCALE, encode
from handmodel import sample
from shapecheck import passes as shape_passes
from train_reader import edit_distance, greedy_decode, out_lengths, pad


def pad_texts(texts, device):
    ids = [encode(t) for t in texts]
    out = torch.zeros(len(ids), max(len(t) for t in ids), dtype=torch.long, device=device)
    for i, t in enumerate(ids):
        out[i, : len(t)] = t.to(device)
    return out


def tidy_word(steps, tiny=2, margin=0.5):
    """Drop a tiny mark at either end of a word that lies outside the word's ink.

    The model leaves such stray dots about 5 times as often as real writers do, and
    on a page they read as specks between words.
    """
    s = steps.detach()
    xy = torch.cumsum(s[:, :2], 0)
    ends = torch.nonzero(s[:, 2] > 0.5)[:, 0].tolist()
    if not ends or ends[-1] != len(s) - 1:
        ends.append(len(s) - 1)
    bounds, start = [], 0
    for e in ends:
        bounds.append((start, e + 1))
        start = e + 1
    big = [(a, b) for a, b in bounds if b - a > tiny]
    if not big or len(bounds) < 2:
        return steps
    lo = min(float(xy[a:b, 0].min()) for a, b in big)
    hi = max(float(xy[a:b, 0].max()) for a, b in big)

    def loose(a, b):
        return b - a <= tiny and (float(xy[a:b, 0].max()) < lo - margin or float(xy[a:b, 0].min()) > hi + margin)

    keep = list(bounds)
    if loose(*keep[-1]):
        keep.pop()
    if len(keep) > 1 and loose(*keep[0]):
        keep.pop(0)
    if len(keep) == len(bounds):
        return steps
    a0, b0 = keep[0][0], keep[-1][1]
    out = s[a0:b0].clone()
    start_xy = xy[a0]
    out[0, :2] = start_xy - torch.stack([start_xy[0], torch.zeros_like(start_xy[1])])  # word still starts at x = 0
    out[-1, 2] = 1.0
    return out


def read_back(checker, words, device, size=256):
    """Edit distances are computed by the caller; this returns decoded id lists."""
    out = []
    with torch.no_grad():
        for i in range(0, len(words), size):
            x, n = pad([w.cpu() for w in words[i : i + size]], device)
            out.extend(greedy_decode(checker(x), out_lengths(n)))
    return out


def allowed_errors(text, tolerance):
    return int(len(text) * tolerance)


SHAPE_FAIL = 1000  # ranks any shape-check failure below every readable attempt


def write_checked(model, checker, texts, style, bias=0.5, tolerance=0.0, tries=8, generator=None, limits=None):
    """Write each word, re-drawing any the checker cannot read as the intended text.

    With shape limits (see shapecheck.py), scribbles the checker can still read are
    re-drawn too. style is (N, style_dim), one per word. Returns the step tensors, how
    many tries each word took, and whether it finally passed. A word that never passes
    keeps its best attempt, so callers can decide what to do with it.
    """
    device = style.device
    n = len(texts)
    best = [None] * n
    best_d = [10**9] * n
    used = [0] * n
    todo = list(range(n))
    for _ in range(tries):
        if not todo:
            break
        ids = pad_texts([texts[i] for i in todo], device)
        # Real words never need more than about 45 points per letter (see build_words.py).
        limit = 45 * max(len(texts[i]) for i in todo) + 25
        made = [tidy_word(w) for w in sample(model, ids, style[todo], bias=bias, generator=generator, max_steps=limit)]
        decoded = read_back(checker, made, device) if checker is not None else [encode(texts[i]).tolist() for i in todo]
        still = []
        for i, word, seq in zip(todo, made, decoded):
            used[i] += 1
            d = edit_distance(seq, encode(texts[i]).tolist())
            if limits is not None and not shape_passes(word, len(texts[i]), limits):
                d += SHAPE_FAIL
            if d < best_d[i]:
                best_d[i], best[i] = d, word
            if d > allowed_errors(texts[i], tolerance):
                still.append(i)
        todo = still
    passed = [best_d[i] <= allowed_errors(texts[i], tolerance) for i in range(n)]
    return best, used, passed


def steps_to_points(steps):
    s = steps.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([np.cumsum(s[:, :2] * SCALE, 0), s[:, 2:3]], 1)


def compose_line(words, gap=(14.0, 4.0), rng=None):
    """Lay words along one baseline. gap is (median, spread) in working pixels."""
    rng = rng or np.random.default_rng(0)
    parts, x = [], 0.0
    for w in words:
        p = steps_to_points(w)
        p[-1, 2] = 1.0  # the pen always leaves the paper between words
        p[:, 0] += x - p[:, 0].min()
        parts.append(p)
        x = p[:, 0].max() + max(2.0, rng.normal(*gap))
    return np.concatenate(parts)

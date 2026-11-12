"""Label each IAM-OnDB pen point with its character by forced alignment.

The trained reader scores every character at every frame. A Viterbi pass finds the most
likely way to walk through the known text in order. Lines the reader finds implausible
are dropped later using the confidence saved here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import encode  # noqa: E402
from train_align import load_steps  # noqa: E402
from train_reader import Reader, out_lengths, pad  # noqa: E402


def force_align(logp, target):
    """logp (T, C) log-probabilities, target list of ids. Returns (T) char index or -1, score."""
    T, L = logp.shape[0], len(target)
    if T < L:
        return None, -np.inf
    tgt = np.array(target)
    same = np.r_[False, tgt[1:] == tgt[:-1]]  # a repeated letter needs a blank between
    INF = -1e18
    char = np.full(L + 1, INF)  # best score ending inside character j (1-based)
    blank = np.full(L + 1, INF)  # best score in a blank after j characters
    blank[0] = 0.0
    back_c = np.zeros((T, L + 1), np.int8)  # 0 stay, 1 from previous char, 2 from previous blank
    back_b = np.zeros((T, L + 1), np.int8)  # 0 from blank, 1 from char
    for t in range(T):
        stay = char[1:]
        from_char = char[:-1].copy()
        from_char[same] = INF
        from_blank = blank[:-1]
        options = np.stack([stay, from_char, from_blank])
        pick = options.argmax(0)
        new_char = np.full(L + 1, INF)
        new_char[1:] = options[pick, np.arange(L)] + logp[t, tgt]
        back_c[t, 1:] = pick
        b_opts = np.stack([blank, char])
        b_pick = b_opts.argmax(0)
        new_blank = b_opts[b_pick, np.arange(L + 1)] + logp[t, 0]
        back_b[t] = b_pick
        char, blank = new_char, new_blank
    in_char = char[L] >= blank[L]
    score = max(char[L], blank[L])
    if not np.isfinite(score) or score < -1e17:
        return None, -np.inf
    j, out = L, np.full(T, -1, np.int64)
    for t in range(T - 1, -1, -1):
        if in_char:
            out[t] = j - 1
            p = back_c[t, j]
            if p == 1:
                j -= 1
            elif p == 2:
                j -= 1
                in_char = False
        else:
            if back_b[t, j] == 1:
                in_char = True
    return out, score / T


def to_point_labels(frames, n_points):
    """Upsample frame labels to pen points, filling blanks from the character before."""
    labels = np.repeat(frames, 2)[:n_points]
    if len(labels) < n_points:
        labels = np.r_[labels, np.full(n_points - len(labels), labels[-1])]
    seen = np.flatnonzero(labels >= 0)
    if not len(seen):
        return None
    labels[: seen[0]] = labels[seen[0]]
    return np.maximum.accumulate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/iam.npz"))
    ap.add_argument("--reader", type=Path, default=Path("runs/align/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/iam_labels.npz"))
    ap.add_argument("--sheet", type=Path, default=Path("runs/inspect/align.png"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    z = np.load(args.cache)
    meta = json.loads(args.cache.with_suffix(".json").read_text(encoding="utf-8"))
    sentences, off, pts = meta["sentences"], z["offsets"], z["points"]
    reader = Reader().to(device).eval()
    reader.load_state_dict(torch.load(args.reader, map_location=device, weights_only=True)["model"])
    lines = load_steps(args.cache)
    assert len(lines) == len(sentences), "every packed IAM line should be in the reader's vocabulary"
    labels_all = np.full(len(pts), -1, np.int16)
    scores = np.full(len(lines), -np.inf, np.float32)
    order = sorted(range(len(lines)), key=lambda i: len(lines[i][0]))
    with torch.no_grad():
        for start in range(0, len(order), 64):
            idx = order[start : start + 64]
            x, n = pad([lines[i][0] for i in idx], device)
            logp = F.log_softmax(reader(x), -1).cpu().numpy()
            for row, i, nf in zip(logp, idx, out_lengths(n).tolist()):
                frames, score = force_align(row[:nf], lines[i][1].tolist())
                if frames is None:
                    continue
                lab = to_point_labels(frames, off[i + 1] - off[i])
                if lab is None:
                    continue
                labels_all[off[i] : off[i + 1]] = lab
                scores[i] = score
    np.savez(args.out, labels=labels_all, scores=scores)
    good = scores[np.isfinite(scores)]
    print(f"aligned {len(good)}/{len(lines)}  score pct 5/25/50/75/95 {np.percentile(good, [5, 25, 50, 75, 95]).round(3)}")

    # Picture: colour every character differently, for lines across the confidence range.
    palette = [(200, 30, 30), (30, 90, 210), (20, 150, 60), (210, 130, 0), (130, 40, 180), (0, 150, 160)]
    ranked = np.argsort(np.where(np.isfinite(scores), scores, np.inf))
    n_finite = len(good)
    picks = [ranked[int(n_finite * q)] for q in (0.02, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.98)]
    H = 110
    img = Image.new("RGB", (1500, H * len(picks)), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    for r, i in enumerate(picks):
        d = pts[off[i] : off[i + 1]]
        lab = labels_all[off[i] : off[i + 1]]
        k = min(1480 / max(np.ptp(d[:, 0]), 1), 2.0)
        x = (d[:, 0] - d[:, 0].min()) * k + 8
        y = (d[:, 1] - d[:, 1].min()) * k + r * H + 16
        dr.text((8, r * H + 1), f"score {scores[i]:.3f}: {sentences[i]}", fill=(90, 90, 90))
        for t in range(len(d) - 1):
            if d[t, 2] < 0.5:
                dr.line([(x[t], y[t]), (x[t + 1], y[t + 1])], fill=palette[int(lab[t]) % len(palette)], width=2)
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.sheet)
    print("saved", args.sheet)


if __name__ == "__main__":
    main()

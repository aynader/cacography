"""Cut every line into words measured from the baseline, at one common size.

TeX decides where words go and how large they are, so the model only has to write one
word sitting on a baseline. Each line is scaled so its x-height is the same, and the
writer's natural size is kept separately for the page layer to bring back later.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import CHAR_TO_ID, SCALE, fix_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cacography_render.marks import classify, rebuild, strokes  # noqa: E402

XH_TARGET = 16.0  # x-height in the working pixel units
XH_CHARS = set("acemnorsuvwxz")
SNAP = 8  # how many points a word boundary may move to reach a pen lift
# IAM's transcripts space out punctuation ("Mr . Gaitskell"); writers don't.
CLOSING = re.compile(r"[.,;:!?)]+")


def metrics(points, labels, sentence):
    """Baseline and x-height from the letters that sit between those two lines."""
    bottoms, heights = [], []
    for i, c in enumerate(sentence):
        if c in XH_CHARS:
            y = points[labels == i, 1]
            if len(y) >= 3:
                bottoms.append(y.max())
                heights.append(y.max() - y.min())
    if len(heights) < 3:
        return None, None
    return float(np.median(bottoms)), float(np.median(heights))


def split_words(points, labels, sentence):
    """Yield (character indices, text, start, end) with each word starting after a pen lift."""
    pen = points[:, 2] > 0.5
    spans, cur = [], []
    for i, c in enumerate(sentence):
        if c == " ":
            if cur:
                spans.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        spans.append(cur)
    joined = []
    for span in spans:
        if joined and CLOSING.fullmatch("".join(sentence[i] for i in span)):
            joined[-1] = joined[-1] + span
        else:
            joined.append(span)
    spans = joined
    starts = []
    for span in spans:
        where = np.flatnonzero(labels >= span[0])
        starts.append(int(where[0]) if len(where) else None)
    good = [True] * len(spans)
    for k in range(1, len(spans)):
        if starts[k] is None:
            good[k] = False
            continue
        s = starts[k]
        lifts = [t + 1 for t in range(max(0, s - SNAP - 1), min(len(pen) - 1, s + SNAP)) if pen[t]]
        if not lifts:
            good[k - 1] = good[k] = False  # the writer joined these words, or alignment is off
            continue
        starts[k] = min(lifts, key=lambda t: abs(t - s))
    if starts[0] is None:
        return
    starts[0] = 0
    ends = [starts[k + 1] if k + 1 < len(starts) and starts[k + 1] is not None else len(points) for k in range(len(starts))]
    for k, span in enumerate(spans):
        if good[k] and starts[k] is not None and ends[k] - starts[k] >= 2:
            yield span, "".join(sentence[i] for i in span), starts[k], ends[k]


def clean_word(steps, labels, text):
    """Drop marks the text can't explain and give punctuation its own ink.

    Forced alignment tends to hand a period's dot to the letter before it and leave the
    period itself with a stub, so the model would learn that "." writes nothing and that
    letters sometimes end in a dot. Marks past the word's ends go to its punctuation, and
    move to the end of the word so the character pointer only runs forward. Returns
    steps, labels and what happened, or None if the word's labels can't be trusted.
    """
    marks = classify(steps, text)
    kinds = {m[0]: m[3] for m in marks}
    spans = strokes(steps)
    xy = np.cumsum(steps[:, :2], 0)
    a = 0
    while a < len(text) and not text[a].isalnum():
        a += 1
    b = len(text)
    while b > a and not text[b - 1].isalnum():
        b -= 1
    if a == b:
        return steps, labels, "punctuation only"  # a practice string: nothing to separate

    def by_x(kind):
        return sorted((i for i, k in kinds.items() if k == kind), key=lambda i: xy[spans[i][0] : spans[i][1], 0].mean())

    lead, trail = by_x("lead"), by_x("trail")
    middle = [i for i in range(len(spans)) if kinds.get(i) not in ("lead", "trail", "stray")]
    if not middle:
        return None
    if text.endswith(".") and not trail:
        return None  # no dot where the text has one: the dot went to a neighbour, or never was

    def fill(group, first, last):
        """Hand marks to punctuation characters in order, as many marks as each one takes."""
        out, c = [], first
        room = {"\"": 2, ":": 2, ";": 2}
        left = room.get(text[c], 1) if c < last else 0
        for i in group:
            n = spans[i][1] - spans[i][0]
            out.append(np.full(n, min(c, last - 1), dtype=labels.dtype))
            left -= 1
            if left <= 0 and c + 1 < last:
                c += 1
                left = room.get(text[c], 1)
        return out

    lead_labels = fill(lead, 0, a)
    trail_labels = fill(trail, b, len(text))
    parts, new_labels = [], []
    for i, lab in zip(lead, lead_labels):
        parts.append(i)
        new_labels.append(lab)
    low = a if lead else 0
    high = b - 1 if trail else len(text) - 1
    for i in middle:
        parts.append(i)
        new_labels.append(np.clip(labels[spans[i][0] : spans[i][1]], low, high))
    for i, lab in zip(trail, trail_labels):
        parts.append(i)
        new_labels.append(lab)
    order = np.concatenate([np.arange(*spans[i]) for i in parts])
    out_labels = np.maximum.accumulate(np.concatenate(new_labels))
    dropped = sum(1 for k in kinds.values() if k == "stray")
    changed = "dropped stray" if dropped else ("punctuation labelled" if (lead or trail) else "unchanged")
    return rebuild(xy[order], steps[order, 2]), out_labels, changed


def load_source(cache: Path, labels_path: Path | None, min_score: float | None):
    z = np.load(cache)
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    off, pts = z["offsets"], z["points"]
    if labels_path is None:
        labels, scores = z["labels"], None
    else:
        lz = np.load(labels_path)
        labels, scores = lz["labels"], lz["scores"]
    for i, sentence in enumerate(meta["sentences"]):
        if scores is not None and (not np.isfinite(scores[i]) or scores[i] < min_score):
            continue
        if any(c not in CHAR_TO_ID for c in sentence):
            continue
        lab = labels[off[i] : off[i + 1]]
        if lab.min() < 0:
            continue
        yield int(z["writers"][i]), sentence, pts[off[i] : off[i + 1]], fix_labels(lab.astype(np.int64), sentence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brush", type=Path, default=Path("data/cache/brush2.npz"))
    ap.add_argument("--iam", type=Path, default=Path("data/cache/iam.npz"))
    ap.add_argument("--iam-labels", type=Path, default=Path("data/cache/iam_labels.npz"))
    ap.add_argument("--min-score", type=float, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/cache/words.npz"))
    args = ap.parse_args()

    lines = [(0, *l) for l in load_source(args.brush, None, None)]
    brush_n = len(lines)
    lines += [(1, *l) for l in load_source(args.iam, args.iam_labels, args.min_score)]
    print(f"lines: brush {brush_n}, iam kept {len(lines) - brush_n}", flush=True)

    per_writer = defaultdict(list)
    measured = []
    for source, writer, sentence, pts, lab in lines:
        base, xh = metrics(pts, lab, sentence)
        measured.append((base, xh))
        if xh:
            per_writer[writer].append(xh)
    writer_xh = {w: float(np.median(v)) for w, v in per_writer.items()}

    steps_out, labels_out, offsets, writers_out, sources, texts = [], [], [0], [], [], []
    rejected, cleaning = Counter(), Counter()
    gaps = defaultdict(list)
    for (source, writer, sentence, pts, lab), (base, xh) in zip(lines, measured):
        typical = writer_xh.get(writer)
        if typical is None:
            rejected["writer_no_size"] += 1
            continue
        if xh is None or not 0.55 < xh / typical < 1.7:
            xh, base = typical, float(np.percentile(pts[:, 1], 85))
        k = XH_TARGET / xh
        previous_end, previous_right = None, None
        for span, text, start, end in split_words(pts, lab, sentence):
            seg = pts[start:end]
            if previous_end == start:
                gap = (seg[:, 0].min() - previous_right) * k
                if -0.5 * XH_TARGET < gap < 6 * XH_TARGET:
                    gaps[writer].append(float(gap))
            previous_end, previous_right = end, seg[:, 0].max()
            T, chars = len(seg), len(text)
            if not 2 * chars <= T <= 45 * chars + 25:
                rejected["points_per_char"] += 1
                continue
            xy = (seg[:, :2] - [seg[0, 0], base]) * k
            if np.ptp(xy[:, 0]) > 6 * XH_TARGET * chars + 3 * XH_TARGET or np.abs(xy[:, 1]).max() > 8 * XH_TARGET:
                rejected["shape"] += 1
                continue
            delta = np.zeros((T, 3), np.float32)
            delta[1:, :2] = np.diff(xy, axis=0) / SCALE
            delta[0, 1] = xy[0, 1] / SCALE
            delta[:, 2] = seg[:, 2]
            delta[-1, 2] = 1.0
            # Positions within this word; a space a punctuation token was joined across goes too.
            rel = np.clip(np.searchsorted(np.array(span), lab[start:end]), 0, chars - 1)
            rel = np.maximum.accumulate(rel)
            cleaned = clean_word(delta.astype(np.float64), rel, text)
            if cleaned is None:
                rejected["marks_dont_fit_text"] += 1
                continue
            delta, rel, outcome = cleaned
            cleaning[outcome] += 1
            delta = delta.astype(np.float32)
            if len(set(rel.tolist())) < chars:
                rejected["char_without_points"] += 1
                continue
            T = len(delta)
            steps_out.append(delta)
            labels_out.append(rel.astype(np.int16))
            offsets.append(offsets[-1] + T)
            writers_out.append(writer)
            sources.append(source)
            texts.append(text)
    np.savez(
        args.out,
        steps=np.concatenate(steps_out),
        labels=np.concatenate(labels_out),
        offsets=np.array(offsets, dtype=np.int64),
        writers=np.array(writers_out, dtype=np.int32),
        sources=np.array(sources, dtype=np.int8),
    )
    writer_gap = {str(w): [float(np.median(v)), float(np.std(v))] for w, v in gaps.items() if len(v) >= 5}
    args.out.with_suffix(".json").write_text(
        json.dumps({"texts": texts, "writer_xh": {str(k): v for k, v in writer_xh.items()}, "writer_gap": writer_gap}),
        encoding="utf-8",
    )
    src = np.array(sources)
    lens = np.diff(offsets)
    print(f"words {len(texts)}  brush {(src == 0).sum()}  iam {(src == 1).sum()}  writers {len(set(writers_out))}  distinct words {len(set(texts))}")
    print(f"cleaning {dict(cleaning)}")
    print(f"rejected {dict(rejected)}  points/word pct 5/50/95 {np.percentile(lens, [5, 50, 95]).astype(int)}")


if __name__ == "__main__":
    main()

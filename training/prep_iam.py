"""Pack IAM-OnDB lines into arrays comparable with BRUSH.

The whiteboard pen samples unevenly and has occasional glitches, so each stroke is
cleaned of spikes, resampled to a steady time step and lightly smoothed. Lines are
levelled, scaled to BRUSH's size, and any late dots and bars are moved next to their
letters. Files come straight from the archives; nothing is extracted to disk.
"""
import argparse
import json
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import CHAR_TO_ID  # noqa: E402
from strokeops import reorder_by_geometry, to_points  # noqa: E402

TARGET_SPAN = 38.0  # BRUSH median 5th-95th percentile line height, measured
MAX_POINTS = 900


def writer_map(path: Path):
    out = {}
    for line in path.read_text("latin1").splitlines():
        parts = line.split()
        if not line.startswith("#") and len(parts) >= 2:
            out[parts[0]] = int(parts[1])
    return out


def transcriptions(path: Path):
    """Line text per form id, joining the tokenizer's split contractions."""
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            raw = tf.extractfile(m).read().decode("latin1")
            if "CSR:" not in raw:
                continue
            lines = [l.strip() for l in raw.split("CSR:", 1)[1].strip().splitlines() if l.strip()]
            lines = [re.sub(r"\s+", " ", re.sub(r" '(?=[a-z])", "'", l)) for l in lines]
            out[Path(m.name).stem] = lines
    return out


def clean_stroke(xyt, dt):
    """Remove spikes, resample in time, smooth. xyt is (N, 3). Returns (M, 2)."""
    if len(xyt) >= 5:
        step = np.hypot(*np.diff(xyt[:, :2], axis=0).T)
        limit = 8 * max(np.median(step), 1.0) + 20
        keep = np.ones(len(xyt), bool)
        for i in range(1, len(xyt) - 1):
            # A spike leaves and returns in one sample: both steps long, neighbours close.
            if step[i - 1] > limit and step[i] > limit and np.hypot(*(xyt[i + 1, :2] - xyt[i - 1, :2])) < limit:
                keep[i] = False
        xyt = xyt[keep]
    t = xyt[:, 2] - xyt[0, 2]
    order = np.flatnonzero(np.diff(t, prepend=-1.0) > 0)
    xyt, t = xyt[order], t[order]
    if len(xyt) < 2 or t[-1] <= 0:
        return xyt[:, :2].astype(np.float64)
    grid = np.arange(0, t[-1] + dt * 0.5, dt)
    xy = np.stack([np.interp(grid, t, xyt[:, 0]), np.interp(grid, t, xyt[:, 1])], 1)
    if len(xy) >= 3:
        xy[1:-1] = 0.25 * xy[:-2] + 0.5 * xy[1:-1] + 0.25 * xy[2:]
    return xy


def parse_line(raw: bytes):
    if b"<!DOCTYPE" in raw[:400]:
        raise ValueError("doctype not allowed")
    root = ET.fromstring(raw)
    return [
        np.array([[float(p.get("x")), float(p.get("y")), float(p.get("time"))] for p in s.iter("Point")])
        for s in root.iter("Stroke")
    ]


def level(strokes):
    """Remove the line's overall tilt, measured through the stroke bottoms."""
    bottoms = np.array([[0.5 * (s[:, 0].min() + s[:, 0].max()), s[:, 1].max()] for s in strokes if len(s) > 3])
    if len(bottoms) < 4:
        return strokes
    a, _ = np.polyfit(bottoms[:, 0], bottoms[:, 1], 1)
    resid = bottoms[:, 1] - np.polyval(np.polyfit(bottoms[:, 0], bottoms[:, 1], 1), bottoms[:, 0])
    good = np.abs(resid) < 1.5 * resid.std() + 1e-6
    if good.sum() >= 4:
        a, _ = np.polyfit(bottoms[good, 0], bottoms[good, 1], 1)
    a = float(np.clip(a, -0.15, 0.15))
    return [np.stack([s[:, 0], s[:, 1] - a * s[:, 0]], 1) for s in strokes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dt", type=float, default=0.024)  # matches BRUSH pen step relative to line height, measured
    args = ap.parse_args()
    writers = writer_map(args.root / "forms.txt")
    texts = transcriptions(args.root / "ascii-all.tar.gz")
    print("forms with writer", len(writers), "forms with text", len(texts), flush=True)
    points, offsets, sentences, line_writers, ids = [], [0], [], [], []
    dropped, moved_total, moved_lines = Counter(), 0, 0
    steps = []
    with tarfile.open(args.root / "lineStrokes-all.tar.gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".xml"):
                continue
            name = Path(m.name).stem
            form, num = name.rsplit("-", 1)
            if form not in writers:
                dropped["no_writer"] += 1
                continue
            lines = texts.get(form)
            if lines is None or int(num) > len(lines):
                dropped["no_text"] += 1
                continue
            sentence = lines[int(num) - 1]
            if any(c not in CHAR_TO_ID for c in sentence):
                dropped["char_outside_vocab"] += 1
                continue
            try:
                raw = [s for s in parse_line(tf.extractfile(m).read()) if len(s)]
            except Exception:  # noqa: BLE001
                dropped["bad_xml"] += 1
                continue
            if len(raw) < 3:
                dropped["too_few_strokes"] += 1
                continue
            strokes = [clean_stroke(s, args.dt) for s in raw]
            strokes = level(strokes)
            all_y = np.concatenate([s[:, 1] for s in strokes])
            span = np.percentile(all_y, 95) - np.percentile(all_y, 5)
            if span <= 0:
                dropped["flat"] += 1
                continue
            k = TARGET_SPAN / span
            x0 = min(s[:, 0].min() for s in strokes)
            y0 = np.percentile(all_y, 50)
            strokes = [np.stack([(s[:, 0] - x0) * k, (s[:, 1] - y0) * k + 70.0], 1) for s in strokes]
            strokes, moved = reorder_by_geometry(strokes, TARGET_SPAN * 0.4)
            moved_total += moved
            moved_lines += moved > 0
            d = to_points(strokes)
            if len(d) > MAX_POINTS:
                dropped["long"] += 1
                continue
            down = d[:-1, 2] < 0.5
            steps.append(np.hypot(*np.diff(d[:, :2], axis=0).T)[down])
            points.append(d)
            offsets.append(offsets[-1] + len(d))
            sentences.append(sentence)
            line_writers.append(writers[form])
            ids.append(name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        points=np.concatenate(points).astype(np.float32),
        offsets=np.array(offsets, dtype=np.int64),
        writers=np.array(line_writers, dtype=np.int32),
    )
    args.out.with_suffix(".json").write_text(json.dumps({"sentences": sentences, "ids": ids}), encoding="utf-8")
    all_steps = np.concatenate(steps)
    lens = np.diff(offsets)
    words = Counter(w for s in sentences for w in s.split())
    print(f"lines kept {len(sentences)} writers {len(set(line_writers))} dropped {dict(dropped)}")
    print(f"lines with late marks {moved_lines}, strokes moved {moved_total}")
    print("pen-down step pct 25/50/75/95", np.percentile(all_steps, [25, 50, 75, 95]).round(2), " points/line pct 5/50/95", np.percentile(lens, [5, 50, 95]).astype(int))
    print(f"distinct words {len(words)}  word instances {sum(words.values())}")


if __name__ == "__main__":
    main()

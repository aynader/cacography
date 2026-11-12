"""Pack MathWriting's isolated handwritten symbols into one array, reading the archive as a stream.

Isolated symbols come from two places: the symbols/ folder, and human-written inks whose
whole expression is a single symbol. Only human inks are used, never the stitched
synthetic ones. Also counts how often each symbol appears inside human expressions, to
show what segmenting those would add. Nothing is extracted to disk.
"""
import argparse
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

NS = "{http://www.w3.org/2003/InkML}"
TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|[^\s{}]")
FOLDERS = ("symbols", "train")
SINGLE_MAX_POINTS = 2000


def parse_ink(raw):
    """(annotations, [(N, 3) x, y, t arrays]) from InkML bytes."""
    if b"<!DOCTYPE" in raw[:400]:
        raise ValueError("doctype not allowed")
    root = ET.fromstring(raw)
    notes = {a.get("type"): a.text for a in root.iter(NS + "annotation")}
    traces = []
    for t in root.iter(NS + "trace"):
        rows = [p.split() for p in t.text.strip().split(",")]
        traces.append(np.array(rows, dtype=np.float64))
    return notes, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/cache/symbols.npz"))
    args = ap.parse_args()
    points, offsets, labels, sources = [], [0], [], []
    inside, bad = Counter(), Counter()
    with tarfile.open(args.archive, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".inkml"):
                continue
            parts = Path(m.name).parts
            if len(parts) < 2 or parts[-2] not in FOLDERS:
                continue
            folder = parts[-2]
            try:
                notes, traces = parse_ink(tf.extractfile(m).read())
            except (ValueError, ET.ParseError):
                bad[folder] += 1
                continue
            if notes.get("inkCreationMethod") != "human" or not traces:
                bad["not human"] += 1
                continue
            label = notes.get("label") if folder == "symbols" else notes.get("normalizedLabel")
            tokens = TOKEN.findall(label or "")
            if folder == "train":
                inside.update(tokens)
                if len(tokens) != 1 or re.fullmatch(r"\\mathbb", tokens[0]):
                    continue
                label = tokens[0]
            n = sum(len(t) for t in traces)
            if n > SINGLE_MAX_POINTS or n < 1:
                bad["size"] += 1
                continue
            rows = []
            for t in traces:
                pen = np.zeros((len(t), 1))
                pen[-1] = 1
                rows.append(np.concatenate([t, pen], 1))
            points.append(np.concatenate(rows).astype(np.float32))
            offsets.append(offsets[-1] + n)
            labels.append(label)
            sources.append(FOLDERS.index(folder))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, points=np.concatenate(points), offsets=np.array(offsets, np.int64), sources=np.array(sources, np.int8))
    args.out.with_suffix(".json").write_text(json.dumps(dict(labels=labels, inside=inside)), encoding="utf-8")
    per = Counter(labels)
    single = Counter(l for l, s in zip(labels, sources) if s == 1)
    print(f"symbols {len(labels)}  labels {len(per)}  from symbols/ {sources.count(0)}  from single-symbol expressions {sources.count(1)}  skipped {dict(bad)}")
    print(f"per label pct 10/50/90 {np.percentile(list(per.values()), [10, 50, 90]).astype(int)}")
    print(f"tokens inside human expressions: {sum(inside.values())}, distinct {len(inside)}")
    needed = ["\\hbar", "\\phi", "\\varphi", "\\psi", "\\partial", "\\nabla", "\\int", "|", "\\bullet", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\pi", "\\sigma", "\\sum", "\\infty", "\\gamma"]
    for k in dict.fromkeys(needed):
        print(f"{k:10s} isolated {per[k]:5d}  of which single-symbol expressions {single[k]:5d}  inside expressions {inside[k]:7d}")


if __name__ == "__main__":
    main()

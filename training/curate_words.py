"""Choose what the model learns from, so it doesn't pick up bad habits.

Everyday words come from IAM-OnDB and from BRUSH writers with careful habits. Many BRUSH
writers leave most of their i and j undotted, and the model copies them faithfully. But
IAM has only a few dozen of each digit and almost no brackets or symbols, so BRUSH's other
writers still teach those: a word of theirs stays only if it holds a character the rest of
the data is short of.

Any word with a fault is out, whoever wrote it, and strays and dry-pen breaks are repaired
first. See cacography_render/marks.py for what counts as a fault.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cacography_render.marks import faults, find_breaks, tidy, undotted  # noqa: E402

CAREFUL_UNDOTTED = 0.15  # a BRUSH writer leaving more i and j than this undotted isn't careful
CAREFUL_BROKEN = 0.06  # nor one with more than this share of words broken up
SHORT_OF = 2000  # a character with fewer samples than this is still worth learning from anyone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=Path, default=Path("data/cache/words.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/words_curated.npz"))
    args = ap.parse_args()
    z = np.load(args.words)
    meta = json.loads(args.words.with_suffix(".json").read_text(encoding="utf-8"))
    off, steps, labels, writers, sources = z["offsets"], z["steps"], z["labels"], z["writers"], z["sources"]
    texts = meta["texts"]

    habits = defaultdict(lambda: Counter())
    words, why = [], Counter()
    for i, text in enumerate(texts):
        s, c, w = steps[off[i] : off[i + 1]].astype(np.float64), labels[off[i] : off[i + 1]], int(writers[i])
        h = habits[w]
        h["words"] += 1
        h["letters"] += sum(ch in "ij" for ch in text)
        h["missing"] += len(undotted(s, c, text))
        h["broken"] += bool(find_breaks(s))
        s, c = tidy(s, c, text)
        found = faults(s, c, text)
        if found:
            why[found[0]] += 1
            continue
        words.append((i, s, c, w))
    careful = {w for w, h in habits.items() if w >= 10000 or (h["missing"] / max(h["letters"], 1) <= CAREFUL_UNDOTTED and h["broken"] / max(h["words"], 1) <= CAREFUL_BROKEN)}
    print(f"writers: IAM {sum(w >= 10000 for w in habits)}, careful BRUSH {sum(w < 10000 for w in careful)} of {sum(w < 10000 for w in habits)}", flush=True)
    print(f"words with a fault, out: {dict(why)}", flush=True)

    counts = Counter()
    for i, _, _, w in words:
        if w in careful:
            counts.update(texts[i])
    kept = []
    borrowed = 0
    for row in words:
        i, _, _, w = row
        if w in careful:
            kept.append(row)
        elif any(counts[ch] < SHORT_OF for ch in texts[i]):
            kept.append(row)
            borrowed += 1
    final = Counter(ch for i, *_ in kept for ch in texts[i])

    out_steps = [s.astype(np.float32) for _, s, _, _ in kept]
    out_labels = [c.astype(np.int16) for _, _, c, _ in kept]
    offsets = np.cumsum([0] + [len(s) for s in out_steps])
    np.savez(
        args.out,
        steps=np.concatenate(out_steps),
        labels=np.concatenate(out_labels),
        offsets=offsets.astype(np.int64),
        writers=np.array([w for *_, w in kept], dtype=np.int32),
        sources=np.array([sources[i] for i, *_ in kept], dtype=np.int8),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps({"texts": [texts[i] for i, *_ in kept], "writer_xh": meta["writer_xh"], "writer_gap": meta["writer_gap"]}), encoding="utf-8"
    )
    iam_before = {int(w) for w in writers if w >= 10000}
    iam_after = {w for *_, w in kept if w >= 10000}
    print(f"words: {len(texts)} in, {len(kept)} kept ({borrowed} borrowed from other BRUSH writers for rare characters)")
    print(f"IAM writers kept {len(iam_after)} of {len(iam_before)}; lost {sorted(iam_before - iam_after)}")
    print("rarest characters now: " + "  ".join(f"{c}:{n}" for c, n in sorted(final.items(), key=lambda kv: kv[1])[:24]))


if __name__ == "__main__":
    main()

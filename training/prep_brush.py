"""Pack BRUSH originals into one array file, keeping lines with late dots and bars."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brushio import load_sample  # noqa: E402
from strokeops import reorder_by_labels  # noqa: E402

MAX_POINTS = 700


def main(root: Path, out: Path):
    points, chars_idx, offsets, sentences, writers = [], [], [0], [], []
    dropped = {"long": 0, "skipped_char": 0, "bad": 0}
    late_strokes = late_lines = 0
    writer_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    for wd in writer_dirs:
        files = [p for p in wd.iterdir() if re.fullmatch(r"\d+", p.name)]
        for path in sorted(files, key=lambda p: int(p.name)):
            try:
                sentence, d, label = load_sample(path.read_bytes())
            except Exception:  # noqa: BLE001
                dropped["bad"] += 1
                continue
            if len(d) > MAX_POINTS or len(d) < 8:
                dropped["long"] += 1
                continue
            idx = label.argmax(1)
            present = set(idx.tolist())
            if any(c != " " and i not in present for i, c in enumerate(sentence)):
                dropped["skipped_char"] += 1
                continue
            d, idx, moved = reorder_by_labels(d, idx)
            late_strokes += moved
            late_lines += moved > 0
            points.append(d)
            chars_idx.append(idx.astype(np.int16))
            offsets.append(offsets[-1] + len(d))
            sentences.append(sentence)
            writers.append(int(wd.name))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        points=np.concatenate(points).astype(np.float32),
        labels=np.concatenate(chars_idx),
        offsets=np.array(offsets, dtype=np.int64),
        writers=np.array(writers, dtype=np.int32),
    )
    out.with_suffix(".json").write_text(json.dumps({"sentences": sentences, "dropped": dropped}), encoding="utf-8")
    print(f"samples kept {len(sentences)} dropped {dropped}  lines with late marks {late_lines}, strokes moved {late_strokes}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

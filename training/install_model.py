"""Copy a trained model into models/ for the renderer, with what it needs about each writer."""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_symbol_bank import slant  # noqa: E402


def writer_slants(words):
    z = np.load(words)
    off, steps, writers = z["offsets"], z["steps"], z["writers"]
    drift = defaultdict(lambda: np.zeros(2))
    for i, w in enumerate(writers.tolist()):
        s = steps[off[i] : off[i + 1]]
        xy = np.cumsum(s[:, :2], 0)
        start = 0
        for e in np.flatnonzero(s[:, 2] > 0.5):
            drift[w] += slant([xy[start : e + 1]])
            start = e + 1
    return {str(w): float(d[0] / d[1]) for w, d in drift.items() if d[1] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--weights", default="best.pt", help="which checkpoint in the run, e.g. ema_best.pt")
    ap.add_argument("--checker", type=Path, required=True)
    ap.add_argument("--limits", type=Path, default=Path("runs/shape_limits.json"))
    ap.add_argument("--words", type=Path, default=Path("data/cache/words_curated.npz"))
    ap.add_argument("--models", type=Path, default=Path("../models"))
    args = ap.parse_args()
    hand, check = args.models / "hand", args.models / "checker"
    hand.mkdir(parents=True, exist_ok=True)
    check.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.run / args.weights, hand / "weights.pt")
    shutil.copy(args.run / "config.json", hand / "config.json")
    shutil.copy(args.checker, check / "weights.pt")
    shutil.copy(args.limits, check / "shape_limits.json")
    meta = json.loads(args.words.with_suffix(".json").read_text(encoding="utf-8"))
    (hand / "writers.json").write_text(
        json.dumps(dict(gap=meta["writer_gap"], xheight=meta["writer_xh"], slant=writer_slants(args.words))), encoding="utf-8"
    )
    print("installed", args.run, "and", args.checker, "; now run calibrate_size.py")


if __name__ == "__main__":
    main()

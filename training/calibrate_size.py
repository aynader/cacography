"""Measure how far biased sampling shrinks a model's writing, for the renderer to undo.

Biased sampling favours the likelier, shorter pen steps, so letters come out small and
close together. Real words are normalized to an x-height of 4 units. Writing the same
texts at several biases, with no checks, shows the size each bias really gives. Saved as
models/hand/size.json: the scale that brings x-height back to 4.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cacography_render.model import NOISE, HandModel, sample  # noqa: E402
from cacography_render.pen import FIRST_IAM_WRITER, LIFT_SHARPEN  # noqa: E402

XH_CHARS = set("acemnorsuvwxz")
BIASES = (0.0, 0.25, 0.5, 0.75, 1.0)
TRUE_XH = 4.0


def measure(words):
    """Median x-height, centre-to-centre advance and width per letter over (steps, chars, text)."""
    xh, adv, width = [], [], []
    for s, c, t in words:
        xy = np.cumsum(s[:, :2], 0)
        h = [np.ptp(xy[c == i, 1]) for i, ch in enumerate(t) if ch in XH_CHARS and (c == i).sum() >= 3]
        if len(h) >= 2:
            xh.append(np.median(h))
        centres = [np.median(xy[c == i, 0]) for i in range(len(t)) if (c == i).sum() >= 3]
        if len(centres) >= 2:
            adv.append(np.median(np.diff(centres)))
        width.append(np.ptp(xy[:, 0]) / len(t))
    return float(np.median(xh)), float(np.median(adv)), float(np.median(width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("../models"))
    ap.add_argument("--words", type=Path, default=Path("data/cache/words.npz"))
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    cfg = json.loads((args.models / "hand/config.json").read_text(encoding="utf-8"))
    model = HandModel(len(cfg["writer_ids"]), hidden=cfg["hidden"]).to(device).eval()
    model.load_state_dict(torch.load(args.models / "hand/weights.pt", map_location=device, weights_only=True)["model"])
    meta = json.loads(args.words.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(args.words)
    off, steps, labels = z["offsets"], z["steps"], z["labels"]
    rng = np.random.default_rng(0)
    idx = [i for i in rng.permutation(len(meta["texts"]))[:10000] if meta["texts"][i].isalpha() and 4 <= len(meta["texts"][i]) <= 10][: args.count]
    texts = [meta["texts"][i] for i in idx]
    index = {w: i for i, w in enumerate(cfg["writer_ids"])}
    writers = rng.choice([w for w in cfg["style_writers"] if w >= FIRST_IAM_WRITER], len(texts))
    style = model.style.weight[[index[int(w)] for w in writers]].detach()
    real = [(steps[off[i] : off[i + 1]], labels[off[i] : off[i + 1]], meta["texts"][i]) for i in idx]
    print("real      x-height %.2f  advance %.2f  width/letter %.2f" % measure(real), flush=True)
    scales = []
    for bias in BIASES:
        out = []
        for k in range(0, len(texts), 128):
            batch = texts[k : k + 128]
            T = 45 * max(len(t) for t in batch) + 25
            r = np.random.default_rng(k)
            noise = r.uniform(size=(len(batch), T, NOISE))
            noise[:, :, 1:3] = r.standard_normal((len(batch), T, 2))
            noise = torch.tensor(noise, dtype=torch.float32)
            out += [(s, c, t) for (s, c), t in zip(sample(model, batch, style[k : k + 128], noise, bias, LIFT_SHARPEN), batch)]
        xh, adv, width = measure(out)
        scales.append(TRUE_XH / xh)
        print(f"bias {bias:.2f} x-height {xh:.2f}  advance {adv:.2f}  width/letter {width:.2f}  -> scale {scales[-1]:.3f}", flush=True)
    (args.models / "hand/size.json").write_text(json.dumps(dict(bias=list(BIASES), scale=scales)), encoding="utf-8")


if __name__ == "__main__":
    main()

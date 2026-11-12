"""Measure the defects a reader sees in generated words, on the same terms as real writing.

Each measure names one complaint from reading whole pages: breaks in a stroke as if the
pen ran dry, i and j without dots, letters squished or stacked on each other, and marks
the text can't explain. Real writing has some of all of these, so the number that matters
is how far generated writing sits above it. Sheets of the worst words check that each
measure finds what the eye finds.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cacography_render.marks import (  # noqa: E402
    SQUISHED, STACKED, classify, find_breaks, flat_lines, stacking, strokes, tightest_three, undotted,
)
from cacography_render.model import NOISE, HandModel, sample  # noqa: E402
from cacography_render.pen import Pen  # noqa: E402


def measure(steps, chars, text, size=1.0):
    xy = np.cumsum(steps[:, :2], 0)
    three = tightest_three(xy, chars, text)
    kinds = [k for _, _, _, k in classify(steps, text)]
    return dict(
        breaks=len(find_breaks(steps)),
        want=sum(c in "ij" for c in text),
        missing=len(undotted(steps, chars, text)),
        stack=stacking(xy, chars, text),
        three=9.0 * SQUISHED if three is None else three * size,
        stray=kinds.count("stray"),
        lines=len(flat_lines(steps, text)),
    )


def report(name, rows):
    s = np.array([r["stack"] for r in rows])
    t = np.array([r["three"] for r in rows])
    want, missing = sum(r["want"] for r in rows), sum(r["missing"] for r in rows)
    out = dict(words=len(rows), breaks=float(np.mean([r["breaks"] > 0 for r in rows])), undotted=missing / max(want, 1),
               stacked=float((s >= STACKED).mean()), squished=float((t < SQUISHED).mean()),
               stray=float(np.mean([r["stray"] > 0 for r in rows])), lines=float(np.mean([r["lines"] > 0 for r in rows])))
    print(f"{name:32s} {len(rows):5d} words | break {100 * out['breaks']:5.1f}% | undotted {100 * out['undotted']:5.1f}%"
          f" | stacked {100 * out['stacked']:5.1f}% | squished {100 * out['squished']:5.1f}%"
          f" | stray {100 * out['stray']:5.1f}% | flat line {100 * out['lines']:5.1f}%", flush=True)
    return out


def draw_worst(words, score, path, count=40):
    """A sheet of the words a measure ranks worst, to check it finds what the eye finds."""
    order = np.argsort(-np.asarray(score, dtype=float))[:count]
    W, H = 260, 120
    sheet = Image.new("RGB", (W * 5, H * (count // 5)), "white")
    d = ImageDraw.Draw(sheet)
    for n, i in enumerate(order):
        steps, _, text = words[i]
        xy = np.cumsum(steps[:, :2], 0) * 4.5
        cx, cy = (n % 5) * W + 8 - xy[:, 0].min(), (n // 5) * H + 78
        for a, b in strokes(steps):
            seg = [(x + cx, y + cy) for x, y in xy[a:b]]
            if len(seg) > 1:
                d.line(seg, fill=(26, 40, 112), width=2)
            else:
                d.ellipse([seg[0][0] - 1, seg[0][1] - 1, seg[0][0] + 1, seg[0][1] + 1], fill=(26, 40, 112))
        d.text(((n % 5) * W + 6, (n // 5) * H + 2), f"{text}  {float(score[i]):.1f}", fill=(0, 120, 0))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("../models"))
    ap.add_argument("--words", type=Path, default=Path("data/cache/words.npz"))
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs/defects"))
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--raw", action="store_true", help="also measure the model before the pen's checks")
    ap.add_argument("--sloppiness", type=float, default=0.5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.words.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(args.words)
    off, all_steps, labels, writers = z["offsets"], z["steps"], z["labels"], z["writers"]
    texts = meta["texts"]
    rng = np.random.default_rng(0)
    natural = [i for i in rng.permutation(len(texts)) if texts[i].isalpha() and 3 <= len(texts[i]) <= 12]
    results = {}
    for name, test in (("real BRUSH", lambda w: w < 10000), ("real IAM", lambda w: w >= 10000)):
        words = [(all_steps[off[i] : off[i + 1]].astype(np.float64), labels[off[i] : off[i + 1]], texts[i]) for i in [j for j in natural if test(writers[j])][:4000]]
        rows = [measure(*w) for w in words]
        results[name] = report(name, rows)
        draw_worst(words, [r["stray"] for r in rows], args.out / f"{name.replace(' ', '_')}_strays.png")
        draw_worst(words, [r["lines"] for r in rows], args.out / f"{name.replace(' ', '_')}_lines.png")
    if args.real_only:
        return
    pool = [texts[i] for i in natural][: args.count]
    device = args.device

    if args.raw:
        cfg = json.loads((args.models / "hand/config.json").read_text(encoding="utf-8"))
        model = HandModel(len(cfg["writer_ids"]), hidden=cfg["hidden"]).to(device).eval()
        model.load_state_dict(torch.load(args.models / "hand/weights.pt", map_location=device, weights_only=True)["model"])
        index = {w: i for i, w in enumerate(cfg["writer_ids"])}
        chosen = rng.choice([w for w in cfg["style_writers"] if w >= 10000], len(pool))
        style = model.style.weight[[index[int(w)] for w in chosen]].detach()
        made = []
        for k in range(0, len(pool), 128):
            batch = pool[k : k + 128]
            T = 45 * max(len(t) for t in batch) + 25
            r = np.random.default_rng(k + 1)
            noise = r.uniform(size=(len(batch), T, NOISE))
            noise[:, :, 1:3] = r.standard_normal((len(batch), T, 2))
            got = sample(model, batch, style[k : k + 128], torch.tensor(noise, dtype=torch.float32), 0.5)
            made += [(s, c, t) for (s, c), t in zip(got, batch)]
        rows = [measure(*m) for m in made]
        results["raw"] = report("model alone, IAM styles", rows)
        draw_worst(made, [r["lines"] for r in rows], args.out / "raw_lines.png")

    # What actually reaches the page: pens, with the checks and retries.
    made = []
    per_pen = 100
    failed = tries = 0
    for k in range(0, len(pool), per_pen):
        pen = Pen(args.models, 7000 + k, args.sloppiness, args.out / "unused.npz", device=device)
        batch = pool[k : k + per_pen]
        got = pen.write([(f"d:{t}#{i}", t) for i, t in enumerate(batch)])
        made += [(*got[f"d:{t}#{i}"], t) for i, t in enumerate(batch)]
        failed += pen.stats["failed"]
        tries += pen.stats["tries"]
        size = pen.size
    rows = [measure(s, c, t, size) for s, c, t in made]
    results["pen"] = report("the pen, with its checks", rows)
    results["pen"]["never_passed"] = failed / len(made)
    results["pen"]["attempts_per_word"] = tries / len(made)
    print(f"never passed {100 * failed / len(made):.1f}%  attempts per word {tries / len(made):.2f}", flush=True)
    draw_worst(made, [r["stack"] for r in rows], args.out / "pen_stacked.png")
    draw_worst(made, [-r["three"] for r in rows], args.out / "pen_squished.png")
    draw_worst(made, [r["breaks"] for r in rows], args.out / "pen_breaks.png")
    draw_worst(made, [r["missing"] for r in rows], args.out / "pen_undotted.png")
    (args.out / "defects.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

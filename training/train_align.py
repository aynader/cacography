"""Train a line reader on BRUSH and IAM-OnDB, whose only job is locating IAM's letters.

IAM-OnDB gives the text of each line but not which strokes form which letter. CTC
training needs no such labels, and a trained reader can then line the text up against
the strokes. This reader is never used to judge generated writing: it sees every
writer and word, so it would flatter the results. train_reader_words.py does that job.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import CHAR_TO_ID, SCALE, Y_ORIGIN, encode  # noqa: E402
from train_reader import Reader, edit_distance, greedy_decode, out_lengths, pad  # noqa: E402


def load_steps(cache: Path):
    z = np.load(cache)
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    off, pts = z["offsets"], z["points"]
    out = []
    for i, s in enumerate(meta["sentences"]):
        if any(c not in CHAR_TO_ID for c in s):
            continue
        d = pts[off[i] : off[i + 1]]
        delta = np.zeros((len(d), 3), np.float32)
        delta[1:, :2] = (d[1:, :2] - d[:-1, :2]) / SCALE
        delta[0, 1] = (d[0, 1] - Y_ORIGIN) / SCALE
        delta[:, 2] = d[:, 2]
        out.append((torch.from_numpy(delta), encode(s)))
    return out


def augment(s):
    k = 1 + 0.1 * (2 * random.random() - 1)
    shear = 0.15 * (2 * random.random() - 1)
    s = s.clone()
    s[:, 0] = (s[:, 0] + shear * s[:, 1]) * k
    s[:, 1] = s[:, 1] * k
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", type=Path, default=[Path("data/cache/brush2.npz"), Path("data/cache/iam.npz")])
    ap.add_argument("--out", type=Path, default=Path("runs/align"))
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=56)
    args = ap.parse_args()
    random.seed(2)
    torch.manual_seed(2)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [l for c in args.caches for l in load_steps(c)]
    random.shuffle(lines)
    val, train = lines[:600], lines[600:]
    print(f"train {len(train)} val {len(val)}", flush=True)
    model = Reader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def make_batches():
        order = list(range(len(train)))
        random.shuffle(order)
        pools = [sorted(order[i : i + args.batch * 30], key=lambda j: len(train[j][0])) for i in range(0, len(order), args.batch * 30)]
        out = [p[i : i + args.batch] for p in pools for i in range(0, len(p), args.batch)]
        random.shuffle(out)
        return out

    total = args.epochs * len(make_batches()) + args.epochs + 5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total)
    best = 9.0
    for epoch in range(1, args.epochs + 1):
        t0, agg = time.time(), 0.0
        model.train()
        bs = make_batches()
        for b in bs:
            x, n = pad([augment(train[i][0]) for i in b], device)
            target = torch.cat([train[i][1] for i in b]).to(device)
            tl = torch.tensor([len(train[i][1]) for i in b], device=device)
            log_probs = F.log_softmax(model(x), -1).transpose(0, 1)
            loss = F.ctc_loss(log_probs, target, out_lengths(n), tl, blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            agg += loss.item()
        model.eval()
        err = tot = 0
        with torch.no_grad():
            for i in range(0, len(val), 64):
                b = val[i : i + 64]
                x, n = pad([l[0] for l in b], device)
                for seq, l in zip(greedy_decode(model(x), out_lengths(n)), b):
                    err += edit_distance(seq, l[1].tolist())
                    tot += len(l[1])
        cer = err / tot
        print(f"epoch {epoch:3d} ctc {agg / len(bs):.3f} val CER {cer:.3f} | {time.time() - t0:.0f}s", flush=True)
        if cer < best:
            best = cer
            torch.save({"model": model.state_dict(), "cer": cer, "epoch": epoch}, args.out / "best.pt")
    print("finished, best val CER", best, flush=True)


if __name__ == "__main__":
    main()

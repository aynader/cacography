"""Train the recognizer that checks generated handwriting says what it should.

It reads pen movement directly with a bidirectional LSTM and CTC. Held-out writers
give an honest baseline: real handwriting from strangers sets how strict the check
can reasonably be.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import VOCAB, load_lines  # noqa: E402


class Reader(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 96, 5, padding=2), nn.GELU(), nn.Conv1d(96, 128, 5, padding=2, stride=2), nn.GELU()
        )
        self.rnn = nn.LSTM(128, hidden, num_layers=3, batch_first=True, bidirectional=True, dropout=0.15)
        self.out = nn.Linear(hidden * 2, len(VOCAB) + 1)

    def forward(self, steps):
        h = self.conv(steps.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.out(h)


def out_lengths(lengths):
    return (lengths + 1) // 2


def pad(batch, device):
    T = max(len(s) for s in batch)
    x = torch.zeros(len(batch), T, 3)
    for i, s in enumerate(batch):
        x[i, : len(s)] = s
    return x.to(device), torch.tensor([len(s) for s in batch], device=device)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_decode(logits, lengths):
    ids = logits.argmax(-1).cpu().numpy()
    out = []
    for row, n in zip(ids, lengths.tolist()):
        seq, last = [], 0
        for v in row[:n]:
            if v != last and v != 0:
                seq.append(int(v))
            last = v
        out.append(seq)
    return out


def char_error(model, lines, device, size=64):
    model.eval()
    err = total = 0
    with torch.no_grad():
        for i in range(0, len(lines), size):
            b = lines[i : i + size]
            x, n = pad([l.steps for l in b], device)
            for seq, l in zip(greedy_decode(model(x), out_lengths(n)), b):
                ref = l.text.tolist()
                err += edit_distance(seq, ref)
                total += len(ref)
    model.train()
    return err / max(total, 1)


def augment(steps):
    """Small shape changes so the reader isn't tied to BRUSH's exact writing."""
    k = 1 + 0.1 * (2 * random.random() - 1)
    shear = 0.15 * (2 * random.random() - 1)
    s = steps.clone()
    s[:, 0] = (s[:, 0] + shear * s[:, 1]) * k
    s[:, 1] = s[:, 1] * k
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/brush.npz"))
    ap.add_argument("--writers", type=Path, default=Path("writers.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/reader"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    random.seed(1)
    torch.manual_seed(1)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    train = load_lines(args.cache, args.writers)
    test = load_lines(args.cache, args.writers, include_holdout=True)
    print(f"train {len(train)} held-out-writer test {len(test)}", flush=True)
    model = Reader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    total = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total + 1)
    best = 9.0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        t0, agg, count = time.time(), 0.0, 0
        for i in range(0, len(train) - args.batch + 1, args.batch):
            b = train[i : i + args.batch]
            x, n = pad([augment(l.steps) for l in b], device)
            target = torch.cat([l.text for l in b]).to(device)
            tl = torch.tensor([len(l.text) for l in b], device=device)
            log_probs = F.log_softmax(model(x), -1).transpose(0, 1)
            loss = F.ctc_loss(log_probs, target, out_lengths(n), tl, blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            agg += loss.item()
            count += 1
        cer = char_error(model, test[:1200], device)
        print(f"epoch {epoch:3d} ctc {agg / count:.3f} real held-out CER {cer:.3f} | {time.time() - t0:.0f}s", flush=True)
        if cer < best:
            best = cer
            torch.save({"model": model.state_dict(), "cer": cer, "epoch": epoch}, args.out / "best.pt")
    print("finished, best real held-out CER", best, flush=True)


if __name__ == "__main__":
    main()

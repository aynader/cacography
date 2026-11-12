"""Train the checker that reads generated words, on real words plus spliced new ones.

Reported honestly on held-back writers writing held-back words. Results on familiar
words are printed alongside, because the gap between the two shows how much of the
checker's skill is memorized vocabulary.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_reader import Reader, edit_distance, greedy_decode, out_lengths, pad  # noqa: E402
from wordsdata import SplicePool, load_splits  # noqa: E402


def augment(s):
    k = 1 + 0.12 * (2 * random.random() - 1)
    shear = 0.2 * (2 * random.random() - 1)
    s = s.clone()
    s[:, 0] = (s[:, 0] + shear * s[:, 1]) * k
    s[:, 1] = s[:, 1] * k
    return s


def score(model, lines, device, size=256):
    """Character error rate and share of words read exactly."""
    model.eval()
    err = total = exact = 0
    with torch.no_grad():
        for i in range(0, len(lines), size):
            b = lines[i : i + size]
            x, n = pad([l.steps for l in b], device)
            for seq, l in zip(greedy_decode(model(x), out_lengths(n)), b):
                ref = l.text.tolist()
                d = edit_distance(seq, ref)
                err += d
                total += len(ref)
                exact += d == 0
    model.train()
    return err / max(total, 1), exact / max(len(lines), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=Path, default=Path("data/cache/words.npz"))
    ap.add_argument("--writers", type=Path, default=Path("writers.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/checker"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--splice", type=float, default=0.5)
    args = ap.parse_args()
    random.seed(4)
    torch.manual_seed(4)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_splits(args.words, args.writers)
    print(
        f"train {len(data.train)} unseen-word test {len(data.test_unseen_words)} (of which unseen writers {len(data.test_unseen)})"
        f" familiar-word test {len(data.test_seen)} held words {len(data.held_words)}",
        flush=True,
    )
    pool = SplicePool(data.train)
    model = Reader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    per_epoch = len(data.train) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * per_epoch + 5)
    n_splice = int(args.batch * args.splice)
    best = 9.0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(data.train)
        t0, agg = time.time(), 0.0
        real_n = args.batch - n_splice
        for i in range(per_epoch):
            b = data.train[i * real_n : (i + 1) * real_n]
            items = [(augment(l.steps), l.text) for l in b]
            for _ in range(n_splice):
                made = pool.make()
                if made is not None:
                    items.append((augment(made[0]), made[1]))
            x, n = pad([s for s, _ in items], device)
            target = torch.cat([t for _, t in items]).to(device)
            tl = torch.tensor([len(t) for _, t in items], device=device)
            log_probs = F.log_softmax(model(x), -1).transpose(0, 1)
            loss = F.ctc_loss(log_probs, target, out_lengths(n), tl, blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            agg += loss.item()
        strict = score(model, data.test_unseen, device)
        unseen = score(model, data.test_unseen_words, device)
        seen = score(model, data.test_seen[:6000], device)
        print(
            f"epoch {epoch:3d} ctc {agg / per_epoch:.3f} | unseen words CER {unseen[0]:.3f} exact {unseen[1]:.3f}"
            f" | unseen words and writers CER {strict[0]:.3f} | familiar words CER {seen[0]:.3f} exact {seen[1]:.3f}"
            f" | {time.time() - t0:.0f}s",
            flush=True,
        )
        if unseen[0] < best:
            best = unseen[0]
            torch.save({"model": model.state_dict(), "cer": unseen[0], "epoch": epoch}, args.out / "best.pt")
    print("finished, best unseen-word CER", round(best, 4), flush=True)


if __name__ == "__main__":
    main()

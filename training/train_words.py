"""Train the word model, choosing the saved version by how well it writes freely.

Loss with the true previous point always available says little about free writing, so
every few epochs the model writes words it never trained on and the checker reads them.
Even that isn't enough: a model with better losses can write worse. So with
--every, checkpoints are kept for gate_models.py to judge on the defects that matter.

--init starts from an installed model's weights (a fine-tune), and --extra mixes in words
a model wrote that passed every check, --extra-share of each epoch.

--ema keeps a running average of the weights as well; its checkpoints (ema_eNNN.pt,
ema_best.pt) sit beside the plain ones, so one run gives both to compare.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import VOCAB, collate  # noqa: E402
from handmodel import HandModel, losses, sample  # noqa: E402
from handwrite import compose_line, pad_texts, read_back  # noqa: E402
from inkplain import render  # noqa: E402
from train_hand import augment, batches  # noqa: E402
from train_reader import Reader, edit_distance  # noqa: E402
from wordsdata import balanced, load_splits, make_word  # noqa: E402

PROBES = [
    "The time-dependent equation is",
    "For a stationary state we can write",
    "boundary conditions matter here!",
]


def free_score(model, checker, texts, writers, index, device, bias, seed=9):
    """Character error when writing freely, with no retries."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = random.Random(seed)
    picks = [rng.choice(writers) for _ in texts]
    style = model.style(torch.tensor([index[w] for w in picks], device=device))
    err = total = exact = 0
    for i in range(0, len(texts), 128):
        chunk = texts[i : i + 128]
        made = sample(model, pad_texts(chunk, device), style[i : i + 128], bias=bias, generator=gen, max_steps=500)
        for seq, t in zip(read_back(checker, made, device), chunk):
            ref = pad_texts([t], "cpu")[0].tolist()
            d = edit_distance(seq, ref)
            err += d
            total += len(ref)
            exact += d == 0
    return err / total, exact / len(texts)


def save_sheet(model, writers, index, gaps, path, device):
    gen = torch.Generator(device=device).manual_seed(5)
    lines = []
    for w in writers:
        style = model.style(torch.tensor([index[w]], device=device))
        for text in PROBES:
            words = text.split()
            made = sample(model, pad_texts(words, device), style.expand(len(words), -1), bias=1.0, generator=gen, max_steps=500)
            lines.append(compose_line(made, gap=tuple(gaps.get(w, (14.0, 4.0))), rng=np.random.default_rng(w)))
    pics = [render(p, scale=1.5, margin=6) for p in lines]
    W, H = 760, 82
    sheet = Image.new("RGB", (W * len(PROBES), H * len(writers)), (255, 255, 255))
    for k, pic in enumerate(pics):
        f = min((W - 6) / pic.width, (H - 4) / pic.height, 1.0)
        pic = pic.resize((max(1, int(pic.width * f)), max(1, int(pic.height * f))), Image.LANCZOS)
        sheet.paste(pic, ((k % len(PROBES)) * W + 2, (k // len(PROBES)) * H + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=Path, default=Path("data/cache/words_curated.npz"))
    ap.add_argument("--writers", type=Path, default=Path("writers.json"))
    ap.add_argument("--checker", type=Path, default=Path("runs/checker/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/hand"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Heavier settings (0.3, 1e-2) against overfitting pen lifts blended the writers into
    # one average hand, and wrote worse words.
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init", type=Path, help="weights to start from, e.g. ../models/hand/weights.pt")
    ap.add_argument("--extra", type=Path, help="extra training words from self_words.py")
    ap.add_argument("--extra-share", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=5, help="epochs between checkpoints")
    ap.add_argument("--seed", type=int, default=0, help="starting weights and batch order; the data split has its own")
    ap.add_argument("--ema", type=float, default=0.0, help="decay of a running average of the weights, e.g. 0.999; 0 for none")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_splits(args.words, args.writers)
    index = {w: i for i, w in enumerate(data.writer_ids)}
    print(f"train {len(data.train)} val {len(data.val)} writers {len(index)} style writers {len(data.style_writers)}", flush=True)

    held = sorted({"".join(VOCAB[c - 1] for c in l.text.tolist()) for l in data.test_unseen_words})
    print(f"held-back test words {len(held)}", flush=True)
    (args.out / "config.json").write_text(
        json.dumps({"writer_ids": data.writer_ids, "style_writers": data.style_writers, "holdout": data.holdout_writers, "hidden": args.hidden}),
        encoding="utf-8",
    )
    checker = Reader().to(device).eval()
    checker.load_state_dict(torch.load(args.checker, map_location=device, weights_only=True)["model"])
    train = data.train[: args.batch * 20] if args.smoke else balanced(data.train)
    print(f"practice per epoch {len(train)} words", flush=True)

    model = HandModel(len(data.writer_ids), hidden=args.hidden, dropout=args.dropout).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True)["model"])
        print("starting from", args.init, flush=True)
    extra = []
    if args.extra:
        z = np.load(args.extra)
        texts = json.loads(args.extra.with_suffix(".json").read_text(encoding="utf-8"))["texts"]
        off = z["offsets"]
        extra = [make_word(z["steps"][off[i] : off[i + 1]], z["labels"][off[i] : off[i + 1]], t, int(z["writers"][i]))
                 for i, t in enumerate(texts) if int(z["writers"][i]) in index]
        print(f"extra words {len(extra)}, {args.extra_share:.0%} of each epoch", flush=True)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * max(1, len(batches(train, args.batch))) / (1 - args.extra_share if extra else 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 300, 1.0) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    )
    ema = None
    if args.ema > 0:
        ema = HandModel(len(data.writer_ids), hidden=args.hidden, dropout=args.dropout).to(device).eval()
        ema.load_state_dict(model.state_dict())
        ema.requires_grad_(False)
    sheet_writers = data.style_writers[:: max(1, len(data.style_writers) // 10)][:10]
    best_free = best_ema = 9.0
    log = open(args.out / "log.txt", "a", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, agg = time.time(), np.zeros(3)
        epoch_lines = train
        if extra:
            n = int(len(train) * args.extra_share / (1 - args.extra_share))
            epoch_lines = train + random.choices(extra, k=n)
        bs = batches(epoch_lines, args.batch)
        for b in bs:
            text, steps, char, frames, lifts, advance, mask, writer = collate([augment(epoch_lines[i]) for i in b], index, device)
            nll, pen, adv = losses(model(text, steps, char, frames, lifts, writer), steps, advance, mask)
            opt.zero_grad(set_to_none=True)
            (nll + pen + adv).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                        if e.dtype.is_floating_point:
                            e.lerp_(m, 1 - args.ema)
                        else:
                            e.copy_(m)
            agg += [nll.item(), pen.item(), adv.item()]
        agg /= len(bs)
        model.eval()
        vagg, vn = np.zeros(3), 0
        with torch.no_grad():
            for b in batches(data.val, args.batch, shuffle=False):
                text, steps, char, frames, lifts, advance, mask, writer = collate([data.val[i] for i in b], index, device)
                nll, pen, adv = losses(model(text, steps, char, frames, lifts, writer), steps, advance, mask)
                vagg += [nll.item(), pen.item(), adv.item()]
                vn += 1
        vagg /= max(vn, 1)
        msg = (
            f"epoch {epoch:3d} train nll {agg[0]:.3f} pen {agg[1]:.4f} adv {agg[2]:.4f}"
            f" | val nll {vagg[0]:.3f} pen {vagg[1]:.4f} adv {vagg[2]:.4f} | {time.time() - t0:.0f}s"
        )
        if epoch % args.every == 0 or epoch == args.epochs or args.smoke:
            cer, exact = free_score(model, checker, held[:1500], data.style_writers, index, device, bias=1.0)
            msg += f" | free writing, unseen words: CER {cer:.3f} exact {exact:.3f}"
            torch.save({"model": model.state_dict(), "epoch": epoch, "free_cer": cer}, args.out / f"e{epoch:03d}.pt")
            if cer < best_free:
                best_free = cer
                torch.save({"model": model.state_dict(), "epoch": epoch, "free_cer": cer}, args.out / "best.pt")
            save_sheet(model, sheet_writers, index, data.writer_gap, args.out / f"sheet_e{epoch:03d}.png", device)
            if ema is not None:
                ecer, eexact = free_score(ema, checker, held[:1500], data.style_writers, index, device, bias=1.0)
                msg += f" | ema CER {ecer:.3f} exact {eexact:.3f}"
                torch.save({"model": ema.state_dict(), "epoch": epoch, "free_cer": ecer}, args.out / f"ema_e{epoch:03d}.pt")
                if ecer < best_ema:
                    best_ema = ecer
                    torch.save({"model": ema.state_dict(), "epoch": epoch, "free_cer": ecer}, args.out / "ema_best.pt")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    print("finished, best free CER", round(best_free, 4), flush=True)


if __name__ == "__main__":
    main()

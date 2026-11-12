"""Train the handwriting generator, writing checkpoints and sample sheets as it goes."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handdata import SCALE, Line, collate, encode, load_lines, writer_split  # noqa: E402
from handmodel import HandModel, losses, sample  # noqa: E402
from inkplain import render  # noqa: E402

PROBES = ["The time-dependent equation is", "For a stationary state we can", "boundary conditions matter!"]


def augment(line):
    """A slight per-line change of size and slant, small enough to keep the writer's style."""
    k = 1 + 0.06 * (2 * random.random() - 1)
    shear = 0.06 * (2 * random.random() - 1)
    s = line.steps.clone()
    s[:, 0] = (s[:, 0] + shear * s[:, 1]) * k
    s[:, 1] = s[:, 1] * k
    return Line(line.text, s, line.char, line.frames, line.lifts, line.advance, line.writer)


def to_points(steps):
    s = steps.detach().cpu().numpy().astype(np.float64)
    xy = np.cumsum(s[:, :2] * SCALE, 0)
    return np.concatenate([xy, s[:, 2:3]], 1)


def batches(lines, size, shuffle=True):
    """Group similar lengths so little of each batch is padding."""
    order = list(range(len(lines)))
    if shuffle:
        random.shuffle(order)
    pool = [order[i : i + size * 40] for i in range(0, len(order), size * 40)]
    out = []
    for chunk in pool:
        chunk.sort(key=lambda i: len(lines[i].steps))
        out += [chunk[i : i + size] for i in range(0, len(chunk), size)]
    if shuffle:
        random.shuffle(out)
    return out


def save_sheet(model, writer_ids, writer_index, path, device):
    from PIL import Image

    model.eval()
    ids = [writer_index[w] for w in writer_ids]
    style = model.style(torch.tensor(ids, device=device))
    rows = []
    for text in PROBES:
        t = encode(text)[None].to(device).expand(len(ids), -1)
        rows.extend(sample(model, t, style, bias=1.0))
    pics = [render(to_points(r), scale=1.5, margin=6) for r in rows]
    W, H = 760, 74
    sheet = Image.new("RGB", (W * len(PROBES), H * len(ids)), (255, 255, 255))
    for k, pic in enumerate(pics):
        f = min((W - 6) / pic.width, (H - 4) / pic.height, 1.0)
        pic = pic.resize((max(1, int(pic.width * f)), max(1, int(pic.height * f))), Image.LANCZOS)
        sheet.paste(pic, ((k // len(ids)) * W + 2, (k % len(ids)) * H + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/brush.npz"))
    ap.add_argument("--writers", type=Path, default=Path("writers.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/hand_lines"))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lines = load_lines(args.cache, args.writers)
    natural, holdout = writer_split(args.writers)
    random.shuffle(lines)
    val_n = max(64, len(lines) // 40)
    val, train = lines[:val_n], lines[val_n:]
    writer_ids = sorted({l.writer for l in lines})
    writer_index = {w: i for i, w in enumerate(writer_ids)}
    style_writers = [w for w in natural if w not in holdout]
    print(f"lines train {len(train)} val {len(val)}  writers {len(writer_ids)}  held out {holdout}  load {time.time() - t0:.0f}s", flush=True)
    (args.out / "config.json").write_text(
        json.dumps({"writer_ids": writer_ids, "style_writers": style_writers, "holdout": holdout, "hidden": args.hidden}),
        encoding="utf-8",
    )
    if args.smoke:
        train = train[: args.batch * 20]

    model = HandModel(len(writer_ids), hidden=args.hidden).to(device)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * max(1, len(batches(train, args.batch)))
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min((s + 1) / warm, 1.0) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))),
    )
    sheet_writers = style_writers[:: max(1, len(style_writers) // 8)][:8]
    best = float("inf")
    log = open(args.out / "log.txt", "a", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = np.zeros(3)
        bs = batches(train, args.batch)
        for b in bs:
            text, steps, char, frames, lifts, advance, mask, writer = collate(
                [augment(train[i]) for i in b], writer_index, device
            )
            nll, pen, adv = losses(model(text, steps, char, frames, lifts, writer), steps, advance, mask)
            loss = nll + pen + adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            agg += [nll.item(), pen.item(), adv.item()]
        model.eval()
        vagg, vn = np.zeros(3), 0
        with torch.no_grad():
            for b in batches(val, args.batch, shuffle=False):
                text, steps, char, frames, lifts, advance, mask, writer = collate([val[i] for i in b], writer_index, device)
                nll, pen, adv = losses(model(text, steps, char, frames, lifts, writer), steps, advance, mask)
                vagg += [nll.item(), pen.item(), adv.item()]
                vn += 1
        vagg /= max(vn, 1)
        agg /= len(bs)
        msg = (
            f"epoch {epoch:3d} train nll {agg[0]:.3f} pen {agg[1]:.4f} adv {agg[2]:.4f}"
            f" | val nll {vagg[0]:.3f} pen {vagg[1]:.4f} adv {vagg[2]:.4f}"
            f" | {time.time() - t0:.0f}s gpu {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
        )
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        score = vagg.sum()
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": score}, args.out / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, args.out / "last.pt")
        if epoch % 5 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.out / f"e{epoch:03d}.pt")
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs or args.smoke:
            save_sheet(model, sheet_writers, writer_index, args.out / f"sheet_e{epoch:03d}.png", device)
    print("finished, best val", best, flush=True)


if __name__ == "__main__":
    main()

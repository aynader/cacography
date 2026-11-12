"""Decide whether a new handwriting model ships: it must beat the installed one, not just differ.

Training losses can't tell: a model with better losses can write worse. So every checkpoint writes
the same words with the same random noise as the installed model, and the defects a
reader sees are counted (see defects.py). Paired like that, a difference comes from the
model, not from luck. The best checkpoints are then written with the full pen, on test
words and on real documents, where what counts is how many words never pass.

A candidate ships only if no defect rate is worse than the installed model's by more
than the noise between runs, and it is better on the total.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cacography_render.model import NOISE, HandModel, sample  # noqa: E402
from cacography_render.page import PageWriter, page_units  # noqa: E402
from cacography_render.pen import LIFT_SHARPEN, Pen  # noqa: E402
from defects import measure  # noqa: E402

MEASURES = ("breaks", "undotted", "stacked", "squished", "stray", "lines")
SLACK = 0.004  # rates closer than this are within the noise of 2000 words


def rates(rows):
    want = sum(r["want"] for r in rows)
    return dict(
        breaks=float(np.mean([r["breaks"] > 0 for r in rows])),
        undotted=sum(r["missing"] for r in rows) / max(want, 1),
        stacked=float(np.mean([r["stack"] >= 1.0 for r in rows])),
        squished=float(np.mean([r["three"] < 4.0 for r in rows])),
        stray=float(np.mean([r["stray"] > 0 for r in rows])),
        lines=float(np.mean([r["lines"] > 0 for r in rows])),
    )


def raw_rates(weights, cfg, texts, writers, device):
    """The model alone, before the pen's checks: the same words, styles and noise for every model."""
    model = HandModel(len(cfg["writer_ids"]), hidden=cfg["hidden"]).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True)["model"])
    index = {w: i for i, w in enumerate(cfg["writer_ids"])}
    style = model.style.weight[[index[w] for w in writers]].detach()
    rows = []
    for k in range(0, len(texts), 128):
        batch = texts[k : k + 128]
        T = 45 * max(len(t) for t in batch) + 25
        r = np.random.default_rng(k + 1)
        noise = r.uniform(size=(len(batch), T, NOISE))
        noise[:, :, 1:3] = r.standard_normal((len(batch), T, 2))
        got = sample(model, batch, style[k : k + 128], torch.tensor(noise, dtype=torch.float32), 0.5, LIFT_SHARPEN)
        rows += [measure(s, c, t) for (s, c), t in zip(got, batch)]
    return rates(rows)


def stage(weights, base, folder):
    """A model folder like ../models with these hand weights, for the pen to load."""
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(base, folder)
    shutil.copy(weights, folder / "hand" / "weights.pt")
    return folder


def pen_rates(models, texts, docs, device):
    """With the full pen: words that never pass, attempts per word, on test words and on documents."""
    failed = tries = 0
    for k in range(0, len(texts), 100):
        pen = Pen(models, 7000 + k, 0.5, Path("runs/gate/unused.npz"), device=device)
        pen.cache = {}
        pen.write([(f"g:{t}#{i}", t) for i, t in enumerate(texts[k : k + 100])])
        failed += pen.stats["failed"]
        tries += pen.stats["tries"]
    out = dict(never_passed=failed / len(texts), attempts=tries / len(texts))
    doc_failed = doc_words = 0
    for job in docs:
        folder = Path(job + ".cacography")
        doc = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        pen = Pen(models, int(doc["seed"]), float(doc["sloppiness"]), Path("runs/gate/unused.npz"), device=device)
        pen.cache = {}
        writer = PageWriter(pen, doc, 200, None)
        requests = []
        for n in range(1, int(doc["pages"]) + 1):
            requests += writer.plan(page_units(json.loads((folder / f"page-{n}.json").read_text(encoding="utf-8"))))
        pen.write(requests)
        doc_failed += pen.stats["failed"]
        doc_words += len(requests)
    out["documents_never_passed"] = doc_failed / max(doc_words, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+", type=Path, help="checkpoints (.pt) to judge")
    ap.add_argument("--base", type=Path, default=Path("../models"), help="the installed model, to beat")
    ap.add_argument("--words", type=Path, default=Path("data/cache/words_curated.npz"))
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--finalists", type=int, default=3)
    ap.add_argument("--docs", nargs="*", default=[], help="built documents (their .cacography folders) to write as well")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs/gate/gate.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.base / "hand/config.json").read_text(encoding="utf-8"))
    meta = json.loads(args.words.with_suffix(".json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    texts = [t for t in (meta["texts"][i] for i in rng.permutation(len(meta["texts"]))) if t.isalpha() and 3 <= len(t) <= 12][: args.count]
    writers = [int(w) for w in rng.choice([w for w in cfg["style_writers"] if w >= 10000], len(texts))]

    results = {"installed": raw_rates(args.base / "hand/weights.pt", cfg, texts, writers, args.device)}
    for c in args.candidates:
        results[str(c)] = raw_rates(c, cfg, texts, writers, args.device)
    base = results["installed"]
    print(f"{'model':40s} " + " ".join(f"{m:>9s}" for m in MEASURES) + "      total")
    for name, r in results.items():
        print(f"{name[-40:]:40s} " + " ".join(f"{100 * r[m]:8.1f}%" for m in MEASURES) + f"  {100 * sum(r.values()):8.1f}%", flush=True)

    def no_worse(r):
        return all(r[m] <= base[m] + SLACK for m in MEASURES)

    ranked = sorted((n for n in results if n != "installed"), key=lambda n: (not no_worse(results[n]), sum(results[n].values())))
    finalists = [n for n in ranked if no_worse(results[n])][: args.finalists]
    print("finalists (no measure worse than installed):", finalists or "none", flush=True)

    full = {"installed": pen_rates(args.base, texts[:1000], args.docs, args.device)}
    for n in finalists:
        full[n] = pen_rates(stage(Path(n), args.base, Path("runs/gate/stage") / Path(n).parent.name / Path(n).stem), texts[:1000], args.docs, args.device)
    for n, r in full.items():
        print(f"{n[-40:]:40s} never passed {100 * r['never_passed']:5.1f}%  attempts {r['attempts']:.2f}  documents never passed {100 * r['documents_never_passed']:5.1f}%", flush=True)
    b = full["installed"]
    winners = [n for n in finalists if full[n]["never_passed"] <= b["never_passed"] + 0.003
               and full[n]["documents_never_passed"] <= b["documents_never_passed"] + 0.003 and full[n]["attempts"] <= b["attempts"] + 0.02]
    verdict = min(winners, key=lambda n: sum(results[n].values())) if winners else None
    print("verdict:", verdict or "keep the installed model", flush=True)
    args.out.write_text(json.dumps(dict(raw=results, pen=full, verdict=verdict), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

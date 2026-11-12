"""Word-level dataset with the splits that keep the results honest.

Two things are held back from all training: some writers, and some words. Scoring on
held-back words from held-back writers shows how a model handles text and people it
has truly never met, instead of rewarding memorized vocabulary.
"""
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from handdata import Line, encode, writer_split

IAM_HOLDOUT = 30
WORD_HOLDOUT_FRACTION = 0.06
WORD_HOLDOUT_MAX_COUNT = 8
SPLIT_SEED = 7


def word_key(text):
    """Compare words without case or attached punctuation."""
    return re.sub(r"[^a-z]", "", text.lower())


def make_word(steps, labels, text, writer):
    T = len(labels)
    pen = steps[:, 2]
    frames = np.zeros(T, np.int64)
    lifts = np.zeros(T, np.int64)
    for t in range(1, T):
        if labels[t] == labels[t - 1]:
            frames[t] = frames[t - 1] + 1
            lifts[t] = lifts[t - 1] + int(pen[t - 1] > 0.5)
    advance = np.ones(T, np.float32)
    advance[:-1] = (labels[1:] != labels[:-1]).astype(np.float32)
    return Line(
        encode(text),
        torch.from_numpy(np.array(steps, np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
        torch.from_numpy(frames),
        torch.from_numpy(lifts),
        torch.from_numpy(advance),
        writer,
    )


@dataclass
class Splits:
    train: list  # training writers, training words
    val: list  # training writers, training words, never trained on
    test_unseen: list  # held-back writers writing held-back words
    test_seen: list  # held-back writers writing familiar words
    test_unseen_words: list  # anyone writing held-back words: larger, but the writers may be familiar
    held_words: set
    holdout_writers: list
    style_writers: list
    writer_ids: list
    writer_xh: dict
    writer_gap: dict


def load_splits(cache: Path, writers_json: Path) -> Splits:
    z = np.load(cache)
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    texts = meta["texts"]
    off, steps, labels, writers = z["offsets"], z["steps"], z["labels"], z["writers"]
    natural, brush_holdout = writer_split(writers_json)
    rng = random.Random(SPLIT_SEED)
    iam_writers = sorted({int(w) for w in writers if w >= 10000})
    iam_holdout = rng.sample(iam_writers, IAM_HOLDOUT)
    holdout = set(brush_holdout) | set(iam_holdout)

    counts = Counter(word_key(t) for t in texts)
    candidates = sorted(k for k, c in counts.items() if 3 <= len(k) and c <= WORD_HOLDOUT_MAX_COUNT)
    held = set(rng.sample(candidates, int(len(candidates) * WORD_HOLDOUT_FRACTION)))

    train, val, test_unseen, test_seen, test_unseen_words = [], [], [], [], []
    for i, text in enumerate(texts):
        w = int(writers[i])
        key = word_key(text)
        is_held_word, is_held_writer = key in held, w in holdout
        line = make_word(steps[off[i] : off[i + 1]], labels[off[i] : off[i + 1]], text, w)
        if is_held_word:
            test_unseen_words.append(line)
            if not is_held_writer:
                continue  # never trained on, whoever wrote it
        if is_held_writer:
            (test_unseen if is_held_word else test_seen).append(line)
        elif rng.random() < 0.015:
            val.append(line)
        else:
            train.append(line)
    writer_ids = sorted({l.writer for l in train})
    style_writers = [w for w in writer_ids if w >= 10000 or w in set(natural)]
    return Splits(
        train, val, test_unseen, test_seen, test_unseen_words, held, sorted(holdout), style_writers, writer_ids,
        {int(k): v for k, v in meta["writer_xh"].items()},
        {int(k): v for k, v in meta["writer_gap"].items()},
    )


def balanced(lines, common=5000, most=3.0, seed=2):
    """Repeat words holding rare characters, so digits get practised more than a few hundred times.

    A word's weight comes from its rarest character: the square root of how far short of
    `common` samples that character falls, at most `most`. Fractions round at random.
    """
    counts = Counter(c for l in lines for c in l.text.tolist())
    rng = random.Random(seed)
    out = []
    for l in lines:
        rarest = min(counts[c] for c in l.text.tolist())
        w = min(max(1.0, (common / rarest) ** 0.5), most)
        out += [l] * (int(w) + (rng.random() < w - int(w)))
    return out


class SplicePool:
    """Make new words from one writer's real letters, for training the checker only.

    Each letter keeps its real height above the baseline and is set beside the previous
    one, lifted from the paper in between. The result reads like careful printing; the
    writing model never sees it. Half the texts are real words this writer never wrote,
    half are letter chains no dictionary holds, so the checker can't lean on memory.
    """

    def __init__(self, lines, seed=3):
        self.rng = random.Random(seed)
        self.pools = defaultdict(lambda: defaultdict(list))
        follows = defaultdict(Counter)
        starts = Counter()
        lengths = Counter()
        self.vocab = sorted({tuple(l.text.tolist()) for l in lines})
        for line in lines:
            ids = line.text.tolist()
            starts[ids[0]] += 1
            lengths[len(ids)] += 1
            for a, b in zip(ids, ids[1:]):
                follows[a][b] += 1
            xy = torch.cumsum(line.steps[:, :2], 0)
            for pos, c in enumerate(ids):
                keep = line.char == pos
                if 2 <= int(keep.sum()) <= 60:
                    seg = xy[keep].clone()
                    seg[:, 0] -= seg[:, 0].min()
                    pen = line.steps[keep, 2].clone()
                    pen[-1] = 1.0
                    self.pools[line.writer][c].append((seg, pen))
        self.follows = {a: (list(c), list(c.values())) for a, c in follows.items()}
        self.starts = (list(starts), list(starts.values()))
        self.lengths = (list(lengths), list(lengths.values()))
        self.writers = list(self.pools)

    def make_text(self):
        if self.rng.random() < 0.5:
            return list(self.rng.choice(self.vocab))
        n = self.rng.choices(*self.lengths)[0]
        out = [self.rng.choices(*self.starts)[0]]
        while len(out) < n:
            next_options = self.follows.get(out[-1])
            if not next_options:
                break
            out.append(self.rng.choices(*next_options)[0])
        return out

    def make(self):
        for _ in range(20):
            pool = self.pools[self.rng.choice(self.writers)]
            ids = self.make_text()
            if not all(c in pool for c in ids):
                continue
            parts, cursor = [], 0.0
            for c in ids:
                seg, pen = self.rng.choice(pool[c])
                placed = seg.clone()
                placed[:, 0] += cursor + (self.rng.uniform(0.1, 0.9) if parts else 0.0)
                parts.append((placed, pen))
                cursor = float(placed[:, 0].max())
            xy = torch.cat([p for p, _ in parts])
            pen = torch.cat([q for _, q in parts])
            steps = torch.zeros(len(xy), 3)
            steps[0, 1] = xy[0, 1]
            steps[1:, :2] = xy[1:] - xy[:-1]
            steps[:, 2] = pen
            return steps, torch.tensor(ids)
        return None

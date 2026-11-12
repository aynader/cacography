"""Turn the packed BRUSH arrays into per-line training tensors."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCALE = 4.0  # pixels per model unit
Y_ORIGIN = 70.0  # roughly mid-line in BRUSH's writing box
HOLDOUT_SEED = 7
HOLDOUT_COUNT = 16
VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 is padding


def encode(text):
    return torch.tensor([CHAR_TO_ID[c] for c in text], dtype=torch.long)


def writer_split(writers_json: Path):
    """Natural writers, and the subset never shown to the model (real test side)."""
    natural = json.loads(writers_json.read_text(encoding="utf-8"))["natural"]
    rng = np.random.default_rng(HOLDOUT_SEED)
    holdout = sorted(rng.choice(natural, HOLDOUT_COUNT, replace=False).tolist())
    return natural, holdout


def fix_labels(label, sentence):
    """Give any point labelled as a space to the neighbouring written character."""
    label = label.copy()
    spaces = {i for i, c in enumerate(sentence) if c == " "}
    if not spaces.intersection(label.tolist()):
        return label
    last = None
    for t in range(len(label)):
        if label[t] in spaces:
            if last is not None:
                label[t] = last
        else:
            last = label[t]
    first = next((v for v in label if v not in spaces), None)
    label[np.isin(label, list(spaces))] = first
    return label


@dataclass
class Line:
    text: torch.Tensor  # character ids
    steps: torch.Tensor  # (T, 3) dx, dy, pen-up in model units
    char: torch.Tensor  # (T) index of the character each point belongs to
    frames: torch.Tensor  # (T) points so far within this character
    lifts: torch.Tensor  # (T) pen lifts so far within this character
    advance: torch.Tensor  # (T) 1 when the next point moves to the next character
    writer: int


def make_line(points, label, sentence, writer):
    label = fix_labels(label, sentence)
    xy = points[:, :2].astype(np.float64)
    pen = (points[:, 2] > 0.5).astype(np.float32)
    delta = np.zeros_like(xy)
    delta[1:] = xy[1:] - xy[:-1]
    delta[0, 1] = xy[0, 1] - Y_ORIGIN
    steps = np.concatenate([delta / SCALE, pen[:, None]], 1).astype(np.float32)
    T = len(label)
    frames = np.zeros(T, np.int64)
    lifts = np.zeros(T, np.int64)
    for t in range(1, T):
        if label[t] == label[t - 1]:
            frames[t] = frames[t - 1] + 1
            lifts[t] = lifts[t - 1] + int(pen[t - 1] > 0.5)
    advance = np.ones(T, np.float32)
    advance[:-1] = (label[1:] != label[:-1]).astype(np.float32)
    return Line(
        encode(sentence),
        torch.from_numpy(steps),
        torch.from_numpy(label.astype(np.int64)),
        torch.from_numpy(frames),
        torch.from_numpy(lifts),
        torch.from_numpy(advance),
        writer,
    )


def load_lines(cache: Path, writers_json: Path, include_holdout=False):
    z = np.load(cache)
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    _, holdout = writer_split(writers_json)
    offs, writers = z["offsets"], z["writers"]
    points, labels = z["points"], z["labels"]
    out = []
    for i, sentence in enumerate(meta["sentences"]):
        w = int(writers[i])
        if (w in holdout) != include_holdout:
            continue
        if any(c not in CHAR_TO_ID for c in sentence):
            continue
        out.append(make_line(points[offs[i] : offs[i + 1]], labels[offs[i] : offs[i + 1]], sentence, w))
    return out


def collate(batch, writer_index, device):
    n = len(batch)
    T = max(len(b.steps) for b in batch)
    L = max(len(b.text) for b in batch)
    text = torch.zeros(n, L, dtype=torch.long)
    steps = torch.zeros(n, T, 3)
    char = torch.zeros(n, T, dtype=torch.long)
    frames = torch.zeros(n, T, dtype=torch.long)
    lifts = torch.zeros(n, T, dtype=torch.long)
    advance = torch.zeros(n, T)
    mask = torch.zeros(n, T)
    for i, b in enumerate(batch):
        t = len(b.steps)
        text[i, : len(b.text)] = b.text
        steps[i, :t] = b.steps
        char[i, :t] = b.char
        frames[i, :t] = b.frames
        lifts[i, :t] = b.lifts
        advance[i, :t] = b.advance
        mask[i, :t] = 1
    writer = torch.tensor([writer_index[b.writer] for b in batch], dtype=torch.long)
    return tuple(x.to(device, non_blocking=True) for x in (text, steps, char, frames, lifts, advance, mask, writer))

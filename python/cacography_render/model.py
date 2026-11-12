"""The handwriting model, its checker and the shape check, for inference only.

The layers match handlab/handmodel.py and handlab/train_reader.py exactly, so trained
weights load unchanged. Sampling draws its random numbers from each word's own seed,
which makes a word's writing independent of whatever else shares its batch: the same
seed gives the same page on every run.
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB)}
SCALE = 4.0  # working pixels per model unit
XH = 16.0  # x-height of the training words in working pixels
MIXTURES = 20
GEO_SCALE = 10.0
NOISE = 5  # random numbers per step: mixture pick, two normals, pen lift, moving on


def encode(text):
    return [CHAR_TO_ID[c] for c in text]


class HandModel(nn.Module):
    def __init__(self, writers, hidden=512, char_dim=128, style_dim=64, layers=3, encoder_dim=256):
        super().__init__()
        self.dropout = nn.Dropout(0.0)
        self.char_embed = nn.Embedding(len(VOCAB) + 1, char_dim, padding_idx=0)
        self.encoder = nn.LSTM(char_dim, encoder_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.char_proj = nn.Linear(encoder_dim, char_dim)
        self.style = nn.Embedding(writers, style_dim)
        self.frame_embed = nn.Embedding(48, 16)
        self.lift_embed = nn.Embedding(6, 8)
        in_dim = 3 + 4 + char_dim + style_dim + 16 + 8
        self.rnns = nn.ModuleList(nn.LSTM(in_dim + (hidden if i else 0), hidden, batch_first=True) for i in range(layers))
        self.out = nn.Linear(hidden * layers, MIXTURES * 6 + 2)

    def encode_text(self, text):
        h, _ = self.encoder(self.char_embed(text))
        return self.char_proj(h)

    def features(self, prev, geo, char_h, style, frames, lifts):
        return torch.cat(
            [prev, geo, char_h, style, self.frame_embed(frames.clamp(max=47)), self.lift_embed(lifts.clamp(max=5))], -1
        )

    def decode(self, x, state=None):
        new_state, outs, h = [], [], None
        for i, rnn in enumerate(self.rnns):
            inp = x if h is None else torch.cat([x, h], -1)
            h, st = rnn(inp, None if state is None else state[i])
            outs.append(h)
            new_state.append(st)
        return self.out(torch.cat(outs, -1)), new_state


def split_params(out):
    pi, mu_x, mu_y, ls_x, ls_y, rho = out[..., : MIXTURES * 6].chunk(6, -1)
    return (
        F.log_softmax(pi, -1), mu_x, mu_y, ls_x.clamp(-7, 5), ls_y.clamp(-7, 5),
        torch.tanh(rho) * 0.999, out[..., -2], out[..., -1],
    )


@torch.no_grad()
def sample(model, texts, style, noise, bias, lift_sharpen=1.0):
    """Write words.

    texts: list of strings in VOCAB. style: (N, style_dim). noise: (N, T, NOISE) with
    uniform numbers in columns 0, 3, 4 and standard normals in 1 and 2; T caps the steps.
    lift_sharpen above 1 makes the pen's lift-or-stay decision less of a coin flip.
    Returns per word (steps (T_i, 3) in model units, char index (T_i) of each step).
    """
    device = style.device
    n, T = len(texts), noise.shape[1]
    L = max(len(t) for t in texts)
    text_ids = torch.zeros(n, L, dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        text_ids[i, : len(t)] = torch.tensor(encode(t), device=device)
    enc = model.encode_text(text_ids)
    written = (text_ids != 0) & (text_ids != CHAR_TO_ID[" "])
    ptr = written.float().argmax(1)
    following = torch.full((n, L + 1), L, dtype=torch.long, device=device)
    for j in range(L - 2, -1, -1):
        following[:, j] = torch.where(written[:, j + 1], torch.full_like(following[:, j], j + 1), following[:, j + 1])
    prev = torch.zeros(n, 1, 3, device=device)
    frames = torch.zeros(n, dtype=torch.long, device=device)
    lifts = torch.zeros(n, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    pos = torch.zeros(n, 2, device=device)
    xmax = torch.zeros(n, device=device)
    origin = torch.zeros(n, 2, device=device)
    buf = torch.zeros(T, n, 3, device=device)
    chars = torch.zeros(T, n, dtype=torch.long, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    noise = noise.to(device)
    state = None
    for t in range(T):
        u = noise[:, t]
        geo = torch.stack([pos[:, 0] - xmax, pos[:, 1], pos[:, 0] - origin[:, 0], pos[:, 1] - origin[:, 1]], -1)
        x = model.features(prev, geo.clamp(-40, 40)[:, None] / GEO_SCALE, enc[rows, ptr][:, None], style[:, None],
                           frames[:, None], lifts[:, None])
        out, state = model.decode(x, state)
        log_pi, mx, my, lsx, lsy, rho, pen, adv = split_params(out[:, 0])
        cdf = F.softmax(log_pi * (1 + bias), -1).cumsum(-1)
        comp = (cdf < u[:, 0:1]).sum(-1).clamp(max=MIXTURES - 1)[:, None]
        pick = lambda v: v.gather(1, comp)[:, 0]  # noqa: E731
        sx, sy, r = torch.exp(pick(lsx) - bias), torch.exp(pick(lsy) - bias), pick(rho)
        dx = pick(mx) + sx * u[:, 1]
        dy = pick(my) + sy * (r * u[:, 1] + torch.sqrt((1 - r * r).clamp_min(1e-6)) * u[:, 2])
        up = (u[:, 3] < torch.sigmoid(pen * lift_sharpen)).float()
        step = torch.stack([dx, dy, up], -1)
        pos = pos + step[:, :2]
        buf[t] = step
        chars[t] = ptr
        lengths += (~done).long()
        go = u[:, 4] < torch.sigmoid(adv)
        moving = go & ~done
        target = following.gather(1, ptr[:, None])[:, 0]
        finished = moving & (target >= L)
        done = done | finished
        ptr = torch.where(moving & ~finished, target, ptr)
        same = ~go
        lifts = torch.where(same, lifts + up.long(), torch.zeros_like(lifts))
        frames = torch.where(same, frames + 1, torch.zeros_like(frames))
        origin = torch.where(same[:, None], origin, pos)
        xmax = torch.maximum(xmax, pos[:, 0])
        prev = step[:, None]
        if t % 8 == 7 and bool(done.all()):
            break
    buf, chars = buf.cpu().numpy(), chars.cpu().numpy()
    out = []
    for i, n_i in enumerate(lengths.clamp_min(1).tolist()):
        s = buf[:n_i, i].astype(np.float64)
        s[-1, 2] = 1.0  # the pen always leaves the paper at the end of a word
        out.append((s, chars[:n_i, i].copy()))
    return out


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


@torch.no_grad()
def read(checker, words, device):
    """Decode a list of (T, 3) step arrays into strings."""
    T = max(len(w) for w in words)
    x = torch.zeros(len(words), T, 3, device=device)
    for i, w in enumerate(words):
        x[i, : len(w)] = torch.tensor(w, dtype=torch.float32)
    ids = checker(x).argmax(-1).cpu().numpy()
    out = []
    for row, w in zip(ids, words):
        seq, last = [], 0
        for v in row[: (len(w) + 1) // 2]:
            if v != last and v != 0:
                seq.append(VOCAB[v - 1])
            last = v
        out.append("".join(seq))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def end_change(reading, text):
    """Does reading `text` as `reading` take a letter added or dropped at either end?

    A sloppy letter misread is a writer's; a letter missing from the start of a word or a
    stroke after its end that reads as one more letter is the model's, and no hand's.
    """
    n, m = len(text), len(reading)
    if not n or not m:
        return n != m
    if m > n and (reading.startswith(text) or reading.endswith(text)):
        return True  # the whole word and more: "Kapitell", even though an l may pair with an l

    def distance(strict):
        far = 1000 if strict else 1
        prev = [far * j for j in range(m + 1)]  # letters read before the first written: added at the start
        for i in range(1, n + 1):
            cur = [far * i]  # letters written before the first read: dropped at the start
            for j in range(1, m + 1):
                add = 1 if not (strict and i == n) else far  # past the last written letter
                drop = 1 if not (strict and j == m) else far  # past the last letter read
                cur.append(min(prev[j] + drop, cur[-1] + add, prev[j - 1] + (text[i - 1] != reading[j - 1])))
            prev = cur
        return prev[-1]

    return distance(True) > distance(False)


def shape_ok(steps, n_chars, limits, column=2.0):
    """Reject scribbles the checker can still read: see handlab/shapecheck.py."""
    xy = np.cumsum(steps[:, :2], 0)
    width = max(float(np.ptp(xy[:, 0])), 1e-3)
    seg = np.hypot(*np.diff(xy, axis=0).T) * (steps[:-1, 2] < 0.5)
    mid = 0.5 * (xy[1:, 0] + xy[:-1, 0])
    if not len(mid):
        return False
    bins = np.floor((mid - xy[:, 0].min()) / column).astype(int)
    per_col = np.bincount(bins, weights=seg)
    used = per_col[per_col > 0]
    m = {
        "ink_per_width": float(seg.sum()) / width,
        "width_per_char": width / max(n_chars, 1),
        "peak_column": float(used.max() / np.median(used)) if len(used) else 0.0,
    }
    return all(limits[k][0] <= m[k] <= limits[k][1] for k in limits)


"""Character-locked handwriting generator: an LSTM with a mixture-density output.

Instead of learning where to read in the text by soft attention, the model is told
exactly which character it is writing and predicts when to move on. That removes the
classic failure where attention slips and letters get skipped or repeated.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from handdata import VOCAB

MIXTURES = 20
LOG_2PI = math.log(2 * math.pi)


GEO_SCALE = 10.0


def geometry(steps, frames):
    """Where the pen is before each step: (T, 4) per line, batched as (N, T, 4).

    Holds distance past the rightmost ink so far, height, and offset from where the
    current character began. Knowing these stops the writing running over itself,
    and it stays true however errors build up while the model writes freely.
    """
    pos = torch.cumsum(steps[..., :2], 1)
    prev = torch.zeros_like(pos)
    prev[:, 1:] = pos[:, :-1]
    xmax = torch.cummax(prev[..., 0], 1).values
    n, t = frames.shape
    index = torch.arange(t, device=frames.device)[None].expand(n, -1)
    start = torch.cummax(torch.where(frames == 0, index, torch.zeros_like(index)), 1).values
    origin = prev.gather(1, start[..., None].expand(-1, -1, 2))
    return torch.stack(
        [prev[..., 0] - xmax, prev[..., 1], prev[..., 0] - origin[..., 0], prev[..., 1] - origin[..., 1]], -1
    ).clamp(-40, 40) / GEO_SCALE


class HandModel(nn.Module):
    def __init__(
        self, writers, hidden=512, char_dim=128, style_dim=64, layers=3, encoder_dim=256,
        dropout=0.15, input_noise=0.06, style_noise=0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_noise = input_noise
        self.style_noise = style_noise
        self.char_embed = nn.Embedding(len(VOCAB) + 1, char_dim, padding_idx=0)
        self.encoder = nn.LSTM(char_dim, encoder_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.char_proj = nn.Linear(encoder_dim, char_dim)
        self.style = nn.Embedding(writers, style_dim)
        nn.init.normal_(self.style.weight, std=0.5)
        self.frame_embed = nn.Embedding(48, 16)
        self.lift_embed = nn.Embedding(6, 8)
        in_dim = 3 + 4 + char_dim + style_dim + 16 + 8
        self.rnns = nn.ModuleList(
            nn.LSTM(in_dim + (hidden if i else 0), hidden, batch_first=True) for i in range(layers)
        )
        self.out = nn.Linear(hidden * layers, MIXTURES * 6 + 2)
        self.style_dim = style_dim

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
            h = self.dropout(h)
            outs.append(h)
            new_state.append(st)
        return self.out(torch.cat(outs, -1)), new_state

    def forward(self, text, steps, char, frames, lifts, writer):
        enc = self.encode_text(text)
        char_h = enc.gather(1, char[..., None].expand(-1, -1, enc.size(-1)))
        prev = torch.zeros_like(steps)
        prev[:, 1:] = steps[:, :-1]
        geo = geometry(steps, frames)
        style = self.style(writer)
        if self.training:
            # Noise teaches recovery from the small errors free-running writing makes,
            # and keeps the space between real writers usable for new styles.
            prev[..., :2] = prev[..., :2] + self.input_noise * torch.randn_like(prev[..., :2])
            style = style + self.style_noise * torch.randn_like(style)
        style = style[:, None].expand(-1, steps.size(1), -1)
        out, _ = self.decode(self.features(prev, geo, char_h, style, frames, lifts))
        return out


def split_params(out):
    pi, mu_x, mu_y, ls_x, ls_y, rho = out[..., : MIXTURES * 6].chunk(6, -1)
    return (
        F.log_softmax(pi, -1),
        mu_x,
        mu_y,
        ls_x.clamp(-7, 5),
        ls_y.clamp(-7, 5),
        torch.tanh(rho) * 0.999,
        out[..., -2],
        out[..., -1],
    )


def losses(out, steps, advance, mask):
    log_pi, mx, my, lsx, lsy, rho, pen, adv = split_params(out)
    x = (steps[..., 0:1] - mx) * torch.exp(-lsx)
    y = (steps[..., 1:2] - my) * torch.exp(-lsy)
    z = x * x + y * y - 2 * rho * x * y
    one_minus = (1 - rho * rho).clamp_min(1e-6)
    comp = -0.5 * z / one_minus - lsx - lsy - LOG_2PI - 0.5 * torch.log(one_minus)
    nll = -torch.logsumexp(log_pi + comp, -1)
    pen_loss = F.binary_cross_entropy_with_logits(pen, steps[..., 2], reduction="none")
    adv_loss = F.binary_cross_entropy_with_logits(adv, advance, reduction="none")
    total = mask.sum().clamp_min(1)
    return (nll * mask).sum() / total, (pen_loss * mask).sum() / total, (adv_loss * mask).sum() / total


@torch.no_grad()
def sample(model, text_ids, style, bias=1.0, pen_bias=0.0, max_steps=1400, generator=None):
    """Write lines. text_ids: (N, L) with 0 padding; style: (N, style_dim).

    Returns one (T, 3) tensor of model-unit steps per line.
    """
    device = text_ids.device
    n = text_ids.size(0)
    enc = model.encode_text(text_ids)
    space_id = 1  # VOCAB[0] is the space
    written = (text_ids != 0) & (text_ids != space_id)
    ptr = written.float().argmax(1)
    prev = torch.zeros(n, 1, 3, device=device)
    frames = torch.zeros(n, dtype=torch.long, device=device)
    lifts = torch.zeros(n, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    state = None
    pos = torch.zeros(n, 2, device=device)
    xmax = torch.zeros(n, device=device)
    origin = torch.zeros(n, 2, device=device)
    L = text_ids.size(1)
    # following[i, j]: the next written character after position j, or L when none is left.
    following = torch.full((n, L + 1), L, dtype=torch.long, device=device)
    for j in range(L - 2, -1, -1):
        following[:, j] = torch.where(written[:, j + 1], torch.full_like(following[:, j], j + 1), following[:, j + 1])
    buf = torch.zeros(max_steps, n, 3, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    for t in range(max_steps):
        char_h = enc[rows, ptr][:, None]
        geo = torch.stack(
            [pos[:, 0] - xmax, pos[:, 1], pos[:, 0] - origin[:, 0], pos[:, 1] - origin[:, 1]], -1
        ).clamp(-40, 40)[:, None] / GEO_SCALE
        x = model.features(prev, geo, char_h, style[:, None], frames[:, None], lifts[:, None])
        out, state = model.decode(x, state)
        log_pi, mx, my, lsx, lsy, rho, pen, adv = split_params(out[:, 0])
        comp = torch.multinomial(F.softmax(log_pi * (1 + bias), -1), 1, generator=generator)
        pick = lambda v: v.gather(1, comp)[:, 0]  # noqa: E731
        sx, sy, r = torch.exp(pick(lsx) - bias), torch.exp(pick(lsy) - bias), pick(rho)
        e1 = torch.randn(n, device=device, generator=generator)
        e2 = torch.randn(n, device=device, generator=generator)
        dx = pick(mx) + sx * e1
        dy = pick(my) + sy * (r * e1 + torch.sqrt((1 - r * r).clamp_min(1e-6)) * e2)
        up = (torch.rand(n, device=device, generator=generator) < torch.sigmoid(pen + pen_bias)).float()
        step = torch.stack([dx, dy, up], -1)
        pos = pos + step[:, :2]
        buf[t] = step
        lengths += (~done).long()
        go = torch.rand(n, device=device, generator=generator) < torch.sigmoid(adv)
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
    lengths = lengths.clamp_min(1).tolist()
    return [buf[: lengths[i], i] for i in range(n)]

"""Image helpers shared by the paper and the camera: colour, blur, noise, resampling.

Everything here works on float32 arrays in linear light unless it says otherwise.
Heavy filtering goes through torch on the CPU, which is fast and gives the same
result on every run.
"""
import numpy as np
import torch
import torch.nn.functional as F


def to_linear(srgb):
    """sRGB in 0..1 to linear light."""
    s = np.asarray(srgb, np.float32)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(lin):
    """Linear light to sRGB in 0..1."""
    v = np.clip(lin, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055).astype(np.float32)


def rgb(colour):
    """An 8-bit sRGB triple as linear light."""
    return to_linear(np.asarray(colour, np.float32) / 255.0)


def _tensor(a):
    a = np.ascontiguousarray(a, np.float32)
    t = torch.from_numpy(a)
    return t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[:, None]


def _array(t, ndim):
    return t[0, 0].numpy() if ndim == 2 else t[:, 0].permute(1, 2, 0).contiguous().numpy()


def _gauss(t, sigma):
    r = max(1, int(3 * sigma + 0.5))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = torch.from_numpy((k / k.sum()).astype(np.float32))
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def blur(a, sigma):
    """Gaussian blur of an HxW or HxWxC array; wide blurs run on a smaller copy."""
    if sigma < 0.25:
        return a
    t = _tensor(a)
    h, w = t.shape[-2:]
    step = int(sigma // 3)
    if step >= 2:
        small = F.avg_pool2d(t, step, ceil_mode=True)
        small = _gauss(small, sigma / step)
        t = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    else:
        t = _gauss(t, sigma)
    return _array(t, np.ndim(a))


def resize(a, shape, mode="bilinear"):
    t = _tensor(a)
    t = F.interpolate(t, size=shape, mode=mode, align_corners=False)
    return _array(t, np.ndim(a))


def band(shape, rng, shortest, longest, slope=1.0, stretch=1.0, angle=0.0):
    """Unit-variance noise with wavelengths between `shortest` and `longest` pixels.

    `slope` tilts the spectrum towards long waves (1 is pink noise). `stretch` above 1
    elongates features along `angle` (radians), the way paper fibres line up along
    the direction the paper machine ran.
    """
    h, w = shape
    step = max(1, int(shortest // 4))  # nothing finer than `shortest`: synthesise small, then enlarge
    sh, sw = -(-h // step), -(-w // step)
    fy = np.fft.fftfreq(sh)[:, None] / step
    fx = np.fft.rfftfreq(sw)[None, :] / step
    c, s = np.cos(angle), np.sin(angle)
    along, across = fx * c + fy * s, -fx * s + fy * c
    f = np.hypot(along * stretch, across)
    lo, hi = 1.0 / longest, 1.0 / shortest
    window = 1 / (1 + (lo / np.maximum(f, 1e-9)) ** 4) / (1 + (f / hi) ** 4)
    amp = window / np.maximum(f, lo) ** slope
    amp[0, 0] = 0.0
    spec = np.fft.rfft2(rng.standard_normal((sh, sw)).astype(np.float32)) * amp
    out = np.fft.irfft2(spec, s=(sh, sw)).astype(np.float32)
    out /= out.std() + 1e-12
    if step > 1:
        out = resize(out, (sh * step, sw * step), mode="bicubic")[:h, :w]
    return np.ascontiguousarray(out)


def wave(n, rng, shortest, longest):
    """1-D unit-variance noise, for things that vary along a line."""
    return band((1, n), rng, shortest, longest)[0] if n > 1 else np.zeros(n, np.float32)


def sample(a, x, y, fill=None):
    """Bilinear lookup of `a` (HxW or HxWxC) at pixel coordinates `x`, `y`.

    Outside the array, the edge value is repeated, or `fill` is used when given.
    """
    t = _tensor(a).permute(1, 0, 2, 3)  # 1 x C x H x W
    h, w = t.shape[-2:]
    x, y = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32))
    shape = x.shape
    x, y = x.reshape(1, -1), y.reshape(1, -1)
    grid = np.stack([2 * x / max(w - 1, 1) - 1, 2 * y / max(h - 1, 1) - 1], -1).astype(np.float32)
    g = torch.from_numpy(grid)[None]
    if fill is None:
        out = F.grid_sample(t, g, mode="bilinear", padding_mode="border", align_corners=True)
    else:
        out = F.grid_sample(t - fill, g, mode="bilinear", padding_mode="zeros", align_corners=True) + fill
    out = out[0, :, 0].numpy()  # C x N
    return out[0].reshape(shape) if np.ndim(a) == 2 else out.T.reshape(*shape, -1)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)

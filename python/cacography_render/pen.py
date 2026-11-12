"""One writer for a whole document: a style chosen by the seed, writing checked words.

The seed blends two real writers' styles into a new person, who writes every word of
the document. Each word gets its own seed from the document seed, its text and how
many times that text has appeared before, so editing one paragraph leaves the writing
everywhere else unchanged. Finished words are cached on disk.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .marks import add_dots, faults, smallest_advance, strip_strays, tidy
from .model import NOISE, HandModel, Reader, edit_distance, end_change, read, sample, shape_ok

MODEL_TAG = "pen-15"  # change whenever the pen code should write the same seed differently
TRIES = 8
# Hard-to-read cursive is fine, as real cursive often is. Broken writing is not: a fault
# outranks any number of misread letters, and only a word free of faults can be accepted.
TOLERANCE = 0.35  # share of characters the checker may misread
FAULT = 1000
BATCH = 64
FINAL = ".,;:!?"
MIN_ADVANCE = 1.0  # model units between neighbouring letters' centres, real writing's 1st percentile
FIRST_IAM_WRITER = 10000  # BRUSH writers' styles carry skipped dots and chopped strokes, so they don't write
GAP_RANGE, GAP_SPREAD = (20.0, 34.0), 8.0
LIFT_SHARPEN = 4.0  # an unlucky lift mid-stroke reads as a pen running dry; this brings breaks to real levels


# Alone, these can't be told apart by shape, so the checker reading one as another is no error.
LOOKALIKES = ["0oO", "1lI|", "5sS", "2zZ", "pP", "cC", "kK", "vV", "wW", "xX", "uU", "9g", "6b"]


def judge(steps, chars, text, reading, limits, size):
    """What the pen holds against one attempt at `text`: (score, reasons).

    A score within len(text) * TOLERANCE keeps the attempt. A single character is judged
    by what it is alone: a lone letter has no word shape to compare with, and look-alikes
    such as 0 and o are the same shape.
    """
    reasons = []
    misread = edit_distance(reading, text)
    if len(text) == 1 and len(reading) == 1 and any(text in group and reading in group for group in LOOKALIKES):
        misread = 0
    score = misread
    if misread > int(len(text) * TOLERANCE):
        reasons.append("misread")
    if len(text) > 1 and any(c.isalpha() for c in text) and end_change(reading, text):
        # Tolerance is for sloppy letters, not for a letter lost or added at an end.
        score += FAULT
        reasons.append("letter added or dropped")
    if len(text) > 1 and not shape_ok(steps, len(text), limits):
        score += FAULT
        reasons.append("shape")
    advance = smallest_advance(steps, chars, text)
    if advance is not None and advance * size < MIN_ADVANCE:
        score += FAULT
        reasons.append("too close")
    found = faults(steps, chars, text, size)
    score += FAULT * len(found)
    reasons += found
    # Tolerance is for sloppy letters. Punctuation it would simply lose.
    if text[-1] in FINAL and reading[-1:] != text[-1]:
        score += len(text)
        reasons.append("lost punctuation")
    return score, reasons


def subseed(*parts):
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class Pen:
    def __init__(self, models: Path, seed: int, sloppiness: float, cache: Path, device="cpu"):
        self.device = device
        cfg = json.loads((models / "hand" / "config.json").read_text(encoding="utf-8"))
        self.model = HandModel(len(cfg["writer_ids"]), hidden=cfg["hidden"]).to(device).eval()
        state = torch.load(models / "hand" / "weights.pt", map_location=device, weights_only=True)["model"]
        self.model.load_state_dict(state)
        self.checker = Reader().to(device).eval()
        self.checker.load_state_dict(torch.load(models / "checker" / "weights.pt", map_location=device, weights_only=True)["model"])
        self.limits = json.loads((models / "checker" / "shape_limits.json").read_text(encoding="utf-8"))
        info = json.loads((models / "hand" / "writers.json").read_text(encoding="utf-8"))

        rng = np.random.default_rng(subseed(seed, "writer"))
        index = {w: i for i, w in enumerate(cfg["writer_ids"])}
        picked = [int(w) for w in rng.choice([w for w in cfg["style_writers"] if w >= FIRST_IAM_WRITER], 2, replace=False)]
        weights = rng.dirichlet([1.0, 1.0])
        with torch.no_grad():
            rows = self.model.style.weight[[index[w] for w in picked]]
            self.style = (rows * torch.tensor(weights, dtype=torch.float32, device=device)[:, None]).sum(0)
        gaps = np.array([info["gap"].get(str(w), [29.0, 6.0]) for w in picked])
        self.gap = (gaps * weights[:, None]).sum(0)  # word gap (median, spread) in working pixels
        # TeX broke the lines for gaps of about 29 pixels: wider ones run off the paper.
        self.gap = np.array([np.clip(self.gap[0], *GAP_RANGE), min(self.gap[1], GAP_SPREAD)])
        slants = [info.get("slant", {}).get(str(w)) for w in picked]
        self.slant = None if None in slants else float(np.dot(slants, weights))
        self.writers = list(zip(picked, weights.round(3).tolist()))
        self.seed = seed
        self.bias = max(0.0, 1.0 - sloppiness)
        # Everything that decides which attempt is kept: a new model never reuses old words.
        made_by = hashlib.md5()
        for part in ("hand/weights.pt", "checker/weights.pt", "checker/shape_limits.json", "hand/size.json"):
            if (models / part).exists():
                made_by.update((models / part).read_bytes())
        self.tag = f"{MODEL_TAG}|{made_by.hexdigest()[:12]}|{seed}|{self.bias:.3f}"
        # Biased sampling suppresses long pen steps, so writing comes out small: see handlab/calibrate_size.py.
        size_path = models / "hand" / "size.json"
        self.size = 1.0
        if size_path.exists():
            curve = json.loads(size_path.read_text(encoding="utf-8"))
            self.size = float(np.interp(self.bias, curve["bias"], curve["scale"]))
        self.cache_path = cache
        self.cache = {}
        if cache.exists():
            with np.load(cache) as z:
                self.cache = {k: z[k] for k in z.files}
        self.used = set()
        self.stats = {"words": 0, "cached": 0, "tries": 0, "failed": 0}

    def _noise(self, key, attempt, steps):
        rng = np.random.default_rng(subseed(self.tag, key, attempt))
        noise = rng.random((steps, NOISE))
        noise[:, 1:3] = rng.standard_normal((steps, 2))
        return noise

    def _name(self, key):
        return hashlib.sha1(f"{self.tag}|{key}".encode("utf-8")).hexdigest()

    def keep(self, requests):
        """Words on a page that is not redrawn: still in use, so they stay in the cache."""
        self.used.update(self._name(key) for key, _ in requests)

    def write(self, requests):
        """requests: list of (key, text). Returns {key: (steps, chars)} in model units."""
        out, todo = {}, []
        for key, text in requests:
            name = self._name(key)
            self.used.add(name)
            if name in self.cache:
                arr = self.cache[name]
                out[key] = (arr[:, :3], arr[:, 3].astype(np.int64))
                self.stats["cached"] += 1
            else:
                todo.append((key, text, name))
        for start in range(0, len(todo), BATCH):
            self._write_batch(todo[start : start + BATCH], out)
        self.stats["words"] += len(requests)
        return out

    def _write_batch(self, batch, out):
        best = {}
        pending = list(batch)
        for attempt in range(TRIES):
            if not pending:
                break
            T = 45 * max(len(t) for _, t, _ in pending) + 25
            noise = torch.tensor(np.stack([self._noise(k, attempt, T) for k, _, _ in pending]), dtype=torch.float32)
            style = self.style[None].expand(len(pending), -1)
            texts = [t for _, t, _ in pending]
            written = sample(self.model, texts, style, noise, self.bias, LIFT_SHARPEN)
            made = [tidy(s, c, t) for (s, c), t in zip(written, texts)]
            readings = read(self.checker, [s for s, _ in made], self.device)
            still = []
            for (key, text, name), (steps, chars), reading in zip(pending, made, readings):
                self.stats["tries"] += 1
                score, _ = judge(steps, chars, text, reading, self.limits, self.size)
                if key not in best or score < best[key][0]:
                    best[key] = (score, steps, chars, name)
                if score > int(len(text) * TOLERANCE):
                    still.append((key, text, name))
            pending = still
        self.stats["failed"] += len(pending)
        texts = {key: text for key, text, _ in batch}
        for key, (_, steps, chars, name) in best.items():
            # What the best attempt still gets wrong, the pen puts right: no stray reaches
            # the page, and a dot the model never wrote gets added.
            steps, chars = strip_strays(steps, chars, texts[key], force=True)
            steps, chars = add_dots(steps, chars, texts[key], np.random.default_rng(subseed(self.tag, key, "dots")))
            out[key] = (steps, chars)
            self.cache[name] = np.concatenate([steps, chars[:, None]], 1)

    def save(self):
        """Keep only the words this document still uses."""
        np.savez_compressed(self.cache_path, **{k: v for k, v in self.cache.items() if k in self.used})

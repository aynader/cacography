"""Math symbols and Greek letters written by real people.

The bank (see handlab/build_symbol_bank.py) holds real samples from MathWriting, which
keeps no record of who wrote what, so samples picked freely would give one page many
hands. But a person writes the same psi every time. So a document settles on one sample
per symbol, leans it the way the writer leans, and varies it a little with each use.
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from .pen import subseed

MIN_SAMPLES = 5  # a symbol with fewer than this falls back to the font
DIGITS = set("0123456789")  # the model sees few digits, and its 1 can pass for a capital lambda

OPERATORS = {
    "ℏ": r"\hbar", "ħ": r"\hbar", "µ": r"\mu", "∂": r"\partial", "∇": r"\nabla", "∫": r"\int", "∬": r"\iint", "∮": r"\oint",
    "∑": r"\sum", "∏": r"\prod", "√": r"\sqrt", "∞": r"\infty", "≈": r"\approx", "≠": r"\ne", "≤": r"\le", "≥": r"\ge",
    "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "≅": r"\cong", "∝": r"\propto", "×": r"\times", "⋅": r"\cdot", "·": r"\cdot",
    "∙": r"\bullet", "•": r"\bullet", "÷": r"\div", "±": r"\pm", "∓": r"\mp", "→": r"\rightarrow", "←": r"\leftarrow",
    "↔": r"\leftrightarrow", "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow", "↦": r"\mapsto", "⟶": r"\longrightarrow",
    "↪": r"\hookrightarrow", "⇌": r"\rightleftharpoons", "∈": r"\in", "∉": r"\notin", "∋": r"\ni", "⊂": r"\subset",
    "⊃": r"\supset", "⊆": r"\subseteq", "⊇": r"\supseteq", "⊊": r"\subsetneq", "∪": r"\cup", "∩": r"\cap", "⋃": r"\bigcup",
    "⋂": r"\bigcap", "∧": r"\wedge", "∨": r"\vee", "⋀": r"\bigwedge", "⋁": r"\bigvee", "¬": r"\neg", "∀": r"\forall",
    "∃": r"\exists", "∅": r"\emptyset", "⊕": r"\oplus", "⊖": r"\ominus", "⊗": r"\otimes", "⊙": r"\odot", "⨁": r"\bigoplus",
    "∘": r"\circ", "†": r"\dagger", "⊥": r"\perp", "⊤": r"\top", "⊢": r"\vdash", "⊨": r"\models", "⊩": r"\Vdash",
    "∠": r"\angle", "⋮": r"\vdots", "⟨": r"\langle", "⟩": r"\rangle", "|": "|", "∣": "|", "‖": r"\|", "∥": r"\|",
    "⌊": r"\lfloor", "⌋": r"\rfloor", "⌈": r"\lceil", "⌉": r"\rceil", "≪": r"\ll", "≫": r"\gg", "≜": r"\triangleq",
    "∖": r"\backslash", "ℵ": r"\aleph",
}
SMALL_GREEK = {
    "ALPHA": "alpha", "BETA": "beta", "GAMMA": "gamma", "DELTA": "delta", "EPSILON": "epsilon", "ZETA": "zeta", "ETA": "eta",
    "THETA": "theta", "IOTA": "iota", "KAPPA": "kappa", "LAMDA": "lambda", "MU": "mu", "NU": "nu", "XI": "xi", "PI": "pi",
    "RHO": "rho", "SIGMA": "sigma", "FINAL SIGMA": "varsigma", "TAU": "tau", "UPSILON": "upsilon", "PHI": "varphi", "CHI": "chi",
    "PSI": "psi", "OMEGA": "omega",
}
# Unicode's "symbol" forms: \phi is the straight one, and its plain small phi is LaTeX's \varphi.
GREEK_SYMBOLS = {"PHI": "phi", "THETA": "vartheta", "EPSILON": "epsilon", "PI": "varpi", "RHO": "rho", "KAPPA": "kappa"}
CAPITAL_GREEK = {
    "GAMMA": "Gamma", "DELTA": "Delta", "THETA": "Theta", "LAMDA": "Lambda", "XI": "Xi", "PI": "Pi", "SIGMA": "Sigma",
    "UPSILON": "Upsilon", "PHI": "Phi", "PSI": "Psi", "OMEGA": "Omega",
}


def label_of(ch):
    """The MathWriting label for a character TeX typeset, or None if it has none."""
    if ch in OPERATORS:
        return OPERATORS[ch]
    plain = unicodedata.normalize("NFKC", ch)
    if plain in DIGITS:
        return plain
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    m = re.search(r"DOUBLE-STRUCK CAPITAL ([A-Z])$", name)
    if m:
        return rf"\mathbb{{{m.group(1)}}}"
    word = None
    if "GREEK" in name or "MATHEMATICAL" in name:
        m = re.search(r"([A-Z]+) SYMBOL$", name)
        if m:
            word = GREEK_SYMBOLS.get(m.group(1))
        else:
            m = re.search(r"(SMALL|CAPITAL)(?: LETTER)? (FINAL SIGMA|[A-Z]+)$", name)
            if m:
                word = (SMALL_GREEK if m.group(1) == "SMALL" else CAPITAL_GREEK).get(m.group(2))
    if word:
        return "\\" + word
    # Math alphabets have their own partial and nabla, which plain forms stand in for.
    return OPERATORS.get(unicodedata.normalize("NFKC", ch))


class SymbolBank:
    def __init__(self, folder: Path, seed: int, writer_slant=None):
        with np.load(folder / "bank.npz") as z:
            self.points, self.offsets = z["points"], z["offsets"]
        info = json.loads((folder / "bank.json").read_text(encoding="utf-8"))
        self.by_label = {}
        for i, label in enumerate(info["labels"]):
            self.by_label.setdefault(label, []).append(i)
        self.seed = seed
        self.lean = 0.0 if writer_slant is None else writer_slant - info["mean_slant"]

    def has(self, label):
        return len(self.by_label.get(label, ())) >= MIN_SAMPLES

    def write(self, label, key):
        """This use of a symbol: strokes as (N, 2) arrays, centred, about 1 across at most."""
        choices = self.by_label[label]
        pick = choices[np.random.default_rng(subseed(self.seed, "hand", label)).integers(len(choices))]
        ink = self.points[self.offsets[pick] : self.offsets[pick + 1]].astype(np.float64)
        rng = np.random.default_rng(subseed(self.seed, "use", key))
        x, y = ink[:, 0], ink[:, 1]
        angle = rng.normal(0, np.radians(1.5))
        c, s = np.cos(angle), np.sin(angle)
        grow = rng.normal(1.0, 0.03, 2)
        # A slow, small waver across the symbol, so no two uses trace the same line.
        waver = 0.012
        u = x * grow[0] + (self.lean + rng.normal(0, 0.03)) * y + waver * np.sin(2 * np.pi * (rng.uniform(0.5, 1.2) * y + rng.uniform(0, 1)))
        v = y * grow[1] + waver * np.sin(2 * np.pi * (rng.uniform(0.5, 1.2) * x + rng.uniform(0, 1)))
        p = np.stack([u * c - v * s, u * s + v * c], 1)
        out, start = [], 0
        for e in np.flatnonzero(ink[:, 2] > 0.5):
            out.append(p[start : e + 1])
            start = e + 1
        return out

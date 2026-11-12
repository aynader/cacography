"""What can be set about the scene, one aspect at a time, and the randomness behind each.

Every aspect draws its random numbers from its own seed, made from the document seed, the
aspect's name, the page (for aspects that differ from page to page) and how often the
user rerolled it. So changing or rerolling one aspect leaves all the others as they were:
a new camera angle keeps the same folds, the same desk and the same handwriting.

In LaTeX: \\usepackage[angle=steep, folds=half, reroll={light, stains=2}]{cacography}, and
for one page, \\cacographypage{crumple=0.8, stains=coffee}.
"""
import re

import numpy as np

from .pen import subseed

# name: (choices, or None for any value checked elsewhere; differs page to page?)
ASPECTS = {
    "background": (None, False),  # checked against surfaces.valid
    "light": (("random", "lamp", "overhead", "window", "soft", "raking", "flash", "dim"), False),
    "angle": (("random", "flat", "slight", "steep"), True),  # or a tilt in degrees
    "framing": (("random", "tight", "loose", "cropped"), True),  # how much desk round the sheet, or part cut off
    "shadow": (("random", "none", "phone"), True),
    "crumple": (("random", "none"), True),  # or how crumpled, 0 to 1
    "folds": (("random", "none", "half", "thirds", "quarters"), True),
    "creases": (("random", "none"), True),  # or a number of creases
    "scores": (("random", "none"), True),  # or a number of pressed lines
    "edges": (("random", "clean", "worn", "torn", "spiral"), True),
    "stains": (("random", "none", "coffee", "water", "smudge"), True),  # or several: coffee+smudge
    "grooves": (("random", "none"), True),  # or how hard the pen pressed in, 0 to 1
    "backside": (("random", "none", "writing"), True),  # writing on the back, showing through
    "curl": (("random", "none"), True),  # or how much the edges roll up, 0 to 1
    "age": (("random", "none"), True),  # or how old the sheet is, 0 to 1: yellowed, spotted, ink faded
    "props": (("random", "none", "tape", "clip", "binder", "staple", "pen", "sheet"), True),  # or several: tape+pen
    "paper": ((), True),  # the sheet's own texture, curl and grime: reroll only
    "camera": ((), False),  # the phone's character, and its noise on each page: reroll only
}


NUMBERS = ("angle", "creases", "crumple", "scores", "grooves", "curl", "age")
# What one page may set for itself with \cacographypage: how that sheet looks and is shot.
PER_PAGE = ("angle", "framing", "shadow", "crumple", "folds", "creases", "scores", "edges", "stains", "grooves", "backside", "curl", "age", "props")


def _number(text):
    return re.fullmatch(r"\s*[-+]?(\d+(\.\d*)?|\.\d+)\s*", text) is not None  # 1, 0.8, .8, 1.


def _valid(name, value, choices):
    if name in ("stains", "props"):
        parts = [p for p in re.split(r"[,+\s]+", value) if p]
        return bool(parts) and all(p in choices for p in parts) and (len(parts) == 1 or not {"random", "none"} & set(parts))
    return value in choices or (name in NUMBERS and _number(value))


def parse_options(text):
    """'crumple=0.8, stains={coffee, water}' as a dict, keeping braced commas together."""
    out, depth, key, buf = {}, 0, None, ""
    for ch in str(text) + ",":
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0 and key is None:
            key, buf = buf.strip(), ""
        elif ch == "," and depth == 0:
            if key is not None:
                out[key] = buf.strip()
            elif buf.strip():
                out[buf.strip()] = ""
            key, buf = None, ""
        else:
            buf += ch
    return out


class Aspects:
    def __init__(self, doc, seed):
        self.seed = seed
        self.values = {}
        for name, (choices, _) in ASPECTS.items():
            if choices == ():
                continue
            value = str(doc.get(name, "random")).strip() or "random"
            if choices and not _valid(name, value, choices):
                raise ValueError(f"{name}={value} (choose from {', '.join(choices)}" + (", or a number)" if name in NUMBERS else ")"))
            self.values[name] = value
        self.rerolls = {}
        for part in str(doc.get("reroll", "")).split(","):
            if not part.strip():
                continue
            name, _, times = part.partition("=")
            name = name.strip()
            if name not in ASPECTS:
                raise ValueError(f"reroll={name}: no such aspect (choose from {', '.join(ASPECTS)})")
            self.rerolls[name] = int(times) if times.strip() else 1

    def for_page(self, options):
        """These aspects with one page's own settings (from \\cacographypage) on top.

        A page may set or reroll only what belongs to its own sheet and shot (PER_PAGE);
        the room, desk and light stay the document's.
        """
        unknown = set(options) - set(PER_PAGE) - {"reroll", "wear"}
        if unknown:
            raise ValueError(f"{', '.join(sorted(unknown))} on one page (a page may set wear, {', '.join(PER_PAGE)}, or reroll them)")
        options = {k: v for k, v in options.items() if k != "wear"}
        if not options:
            return self
        page = Aspects({**self.values, **{k: v for k, v in options.items() if k != "reroll"}}, self.seed)
        page.rerolls = dict(self.rerolls)
        for name, times in parse_options(options.get("reroll", "")).items():
            if name not in PER_PAGE:
                raise ValueError(f"reroll={name} on one page (a page may reroll {', '.join(PER_PAGE)})")
            page.rerolls[name] = page.rerolls.get(name, 0) + (int(times) if times else 1)
        return page

    def value(self, name):
        return self.values.get(name, "random")

    def rng(self, name, page=None, part=None):
        """The random numbers for one aspect (on one page, if it differs between pages).

        `part` gives a separate stream within the aspect, for a part of it drawn elsewhere.
        """
        per_page = ASPECTS[name][1]
        keys = (self.seed, "aspect", name, page if per_page else None, self.rerolls.get(name, 0)) + ((part,) if part else ())
        return np.random.default_rng(subseed(*keys))

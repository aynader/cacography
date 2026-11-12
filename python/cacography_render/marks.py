"""What makes a written word wrong, whoever wrote it.

A word's small strokes are usually dots of i and j, punctuation, or short bars. Text says
which of those a word can have, and where. A mark the text cannot explain is a stray. Past
that: breaks where the pen seems to run dry, i and j without dots, and letters written on
top of each other. The same rules clean the training data and the model's output, and
score both.

Everything is in model units: the baseline is y = 0, y grows downward, and an x-height
is 4 units.
"""
import numpy as np

XH = 4.0
MARK = 1.8  # past a word's ends, a stroke up to this wide and tall counts as a mark
SPECK = 1.2  # within a word's span, only strokes up to this size do: bigger ones are letter parts
BAR = 0.6  # a speck longer than this could be a short bar; a dot can't be
CLEAR = 0.75  # closer than this to other ink, a mark doesn't show as a separate speck
NEAR = 0.75 * XH  # how far past the word's ink a late i-dot can land
HIGH = -0.75 * XH  # above this line a mark sits where dots and quotes do
FLOATING = 1.0  # a small high stroke this clear of all ink is no part of a letter
BREAK_GAP = 1.5  # a lift that picks up this close, heading the same way, reads as a dry pen
DOT_REACH = 3.3  # how far from its letter a dot can land and still be that letter's: real writing's 99th percentile
BAR_REACH = 1.5  # how far past its letter's ink a crossing bar may run
TICK = 0.8 * XH  # a high stroke this small, clear of a word's end, is a tick
TICK_GAP = 0.2
TANGLE = 6  # this many pen crossings within one x-height is a blob; 1.3% of real words reach it
EDGE_WIDE, EDGE_TALL = XH, 0.6 * XH  # past all other ink, a stroke within this is a dash or slash, not a letter
MISSING = 1.6  # a letter's ink spanning less than this was never written
OVERLONG = 2.6  # a letter with this many times the points of the word's median letter has something extra
STACKED = 1.0  # neighbouring letters overlapping by this share of the narrower one
SQUISHED = XH  # three letters in a row in less than this
LINE_LONG, LINE_STRAIGHT = 0.75 * XH, 0.95  # a stroke this long and straight, lying flat, is a ruled line
LINE_FOOT, LINE_TOP = -0.1 * XH, -1.15 * XH  # lower than the first, it underlines; higher than the second, it overlines
FOOT_BARS = set("LEZzD2_IJ")
MIDDLE_BARS = set("tfAHEFTzZxGLIJ457+-=#%")
TOP_BARS = set("tfTFEZ57IJ")
STEM_LONG, STEM_STRAIGHT = 1.25 * XH, 0.8  # up to this tall and this straight, a stroke past the edge could be a loose stem
STEMS = set("iljtfIJ1")
# How far real words stray, at their 99.9th percentile: lowest ink, highest ink, and how far
# the end of a word sits above or below its start, overall and over nine letters or more.
LOWEST, HIGHEST = 14.1, -20.2
DRIFT, LONG_DRIFT = 8.1, 5.6

MARKS_OF = {".": 1, ",": 1, ":": 2, ";": 2, "!": 1, "?": 1, "'": 1, '"': 2, "`": 1, "-": 1}
HIGH_PUNCT, LOW_PUNCT = set("'\"`!?:;"), set(".,;:!?-_")
DOTTED = "ij"
BARRED = set("tfTFEHAxkzZ457")


def strokes(steps):
    """(start, end) of each stroke, end exclusive, from (T, 3) steps with pen flags."""
    out, start = [], 0
    for e in np.flatnonzero(steps[:, 2] > 0.5):
        out.append((start, int(e) + 1))
        start = int(e) + 1
    if start < len(steps):
        out.append((start, len(steps)))
    return out


def _capacity(run):
    return sum(MARKS_OF.get(c, 2) for c in run)


def _fits(run, high):
    """Can this punctuation have a mark at this height? A comma has nothing above the letters."""
    known = HIGH_PUNCT | LOW_PUNCT
    return any(c not in known or c in (HIGH_PUNCT if high else LOW_PUNCT) for c in run)


def _runs(text):
    """Text split as leading punctuation, the middle, trailing punctuation."""
    a = 0
    while a < len(text) and not text[a].isalnum():
        a += 1
    b = len(text)
    while b > a and not text[b - 1].isalnum():
        b -= 1
    return text[:a], text[a:b], text[b:]


def classify(steps, text):
    """Name each small mark in a word.

    Returns a list of (stroke index, start, end, kind) with kind one of: lead, trail
    (leading or trailing punctuation), inner (punctuation within the word), dot, bar,
    stray, and close (unexplained, but touching ink so it doesn't show).
    """
    xy = np.cumsum(steps[:, :2], 0)
    spans = strokes(steps)
    if len(spans) < 2:
        return []
    size = [(np.ptp(xy[a:b, 0]), np.ptp(xy[a:b, 1])) for a, b in spans]
    body = [i for i, (w, h) in enumerate(size) if w > MARK or h > MARK]
    for i in list(body):
        w, h = size[i]
        if w <= EDGE_WIDE and h <= EDGE_TALL:
            a, b = spans[i]
            others = np.concatenate([xy[c:d, 0] for j, (c, d) in enumerate(spans) if j != i])
            if xy[a:b, 0].min() > others.max() or xy[a:b, 0].max() < others.min():
                body.remove(i)  # a dash or slash out past the word
    if not body:
        return []
    lo = min(xy[spans[i][0] : spans[i][1], 0].min() for i in body)
    hi = max(xy[spans[i][0] : spans[i][1], 0].max() for i in body)
    leading, middle, trailing = _runs(text)
    if not middle:
        return []  # punctuation alone: every mark is the text
    cap_lead, cap_trail = _capacity(leading), _capacity(trailing)
    inner = sum(MARKS_OF.get(c, 2) for c in middle if not c.isalnum())
    dots = sum(c in DOTTED for c in text)
    barred = any(c in BARRED for c in text)
    marks = [i for i in range(len(spans)) if i not in body]
    marks.sort(key=lambda i: xy[spans[i][0] : spans[i][1], 0].mean())
    third = (hi - lo) / 3

    def clearance(i):
        p = xy[spans[i][0] : spans[i][1]]
        rest = np.concatenate([xy[c:d] for j, (c, d) in enumerate(spans) if j != i])
        return np.sqrt(((p[:, None, :2] - rest[None, :, :2]) ** 2).sum(-1).min())

    found = []
    for i in marks:
        a, b = spans[i]
        p = xy[a:b]
        w, h = size[i]
        left, right, x = p[:, 0].min(), p[:, 0].max(), p[:, 0].mean()
        high = p[:, 1].mean() < HIGH
        if right < lo:
            if cap_lead > 0 and _fits(leading, high):
                kind, cap_lead = "lead", cap_lead - 1
            elif high and dots > 0 and any(c in DOTTED for c in text[:2]) and lo - right < NEAR:
                kind, dots = "dot", dots - 1
            elif w > BAR and w >= h and clearance(i) < SPECK and any(c in BARRED for c in text[:2]):
                kind = "bar"  # a t crossed only to the left of its stem
            else:
                kind = "stray"
        elif left > hi:
            if cap_trail > 0 and _fits(trailing, high):
                kind, cap_trail = "trail", cap_trail - 1
            elif high and dots > 0 and any(c in DOTTED for c in text[-2:]) and left - hi < NEAR:
                kind, dots = "dot", dots - 1
            elif w > BAR and w >= h and clearance(i) < SPECK and any(c in BARRED for c in text[-2:]):
                kind = "bar"
            else:
                kind = "stray"
        elif max(w, h) > SPECK:
            # A floating bar may yet cross a stem; with no such letter, even a near mark floats.
            if not (high and dots + inner + cap_lead + cap_trail == 0 and clearance(i) >= (FLOATING if barred else CLEAR)):
                continue  # within the word and this large: part of a letter
            kind = "stray"
        elif high and dots > 0:
            kind, dots = "dot", dots - 1
        elif inner > 0:
            kind, inner = "inner", inner - 1
        elif cap_trail > 0 and x > hi - third and _fits(trailing, high):  # a period under a tail, or a quote over the last letter
            kind, cap_trail = "trail", cap_trail - 1
        elif cap_lead > 0 and x < lo + third and _fits(leading, high):
            kind, cap_lead = "lead", cap_lead - 1
        elif barred and w > BAR and w >= h and clearance(i) < SPECK:  # a bar lies flat and near a stem
            kind = "bar"
        else:
            kind = "stray"
        if kind == "stray" and clearance(i) < CLEAR:
            kind = "close"
        found.append((i, a, b, kind))
    return found


def rebuild(xy, pen):
    """Steps from absolute points, with the word starting at x = 0 as words do."""
    out = np.zeros((len(xy), 3), dtype=xy.dtype)
    out[1:, :2] = np.diff(xy, axis=0)
    out[0, 1] = xy[0, 1]
    out[:, 2] = pen
    out[-1, 2] = 1.0
    return out


def flat_lines(steps, text):
    """Ruler-straight flat strokes at a height where none of the word's letters has a bar."""
    xy = np.cumsum(steps[:, :2], 0)
    spans = strokes(steps)
    if len(spans) < 2:
        return []
    found = []
    letters = set(text)
    for a, b in spans:
        if b - a < 2:
            continue
        p = xy[a:b]
        d = p[-1] - p[0]
        chord = np.hypot(*d)
        if chord < LINE_LONG or abs(d[1]) > 0.3 * abs(d[0]) or chord / np.hypot(*np.diff(p, axis=0).T).sum() < LINE_STRAIGHT:
            continue
        y = p[:, 1].mean()
        allowed = FOOT_BARS if y > LINE_FOOT else TOP_BARS if y < LINE_TOP else MIDDLE_BARS
        if not letters & allowed:
            found.append((a, b))
    return found


def loose_stems(steps, chars, text):
    """A stroke standing past the word's edge, when its letter already has its ink.

    Too tall to count as a mark, so only the letter labels can tell it from a real last
    letter. Letters that are nothing but a stem stay, since their label may lag a little.
    Before a word it must also be straight: first letters often start with a curved stroke
    that only just meets the rest. Past a word, a hook is as wrong as a stem.
    """
    xy = np.cumsum(steps[:, :2], 0)
    spans = strokes(steps)
    found = []
    for k, (a, b) in enumerate(spans):
        if b - a < 2 or len(spans) < 2:
            continue
        p = xy[a:b]
        if max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) > STEM_LONG:
            continue
        others = np.concatenate([xy[c:d, 0] for j, (c, d) in enumerate(spans) if j != k])
        if p[:, 0].min() > others.max() + CLEAR:
            edge = text[-1]
        elif p[:, 0].max() < others.min() - CLEAR:
            edge = text[0]
            if np.hypot(*(p[-1] - p[0])) / max(np.hypot(*np.diff(p, axis=0).T).sum(), 1e-6) < STEM_STRAIGHT:
                continue
        else:
            continue
        rest = np.ones(len(chars), bool)
        rest[a:b] = False
        if edge in STEMS or not edge.isalnum() or not np.isin(chars[a:b], chars[rest]).all():
            continue
        found.append((a, b))
    return found


def misplaced_dots(steps, chars, text):
    """Dots that landed further from every i and j than real writers' dots do."""
    centres = [np.median(np.cumsum(steps[:, 0])[chars == i]) for i, c in enumerate(text) if c in DOTTED and (chars == i).any()]
    if not centres:
        return []
    x = np.cumsum(steps[:, 0])
    return [(a, b) for _, a, b, kind in classify(steps, text) if kind == "dot" and min(abs(x[a:b].mean() - c) for c in centres) > DOT_REACH]


def misplaced_bars(steps, chars, text):
    """Bars nowhere near a letter that takes one: a t's cross is over the t, not past the e."""
    x = np.cumsum(steps[:, 0])
    reach = [(x[chars == i].min() - BAR_REACH, x[chars == i].max() + BAR_REACH)
             for i, c in enumerate(text) if c in BARRED and (chars == i).any()]
    return [(a, b) for _, a, b, kind in classify(steps, text)
            if kind == "bar" and not any(lo <= x[a:b].mean() <= hi for lo, hi in reach)]


def end_ticks(steps, chars, text):
    """A small high stroke just clear of a word's end, when its letter already has its ink."""
    xy = np.cumsum(steps[:, :2], 0)
    spans = strokes(steps)
    if len(spans) < 2:
        return []
    leading, _, trailing = _runs(text)
    centres = [np.median(xy[chars == i, 0]) for i, c in enumerate(text) if c in DOTTED and (chars == i).any()]
    found = []
    for k, (a, b) in enumerate(spans):
        p = xy[a:b]
        if max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) > TICK or p[:, 1].mean() >= HIGH:
            continue
        others = np.concatenate([xy[c:d, 0] for j, (c, d) in enumerate(spans) if j != k])
        past = p[:, 0].min() > others.max() + TICK_GAP
        before = p[:, 0].max() < others.min() - TICK_GAP
        if not (past or before) or (trailing if past else leading):
            continue  # inside the word, or its punctuation may explain it
        if any(abs(p[:, 0].mean() - c) <= DOT_REACH for c in centres):
            continue  # a dot, landed a little outside
        rest = np.ones(len(chars), bool)
        rest[a:b] = False
        if np.isin(chars[a:b], chars[rest]).all():
            found.append((a, b))
    return found


def wrong_strokes(steps, chars, text):
    """Every stroke that shouldn't be there, as (start, end) spans."""
    found = [(a, b) for _, a, b, kind in classify(steps, text) if kind == "stray"]
    found += flat_lines(steps, text) + loose_stems(steps, chars, text) + misplaced_dots(steps, chars, text) + end_ticks(steps, chars, text)
    found += misplaced_bars(steps, chars, text)
    return sorted(set(found))


def strip_strays(steps, chars, text, force=False):
    """Remove stray marks and lines, unless one is all the ink its character has.

    With `force`, those go too: their character's label was wrong, or it was never written.
    """
    drop = wrong_strokes(steps, chars, text)
    if not drop:
        return steps, chars
    keep = np.ones(len(steps), bool)
    for a, b in drop:
        keep[a:b] = False
        if not force and not np.isin(chars[a:b], chars[keep]).all():
            keep[a:b] = True
    if keep.all() or not keep.any():
        return steps, chars
    return rebuild(np.cumsum(steps[:, :2], 0)[keep], steps[keep, 2]), chars[keep]


def has_ending(steps, text):
    """Does a word whose text ends in a period actually end in a small mark?"""
    if not text.endswith(".") or not any(c.isalnum() for c in text):
        return True
    return any(kind == "trail" for _, _, _, kind in classify(steps, text))


def smallest_advance(steps, chars, text):
    """The least distance between the centres of neighbouring letters, or None if under two.

    When it nears zero the next letter was written on top of the last one. Real writers
    almost never go below 1 unit, a quarter of an x-height.
    """
    x = np.cumsum(steps[:, 0])
    centres = []
    for i, c in enumerate(text):
        mine = x[chars == i]
        if c.isalpha() and len(mine) >= 3:
            centres.append((i, np.median(mine)))
    gaps = [b[1] - a[1] for a, b in zip(centres, centres[1:]) if b[0] == a[0] + 1]
    return float(min(gaps)) if gaps else None


def letter_centres(xy, chars, text):
    """(index, median x) of each letter with enough ink to measure."""
    return [(i, np.median(xy[chars == i, 0])) for i, c in enumerate(text) if c.isalpha() and (chars == i).sum() >= 3]


def stacking(xy, chars, text):
    """The worst overlap between neighbouring letters, as a share of the narrower one."""
    boxes = []
    for i, c in enumerate(text):
        x = xy[chars == i, 0]
        boxes.append((np.percentile(x, 10), np.percentile(x, 90)) if c.isalpha() and len(x) >= 3 else None)
    worst = 0.0
    for a, b in zip(boxes, boxes[1:]):
        if a and b:
            worst = max(worst, (a[1] - b[0]) / max(min(a[1] - a[0], b[1] - b[0]), 1.0))
    return float(worst)


def tightest_three(xy, chars, text):
    """The least distance spanned by three letters in a row, or None under three letters."""
    c = letter_centres(xy, chars, text)
    spans = [z[1] - a[1] for a, b, z in zip(c, c[1:], c[2:]) if z[0] == a[0] + 2]
    return float(min(spans)) if spans else None


def find_breaks(steps):
    """Stroke ends where the next stroke picks up right there, heading the same way."""
    xy = np.cumsum(steps[:, :2], 0)
    spans = strokes(steps)
    found = []
    for (a, b), (c, d) in zip(spans, spans[1:]):
        if b - a < 3 or d - c < 3 or np.hypot(*(xy[c] - xy[b - 1])) > BREAK_GAP:
            continue
        out, into = xy[b - 1] - xy[b - 3], xy[c + 2] - xy[c]
        norm = np.hypot(*out) * np.hypot(*into)
        if norm > 0 and out @ into / norm > 0.5:
            found.append(b - 1)
    return found


def join_breaks(steps):
    """Keep the pen down across dry-pen breaks."""
    ends = find_breaks(steps)
    if not ends:
        return steps
    out = steps.copy()
    out[ends, 2] = 0.0
    return out


def undotted(steps, chars, text):
    """Which i and j have no small high mark near them."""
    if not any(c in DOTTED for c in text):
        return []
    xy = np.cumsum(steps[:, :2], 0)
    marks = [xy[a:b, 0].mean() for a, b in strokes(steps) if max(np.ptp(xy[a:b, 0]), np.ptp(xy[a:b, 1])) <= MARK and xy[a:b, 1].mean() < HIGH]
    missing = []
    for i, c in enumerate(text):
        if c not in DOTTED:
            continue
        x = xy[chars == i, 0]
        centre = np.median(x) if len(x) else None
        if centre is not None and marks:
            nearest = int(np.argmin(np.abs(np.array(marks) - centre)))
            if abs(marks[nearest] - centre) <= DOT_REACH:
                marks.pop(nearest)
                continue
        missing.append(i)
    return missing


def add_dots(steps, chars, text, rng):
    """Put in the dots a word still lacks, just above and a little right of each stem's top."""
    missing = undotted(steps, chars, text)
    if not missing:
        return steps, chars
    xy = np.cumsum(steps[:, :2], 0)
    points, pens, owners = [xy], [steps[:, 2]], [chars]
    for i in missing:
        ink = xy[chars == i]
        if not len(ink):
            continue
        top = ink[np.argmin(ink[:, 1])]
        centre = top + [rng.normal(0.3, 0.2), -rng.uniform(1.0, 1.8)]
        tail = rng.normal(0, 0.12, 2) + [0.12, 0.1]
        points.append(np.stack([centre, centre + tail]))
        pens.append(np.array([0.0, 1.0]))
        owners.append(np.full(2, i, dtype=chars.dtype))
    return rebuild(np.concatenate(points), np.concatenate(pens)), np.concatenate(owners)


def tidy(steps, chars, text):
    """Everything that can be put right without writing the word again."""
    steps, chars = strip_strays(steps, chars, text)
    return join_breaks(steps), chars


def tangle(xy, pen):
    """The most pen crossings within one x-height of each other: a blob has many."""
    a, b = xy[:-1], xy[1:]
    down = pen[:-1] < 0.5
    a, b = a[down], b[down]
    if len(a) < 4:
        return 0
    d = b - a
    ax, ay, dx, dy = a[:, 0][:, None], a[:, 1][:, None], d[:, 0][:, None], d[:, 1][:, None]
    between = dx * dy.T - dy * dx.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ax.T - ax) * dy.T - (ay.T - ay) * dx.T) / between
        u = ((ax.T - ax) * dy - (ay.T - ay) * dx) / between
    hit = (t > 0) & (t < 1) & (u > 0) & (u < 1) & np.triu(np.ones((len(a), len(a)), bool), 2)
    i, j = np.nonzero(hit)
    if not len(i):
        return 0
    where = a[i, 0] + t[i, j] * d[i, 0]
    return int(max(((where > c - XH / 2) & (where < c + XH / 2)).sum() for c in where))


def runaway(xy, text, size=1.0):
    """Has the word wandered off its line, further than real words ever do?"""
    y = xy[np.argsort(xy[:, 0]), 1] * size
    if y.max() > LOWEST or y.min() < HIGHEST:
        return True
    third = len(y) // 3
    if not third:
        return False
    return abs(np.median(y[-third:]) - np.median(y[:third])) > (LONG_DRIFT if len(text) >= 9 else DRIFT)


def letter_ink(steps, chars, text):
    """Letters the model left without ink, and letters it ran on far too long.

    The model writes one character at a time and labels its points with it, so a letter
    with almost no ink was never written ("icht" for "nicht"), and one with several times
    the points of the word's others carries something extra (a third hump on an n).
    """
    xy = np.cumsum(steps[:, :2], 0)
    letters = [i for i, c in enumerate(text) if c.isalnum()]
    counts = {i: int((chars == i).sum()) for i in letters}
    missing = [i for i in letters if counts[i] < 3 or max(np.ptp(xy[chars == i, 0]), np.ptp(xy[chars == i, 1])) < MISSING]
    typical = np.median([counts[i] for i in letters]) if len(letters) >= 3 else None
    overlong = [i for i in letters if typical is not None and counts[i] > max(OVERLONG * typical, 40)]
    return missing, overlong


def faults(steps, chars, text, size=1.0):
    """Faults only writing the word again can fix. `size` undoes the sampler's shrinking."""
    xy = np.cumsum(steps[:, :2], 0)
    out = []
    if wrong_strokes(steps, chars, text):
        out.append("stray")
    missing, overlong = letter_ink(steps, chars, text)
    if missing:
        out.append("missing letter")
    if overlong:
        out.append("overlong letter")
    if tangle(xy, steps[:, 2]) >= TANGLE:
        out.append("tangled")
    if undotted(steps, chars, text):
        out.append("undotted")
    if not has_ending(steps, text):
        out.append("no period")
    if stacking(xy, chars, text) >= STACKED:
        out.append("stacked")
    three = tightest_three(xy, chars, text)
    if three is not None and three * size < SQUISHED:
        out.append("squished")
    if runaway(xy, text, size):
        out.append("runaway")
    return out

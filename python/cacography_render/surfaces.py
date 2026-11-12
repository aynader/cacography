"""What the sheet lies on: real surfaces, photographed for rendering, or a photo of yours.

Each background is a family of CC0 textures from Poly Haven and ambientCG: colour, normal
and roughness maps scanned from real wood, stone, leather and cloth, downloaded once into
a cache folder on first use, never into the document. A document's `background` aspect
picks one of the family, then crops, turns, scales, tints and wears it, so two documents
rarely share a desk exactly. Cloth is not flat: it gets wrinkles as real geometry, and
the sheet drapes over them, bridging the hollows the way stiff paper does.
`background=file:<path>` lays your own photo of a surface under the sheet.

The room around the desk comes from CC0 HDRI photographs of real interiors (Poly Haven):
they light the scene from all round and show in glossy surfaces. Their brightest spots
are tamed, so the lamp, window or ceiling light the `light` aspect chose stays the main
light, and the room is turned so its own brightest side agrees with it.

Offline, or with CACOGRAPHY_OFFLINE set, the procedural desks in camera.DESKS stand in.
The cache is CACOGRAPHY_CACHE, or cacography/ in the user's cache folder.
"""
import io
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import blur, resize, rgb, sample, to_linear


def _ph(name):
    return ("polyhaven", name, None)


def _acg(name, mm):
    return ("ambientcg", name, mm)  # ambientCG doesn't say how big its scans are


# Each surface: its family of textures, the procedural desk standing in offline, how glossy
# a finish it may have (a clear coat over the texture), and for cloth how deep its
# wrinkles run (mm) and whether it comes dyed in any colour.
SURFACES = {
    # wood
    "oak": dict(maps=[_ph("oak_veneer_02"), _ph("oak_veneer_03"), _ph("white_oak_veneer"), _ph("red_oak_veneer")], fallback="oak", coat=(0.2, 0.8)),
    "walnut": dict(maps=[_ph("american_walnut_veneer"), _ph("black_walnut_veneer_01"), _ph("natural_walnut_veneer"), _ph("walnut_veneer_02")], fallback="walnut", coat=(0.3, 0.9)),
    "birch": dict(maps=[_ph("ash_veneer"), _ph("white_maple_veneer"), _ph("silver_oak_veneer_01")], fallback="birch", coat=(0.2, 0.7)),
    "pine": dict(maps=[_ph("coated_pine"), _ph("stained_pine"), _ph("coated_pine_02")], fallback="birch", coat=(0.3, 0.8)),
    "cherry": dict(maps=[_ph("cherry_veneer"), _ph("lacquered_cherry_wood")], fallback="walnut", coat=(0.5, 1.0)),
    "worn": dict(maps=[_ph("wood_table_worn"), _ph("wood_table"), _ph("kitchen_wood"), _ph("fine_grained_wood")], fallback="oak", coat=(0.0, 0.3)),
    "painted": dict(maps=[_ph("distressed_painted_planks"), _ph("black_painted_planks"), _ph("white_planks_clean")], fallback="laminate", coat=(0.1, 0.4)),
    "plywood": dict(maps=[_ph("plywood"), _ph("oriented_strand_board")], fallback="birch", coat=(0.0, 0.2)),
    # stone and the like
    "marble": dict(maps=[_acg("Marble006", 1200), _acg("Marble012", 1200), _acg("Marble016", 1200), _acg("Marble021", 1200)], fallback="laminate", coat=(0.6, 1.0)),
    "terrazzo": dict(maps=[_acg("Terrazzo005", 350), _acg("Terrazzo009", 350), _acg("Terrazzo013", 350)], fallback="laminate", coat=(0.4, 0.9)),
    "concrete": dict(maps=[_ph("concrete_floor_01"), _ph("concrete_floor_worn_001"), _ph("brushed_concrete")], fallback="felt", coat=(0.0, 0.3)),
    # desk pads
    "leather": dict(maps=[_ph("brown_leather"), _ph("fabric_leather_01"), _ph("fabric_leather_02")], fallback="felt", coat=(0.1, 0.3)),
    "cork": dict(maps=[_acg("Cork001", 500), _acg("Cork002", 500), _acg("Cork003", 500)], fallback="birch", coat=(0.0, 0.1)),
    # cloth, lying in folds
    "blanket": dict(maps=[_ph("knitted_fleece"), _ph("polar_fleece"), _ph("caban"), _ph("wool_boucle")], fallback="felt", wrinkles=(6, 16), dyed=True, sheen=0.6),
    "knit": dict(maps=[_ph("jersey_melange"), _ph("jogging_melange"), _ph("poly_wool_herringbone")], fallback="felt", wrinkles=(3, 10), dyed=True, sheen=0.4),
    "corduroy": dict(maps=[_ph("ribbed_corduroy")], fallback="felt", wrinkles=(2, 7), dyed=True, sheen=0.5),
    "velvet": dict(maps=[_ph("velour_velvet")], fallback="felt", wrinkles=(3, 9), dyed=True, sheen=1.0),
    # cloth made thread by thread (fabric.py): no download, any colour, weave or print
    "bedsheet": dict(cloth="bedsheet", wrinkles=(2, 8), crinkle=(0.0, 0.25)),
    "linen": dict(cloth="linen", wrinkles=(3, 9), crinkle=(0.3, 0.8)),
    "canvas": dict(cloth="canvas", wrinkles=(2, 6), crinkle=(0.0, 0.15)),
    "twill": dict(cloth="twill", wrinkles=(2, 7), crinkle=(0.0, 0.2)),
    "denim": dict(cloth="denim", wrinkles=(2, 7), crinkle=(0.05, 0.3)),
    "towel": dict(cloth="towel", wrinkles=(4, 12)),
    "stripes": dict(cloth="stripes", wrinkles=(2, 8), crinkle=(0.0, 0.25)),
    "plaid": dict(cloth="plaid", wrinkles=(2, 8), crinkle=(0.0, 0.3)),
    "gingham": dict(cloth="gingham", wrinkles=(2, 8), crinkle=(0.0, 0.25)),
    "dots": dict(cloth="dots", wrinkles=(2, 8), crinkle=(0.0, 0.25)),
    "print": dict(cloth="print", wrinkles=(2, 8), crinkle=(0.0, 0.25)),
    "grid": dict(cloth="grid", wrinkles=(3, 10), crinkle=(0.0, 0.2)),
    "jute": dict(cloth="jute", wrinkles=(0.2, 0.8)),
    "carpet": dict(maps=[_ph("dirty_carpet")], fallback="felt", wrinkles=(0.3, 1.0), sheen=0.3),
}
PROCEDURAL = ("laminate", "felt", "white", "black", "glass")  # drawn by camera._desk, no textures needed
GROUPS = {
    "wood": ["oak", "walnut", "birch", "pine", "cherry", "worn", "painted", "plywood"],
    "plain": ["laminate", "white", "black", "glass"],
    "stone": ["marble", "terrazzo", "concrete"],
    "fabric": ["blanket", "bedsheet", "linen", "canvas", "twill", "denim", "knit", "towel", "corduroy", "velvet",
               "stripes", "plaid", "gingham", "dots", "print", "grid"],
    "patterned": ["stripes", "plaid", "gingham", "dots", "print", "grid"],
    "pad": ["leather", "cork", "felt", "carpet", "jute"],
}
# How often a random background is each kind: most notes are photographed on a desk.
GROUP_ODDS = {"wood": 0.45, "plain": 0.17, "stone": 0.1, "fabric": 0.2, "pad": 0.08}
DYES = [(70, 72, 78), (150, 150, 152), (40, 52, 88), (110, 36, 48), (46, 82, 60), (196, 152, 60), (226, 216, 196),
        (196, 150, 150), (235, 235, 232), (32, 32, 34), (40, 110, 116), (190, 170, 140)]

# Rooms for each kind of light, chosen for the light they hold.
ROOMS = {
    "lamp": ["small_empty_room_2", "unfinished_office_night", "hotel_room", "wooden_lounge", "comfy_cafe", "small_empty_room_3"],
    "overhead": ["school_hall", "yoga_room", "hospital_room", "newman_cafeteria", "st_fagans_interior", "blender_institute"],
    "window": ["lythwood_room", "old_room", "small_empty_room_1", "small_empty_room_4", "unfinished_office", "kiara_interior", "reading_room"],
    "soft": ["lythwood_room", "abandoned_games_room_01", "brown_photostudio_04", "church_meeting_room", "interior_construction", "empty_play_room"],
}

CELL = 2.0  # mm between wrinkle samples, the same as Sheet.BEND_MM so the two grids line up
MARGIN = 700.0  # how far past the sheet the cloth's wrinkles reach, mm
FINE = 1.0  # mm between samples of the cloth's crinkles
STIFF = 55.0  # paper bridges hollows narrower than this, mm: it is too stiff to follow small rucks
SAG = 0.35  # but it sags this share of the way into a hollow it bridges, under its own weight


def names():
    return ["random", *GROUPS, *SURFACES, *(p for p in PROCEDURAL if p not in SURFACES), "file:<path>"]


def valid(value):
    return value == "random" or value in GROUPS or value in SURFACES or value in PROCEDURAL or value.startswith("file:")


def choose(value, rng):
    """The surface a `background` value names: itself, one of its group, or any at all."""
    if value == "random":
        value = str(rng.choice(list(GROUP_ODDS), p=list(GROUP_ODDS.values())))
    if value in GROUPS:
        value = str(rng.choice(GROUPS[value]))
    return value


# -- downloads ---------------------------------------------------------------

def cache_dir():
    if os.environ.get("CACOGRAPHY_CACHE"):
        return Path(os.environ["CACOGRAPHY_CACHE"])
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "cacography"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cacography (https://ctan.org)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _save(path, data):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


_warned = set()


def _offline(what, err):
    if what not in _warned:
        _warned.add(what)
        print(f"note: couldn't fetch {what} ({err}); using a procedural stand-in")


def fetch(source):
    """The cache folder holding one texture's maps, downloading them the first time; None offline."""
    site, name, mm = source
    folder = cache_dir() / site / name
    if (folder / "meta.json").exists():
        return folder
    if os.environ.get("CACOGRAPHY_OFFLINE"):
        return None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"downloading the {name} texture from {site} (CC0), once")
        if site == "polyhaven":
            files = json.loads(_get(f"https://api.polyhaven.com/files/{name}"))
            info = json.loads(_get(f"https://api.polyhaven.com/info/{name}"))
            tile = [float(d) for d in info["dimensions"]]
            res = "4k" if max(tile) > 700 else "2k"  # about 4 pixels per mm or more
            for key, out in (("Diffuse", "colour.jpg"), ("nor_gl", "normal.jpg"), ("Rough", "rough.jpg")):
                if key in files:
                    _save(folder / out, _get(files[key][res]["jpg"]["url"]))
        else:
            tile = [float(mm), float(mm)]
            with zipfile.ZipFile(io.BytesIO(_get(f"https://ambientcg.com/get?file={name}_2K-JPG.zip"))) as z:
                for member in z.namelist():
                    base = Path(member).name  # never a path from the archive
                    for end, out in (("_Color.jpg", "colour.jpg"), ("_NormalGL.jpg", "normal.jpg"), ("_Roughness.jpg", "rough.jpg")):
                        if base.endswith(end):
                            _save(folder / out, z.read(member))
        if not (folder / "colour.jpg").exists():
            raise OSError("no colour map")
        (folder / "meta.json").write_text(json.dumps(dict(tile_mm=tile, source=f"{site}:{name}", license="CC0")))
        return folder
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        _offline(f"{site}:{name}", e)
        return None


def room(name):
    """A room's HDRI file, downloading it the first time; None offline."""
    path = cache_dir() / "polyhaven" / "hdri" / f"{name}_2k.hdr"
    if path.exists():
        return path
    if os.environ.get("CACOGRAPHY_OFFLINE"):
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading the {name} room light from polyhaven (CC0), once")
        files = json.loads(_get(f"https://api.polyhaven.com/files/{name}"))
        _save(path, _get(files["hdri"]["2k"]["hdr"]["url"]))
        return path
    except (OSError, ValueError, KeyError) as e:
        _offline(f"room {name}", e)
        return None


@lru_cache(maxsize=6)
def _maps(folder):
    folder = Path(folder)
    colour = to_linear(np.asarray(Image.open(folder / "colour.jpg").convert("RGB"), np.float32) / 255)
    h, w = colour.shape[:2]
    if (folder / "normal.jpg").exists():
        n = np.asarray(Image.open(folder / "normal.jpg").convert("RGB").resize((w, h)), np.float32) / 127.5 - 1
        n[..., 1] *= -1  # OpenGL normal maps point green up the image; here v runs down it
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    else:
        n = np.zeros((h, w, 3), np.float32) + [0, 0, 1]
    if (folder / "rough.jpg").exists():
        rough = np.asarray(Image.open(folder / "rough.jpg").convert("L").resize((w, h)), np.float32) / 255
    else:
        rough = np.full((h, w), 0.6, np.float32)
    tile = json.loads((folder / "meta.json").read_text())["tile_mm"]
    return colour, n.astype(np.float32), rough, (float(tile[0]), float(tile[1]))


def _photo(path):
    colour = to_linear(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255)
    h, w = colour.shape[:2]
    return colour, np.zeros((h, w, 3), np.float32) + [0, 0, 1], np.full((h, w), 0.55, np.float32)


# -- the surface under one page ---------------------------------------------

@dataclass
class Surface:
    name: str
    procedural: str = None  # a camera.DESKS name, when there are no textures
    colour: np.ndarray = None  # one tile, linear
    normal: np.ndarray = None  # tangent space: x along u, y along v, z up
    rough: np.ndarray = None
    tile_mm: tuple = (1000.0, 1000.0)
    angle: float = 0.0  # how the texture is turned on the desk
    offset: tuple = (0.0, 0.0)  # mm into the tile
    cover: tuple = None  # (x0, y0, w, h) for a photo stretched once over the desk, no repeats
    coat: float = 0.0  # clear coat over the surface, 0 to 1
    coat_gloss: float = 0.8
    sheen: float = 0.0  # cloth's soft shine at grazing angles
    height: np.ndarray = None  # wrinkles, mm, one sample every CELL from `origin`
    origin: tuple = (0.0, 0.0)
    crinkle: np.ndarray = None  # fine creases of washed cloth, mm, one sample every FINE from `origin`
    true_normals: bool = False  # normal maps built at the cloth's real scale (fabric.py), not a close-up scan's

    @property
    def textured(self):
        return self.colour is not None

    def texcoords(self, x, y):
        """Where points on the desk (mm) fall in the texture, in tiles (repeating past 1)."""
        if self.cover is not None:
            x0, y0, w, h = self.cover
            return (x - x0) / w, (y - y0) / h
        c, s = np.cos(self.angle), np.sin(self.angle)
        u = (c * x + s * y) + self.offset[0]
        v = (-s * x + c * y) + self.offset[1]
        return u / self.tile_mm[0], v / self.tile_mm[1]

    def colour_at(self, x, y):
        u, v = self.texcoords(x, y)
        h, w = self.colour.shape[:2]
        if self.cover is None:
            u, v = u - np.floor(u), v - np.floor(v)
        return sample(self.colour, u * (w - 1), v * (h - 1))

    def height_at(self, x, y):
        if self.height is None:
            return np.zeros_like(np.asarray(x, np.float32))
        return sample(self.height, (x - self.origin[0]) / CELL, (y - self.origin[1]) / CELL)

    def surface_at(self, x, y):
        """The cloth's top, crinkles included: what the eye sees, where the sheet rests on
        the wrinkles alone (paper bridges crinkles)."""
        z = self.height_at(x, y)
        if self.crinkle is not None:
            z = z + sample(self.crinkle, (x - self.origin[0]) / FINE, (y - self.origin[1]) / FINE)
        return z

    def drape(self, sheet):
        """The sheet lying on this surface: over cloth it rests on the crests, bridging the hollows."""
        if self.height is None:
            return sheet
        bh, bw = sheet.bend.shape
        i0, j0 = int(round(-self.origin[1] / CELL)), int(round(-self.origin[0] / CELL))
        pad = int(2 * STIFF / CELL)
        height = self.height
        if self.crinkle is not None:
            # The cloth is drawn with its fine crinkles on top; the paper must clear their
            # peaks too, or they poke up through the sheet as little lumps.
            peaks = _dilate(self.crinkle, int(2 * CELL / FINE) | 1)
            gy, gx = np.mgrid[0 : height.shape[0], 0 : height.shape[1]].astype(np.float32) * CELL
            height = height + sample(peaks, gx / FINE, gy / FINE)
        under = height[max(i0 - pad, 0) : i0 + bh + pad, max(j0 - pad, 0) : j0 + bw + pad]
        top, left = min(pad, i0), min(pad, j0)
        # The sheet settles: let go from above the highest crest, it is pulled down by its
        # weight a little at a time and stays smooth because it resists bending, and it can't
        # go into the cloth. It comes to rest on the crests, bridging hollows narrower than
        # about STIFF and sagging into wider ones; its edges are free.
        inside = np.zeros_like(under)
        inside[top : top + bh, left : left + bw] = 1.0
        spread = 0.1 * STIFF / CELL  # cells the bending stiffness evens out per step
        drop = 0.0065 * spread**2  # mm per step: a 5 cm span sags about a millimetre
        h = np.maximum(under, blur(_dilate(under, 9), 3.0))  # let go from just above the crests
        weight = blur(inside, spread) + 1e-6
        for _ in range(800):
            h = np.maximum(under, blur(h * inside, spread) / weight - drop)
        # That alone would hug every small ruck like a stretched skin. Paper is stiff: it
        # keeps no bend tighter than about a centimetre, so smooth it at that scale, and
        # where a crest then stands above it, lift it over the crest in a broad tent.
        smooth = 0.11 * STIFF / CELL
        weight = blur(inside, smooth) + 1e-6
        for _ in range(6):
            h = blur(h * inside, smooth) / weight
            h = h + blur(_dilate(np.maximum(under - h, 0), int(2 * smooth) | 1), smooth)
        h = np.maximum(h, under)
        rest = h[top : top + bh, left : left + bw]
        return replace(sheet, bend=(sheet.bend + rest).astype(np.float32))


def _slide(a, k, axis, pick):
    pad = [(k // 2, k // 2) if ax == axis else (0, 0) for ax in range(2)]
    return pick(np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, mode="edge"), k, axis=axis), -1)


def _dilate(a, k):
    return _slide(_slide(a, k, 0, np.max), k, 1, np.max)


def _erode(a, k):
    return _slide(_slide(a, k, 0, np.min), k, 1, np.min)


def _periodic(shape, rng, cells):
    """Smooth noise that wraps round the tile, so tiling shows no seam: `cells` blobs across."""
    small = rng.standard_normal((cells, cells)).astype(np.float32)
    f = np.fft.fft2(small)
    fy, fx = np.fft.fftfreq(cells)[:, None], np.fft.fftfreq(cells)[None, :]
    small = np.real(np.fft.ifft2(f * np.exp(-((fx**2 + fy**2) * cells**2) / 6))).astype(np.float32)
    small /= small.std() + 1e-9
    big = resize(np.tile(small, (3, 3)), (3 * shape[0], 3 * shape[1]), mode="bicubic")
    return big[shape[0] : 2 * shape[0], shape[1] : 2 * shape[1]]


def _wrinkles(rng, shape, amplitude):
    """Cloth lying loose: a few long folds, rounded on top with a tight crease between
    them, mostly running one way, with a gentle swell under them and smaller rucks.

    Real bedding from above: folds a few centimetres across, softly shaded, not an even
    field of bumps."""
    from .imaging import band

    a = rng.uniform(0, np.pi)
    folds = band(shape, rng, 60 / CELL, 300 / CELL, stretch=rng.uniform(2.5, 5), angle=a)
    tubes = np.abs(folds) ** 0.7  # rounded crests, creased where two folds meet
    swell = band(shape, rng, 250 / CELL, 900 / CELL)
    rucks = band(shape, rng, 15 / CELL, 60 / CELL, stretch=1.5, angle=a + rng.normal(0, 0.5))
    rucks *= np.clip(0.3 + band(shape, rng, 150 / CELL, 500 / CELL), 0, None)  # here and there, not everywhere
    h = 0.75 * (tubes - tubes.mean()) + 0.5 * swell + 0.15 * rucks
    return (amplitude * 0.6 * h).astype(np.float32)


def prepare(st, aspects, page, wmm, hmm, user_dir=Path(".")):
    """The surface under one page's sheet. Its choice, crop and tint stay for the whole
    document; cloth falls in new wrinkles on every page, as it would."""
    name = st.desk
    rng = aspects.rng("background", part="surface")
    if name.startswith("file:"):
        path = Path(name[5:].split("|")[0])
        path = path if path.is_absolute() else user_dir / path
        colour, normal, rough = _photo(path)
        h, w = colour.shape[:2]
        # Stretched once over the stretch of desk a photo shows, keeping its proportions.
        span = max((wmm + 520) / w, (hmm + 520) / h)
        cw, ch = w * span, h * span
        return Surface(name, colour=colour, normal=normal, rough=rough,
                       cover=(wmm / 2 - cw / 2, hmm / 2 - ch / 2, cw, ch), coat=0.15)
    spec = SURFACES.get(name)
    if spec is None:
        return Surface(name, procedural=name)
    if "cloth" in spec:
        colour, normal, rough, tile, sheen = _cloth(spec["cloth"], int(rng.integers(2**31)))
        surface = Surface(name, colour=colour, normal=normal, rough=rough, tile_mm=tile, angle=float(st.desk_angle) + rng.normal(0, 0.3),
                          offset=(rng.uniform(0, tile[0]), rng.uniform(0, tile[1])), sheen=sheen, true_normals=True)
        return _wrinkled(surface, spec, aspects, page, wmm, hmm)
    source = spec["maps"][int(rng.integers(len(spec["maps"])))]
    folder = fetch(source)
    if folder is None:
        return Surface(name, procedural=spec["fallback"])
    colour, normal, rough, tile = _maps(str(folder))
    if source[2]:
        tile = (float(source[2]), float(source[2]))  # the catalogue's size wins over what was cached
    scale = rng.uniform(0.9, 1.1)
    # A little of its own colour: no two tables, or blankets, are quite alike.
    tint = np.exp(rng.normal(0, 0.06) + rng.normal(0, 0.02, 3)).astype(np.float32)
    if spec.get("dyed") and source[1] != "gingham_check" and rng.random() < 0.8:
        lum = colour.mean(-1, keepdims=True)
        colour = lum / lum.mean() * rgb(DYES[int(rng.integers(len(DYES)))])
    worn = 1 + 0.05 * _periodic(colour.shape[:2], rng, 6)[..., None]
    coat = rng.uniform(*spec.get("coat", (0.0, 0.0)))
    # Varnish or polish soaks in and deepens the colour, the way wet wood looks darker;
    # the textures are scans of the bare surface.
    colour = np.clip(colour * tint * worn, 0, 1) ** (1 + 0.6 * coat)
    colour = colour.astype(np.float32)
    surface = Surface(
        name, colour=colour, normal=normal, rough=rough, tile_mm=(tile[0] * scale, tile[1] * scale),
        angle=float(st.desk_angle), offset=(rng.uniform(0, tile[0]), rng.uniform(0, tile[1])),
        coat=float(coat), coat_gloss=float(rng.uniform(0.6, 0.95)), sheen=spec.get("sheen", 0.0),
    )
    return _wrinkled(surface, spec, aspects, page, wmm, hmm)


def _wrinkled(surface, spec, aspects, page, wmm, hmm):
    """Cloth lies in folds, new on every page; washed cloth is crinkled too."""
    if "wrinkles" not in spec:
        return surface
    folds = aspects.rng("background", part=f"wrinkles {page}")
    shape = (int((hmm + 2 * MARGIN) / CELL) + 1, int((wmm + 2 * MARGIN) / CELL) + 1)
    surface = replace(surface, height=_wrinkles(folds, shape, folds.uniform(*spec["wrinkles"])), origin=(-MARGIN, -MARGIN))
    amount = folds.uniform(*spec.get("crinkle", (0.0, 0.0)))
    if amount > 0.02:
        fine = (int((hmm + 2 * MARGIN) / FINE) + 1, int((wmm + 2 * MARGIN) / FINE) + 1)
        surface = replace(surface, crinkle=_crinkles(folds, fine, amount, surface.angle))
    return surface


def _crinkles(rng, shape, amount, angle):
    """Washed linen and cotton: a web of small sharp creases a few millimetres apart,
    running mostly along the threads, denser in some places than others."""
    from .imaging import band

    a = band(shape, rng, 4 / FINE, 18 / FINE, stretch=rng.uniform(1.5, 3), angle=angle + rng.normal(0, 0.3))
    b = band(shape, rng, 3 / FINE, 10 / FINE, stretch=1.2, angle=angle + np.pi / 2)
    ridges = -np.abs(a) + 0.4 * b  # creases are sharp valleys between soft swells
    patch = np.clip(0.5 + 0.6 * band(shape, rng, 60 / FINE, 250 / FINE), 0.1, 1.0)
    return (amount * 0.35 * (ridges - ridges.mean()) * patch).astype(np.float32)


@lru_cache(maxsize=4)
def _cloth(kind, seed):
    from . import fabric

    return fabric.make(kind, np.random.default_rng(seed))


def pick_room(kind, rng):
    # A flash or a dim evening lamp is shot in the rooms one uses at night.
    kind = {"raking": "window", "flash": "lamp", "dim": "lamp"}.get(kind, kind)
    return str(rng.choice(ROOMS.get(kind, ROOMS["soft"])))

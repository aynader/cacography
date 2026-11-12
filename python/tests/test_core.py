"""Checks that need no GPU, models or TeX: drawings, cloth, props, page grouping.

    cd render && python -m unittest discover tests
"""
import unittest
from types import SimpleNamespace

import numpy as np

from cacography_render import camera, drawing, fabric, page, pens, props


def flat(x, y):
    return np.zeros_like(np.asarray(x, float))


class Drawing(unittest.TestCase):
    def lit(self, d):
        return {"x": 0.0, "y": 100.0, "d": d, "line": 1}

    def test_clip_cuts_a_line_to_the_axis_box(self):
        draw = [self.lit("q 0 0 10 10 re W n"), self.lit("-5 5 m 15 5 l S"), self.lit("Q"), self.lit("-5 2 m 15 2 l S")]
        paths = drawing.paths(draw)
        self.assertEqual(len(paths), 2)
        clipped, free = paths
        span = np.ptp(clipped.points[:, 0]) / drawing.PT_PER_BP
        self.assertAlmostEqual(span, 10, delta=0.6)  # cut to the box
        self.assertAlmostEqual(np.ptp(free.points[:, 0]) / drawing.PT_PER_BP, 20, delta=0.01)  # after Q: not clipped

    def test_fill_is_cut_to_the_clip_and_keeps_its_grey(self):
        draw = [self.lit("q 0 0 10 10 re W n 0.7 g -5 -5 10 10 re f Q")]
        (p,) = drawing.paths(draw)
        self.assertTrue(p.filled)
        self.assertAlmostEqual(p.fill_grey, 0.7)
        self.assertLessEqual(np.ptp(p.points[:, 0]) / drawing.PT_PER_BP, 5.01)

    def test_stroke_grey_from_rgb(self):
        (p,) = drawing.paths([self.lit("0.75 0.75 0.75 RG 0 0 m 5 5 l S")])
        self.assertAlmostEqual(p.stroke_grey, 0.75, places=3)


class Cloth(unittest.TestCase):
    def check(self, kind):
        colour, normal, rough, tile, sheen = fabric.make(kind, np.random.default_rng(3))
        self.assertEqual(colour.shape[:2], normal.shape[:2])
        self.assertTrue(np.isfinite(colour).all() and colour.min() >= 0 and colour.max() <= 1)
        self.assertTrue(np.allclose(np.linalg.norm(normal, axis=-1), 1, atol=1e-3))
        self.assertGreater(tile[0], 10)
        # The tile repeats without a seam: across the wrap, no bigger a jump than inside.
        inside = np.abs(np.diff(colour, axis=1)).mean()
        seam = np.abs(colour[:, 0] - colour[:, -1]).mean()
        self.assertLess(seam, 3 * inside)

    def test_bedsheet(self):
        self.check("bedsheet")

    def test_towel(self):
        self.check("towel")

    def test_gingham(self):
        self.check("gingham")


class Props(unittest.TestCase):
    def setUp(self):
        self._top = props.top
        props.top = lambda sheet, x, y: np.zeros_like(np.asarray(x, float))
        self.sheet = SimpleNamespace(width_mm=216, height_mm=279, ppmm=2.0, mask=np.ones((558, 432), np.float32), ink="blue")

    def tearDown(self):
        props.top = self._top

    def valid(self, things):
        for th in things:
            self.assertTrue(np.isfinite(th["verts"]).all())
            self.assertLess(int(th["faces"].max()), len(th["verts"]))
            if th.get("uv") is not None and not isinstance(th["uv"], str):
                self.assertEqual(len(th["uv"]), len(th["verts"]))

    def test_every_pen(self):
        for ink in ("blue", "black", "red"):
            self.sheet.ink = ink
            for seed in range(40):
                self.valid(props._pen(self.sheet, np.random.default_rng(seed), flat))

    def test_every_pencil(self):
        self.sheet.ink = "pencil"
        for seed in range(40):
            self.valid(props._pen(self.sheet, np.random.default_rng(seed), flat))

    def test_tubes_face_outward(self):
        path = np.column_stack([np.linspace(0, 50, 20), np.zeros(20), np.full(20, 5.0)])
        for v, f in (props._tube(path, np.full(20, 3.0), 12), props._tube_tex(path, np.full(20, 3.0), 12)[:2]):
            tri = v[f[:200]]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            c = tri.mean(1)
            out = c - np.column_stack([c[:, 0], np.zeros(len(c)), np.full(len(c), 5.0)])
            self.assertTrue((np.sum(n * out, 1) > 0).all())


    def test_every_pen_style_is_closed_and_faces_outward(self):
        ref = np.sign(pens._volume(*props._tube(np.array([[0, 0, 0], [10, 0, 0.0]], np.float32), np.array([1.0, 1.0], np.float32), 12)))
        for style in pens.STYLES:
            for seed in range(12):
                things = pens.build(self.sheet, np.random.default_rng(seed), flat, style=style, place=((100.0, 100.0), 0.3))
                self.valid(things)
                for th in things:
                    vol = pens._volume(th["verts"], th["faces"])
                    if abs(vol) > 1e-3:
                        self.assertEqual(np.sign(vol), ref, (style, seed, th["material"]["kind"]))

    def test_pen_lies_on_the_page(self):
        for seed in range(10):
            things = pens.build(self.sheet, np.random.default_rng(seed), flat, style="click", place=((100.0, 100.0), 0.0))
            low = min(float(th["verts"][:, 2].min()) for th in things)
            self.assertGreater(low, -0.05)  # resting on the desk, not sunk into it
            self.assertLess(low, 0.6)  # and not hovering over it

    def test_wooden_pencils(self):
        for seed in range(20):
            self.valid(pens.pencil(self.sheet, np.random.default_rng(seed), flat, place=((100.0, 100.0), 1.0)))

    def test_binder_clips(self):
        sheet = SimpleNamespace(width_mm=216, height_mm=279, ppmm=2.0, mask=np.ones((558, 432), np.float32), ink="blue",
                                bend=np.zeros((140, 108), np.float32), BEND_MM=2.0)
        for seed in range(12):
            things = props._binder(sheet, np.random.default_rng(seed), flat)
            self.valid(things)
            self.assertGreater(len(things), 3)  # body, two rolls, at least one handle


class Page(unittest.TestCase):
    def test_hatch_ends_wander_without_an_outline(self):
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
        starts = []
        for seed in range(8):
            for p in page._hatch(square, 5.0, 0.0, np.random.default_rng(seed), outlined=False):
                if len(p) == 3:
                    starts.append(p[0, 0])
        self.assertGreater(np.std(starts), 1.5)  # not all on one invisible line

    def test_erased_text_goes_under_the_first_word_only(self):
        g = dict(y=100.0, w=5.0, h=7.0, d=0.0, size=16.0, xh=6.9, math=False, line=1, italic=False, bold=False)
        glyphs = [dict(g, x=10.0, t="e", word=1, erased=1), dict(g, x=15.0, t="s", word=1, erased=1),
                  dict(g, x=40.0, t="i", word=2, erased=1), dict(g, x=60.0, t="x", word=3)]
        scene = {"lines": [{"id": 1, "x": 0.0, "y": 100.0, "w": 400.0, "h": 10.0, "d": 2.0}], "glyphs": glyphs, "erased": {"1": "das war"}}
        (line,) = page.page_units(scene)
        self.assertEqual([u.erased for u in line["units"]], ["das war", "", ""])

    def test_no_slips_in_ink(self):
        w = page.PageWriter.__new__(page.PageWriter)
        w.doc, w.pen = {"ink": "blue", "wear": 1.0}, SimpleNamespace(seed=1)
        for k in range(50):
            self.assertEqual(w._slip(page.Unit([], False, 0, 1, text="Mitochondrien", key=f"text:x#{k}")), "")

    def test_the_hand_shades_past_the_phone(self):
        occ = camera._occluder(np.random.default_rng(3), 210, 297, "phone")
        hx, hy = occ["half"]
        palm = next(p for p in occ["hand"] if p[5] == -12.0)
        c, s_ = np.cos(occ["angle"]), np.sin(occ["angle"])
        u, v = palm[0] * hx, (palm[1] + 0.5 * palm[3]) * hy  # in the palm, below the phone's end
        x, y = occ["cx"] + u * c - v * s_, occ["cy"] + u * s_ + v * c
        self.assertGreater(float(camera._shade_of(occ, np.array([x]), np.array([y]))[0]), 0.3)


class Page2(unittest.TestCase):
    def test_crossed_words_are_marked(self):
        g = dict(y=100.0, w=5.0, h=7.0, d=0.0, size=16.0, xh=6.9, math=False, line=1, italic=False, bold=False)
        glyphs = [dict(g, x=10.0, t="a", word=1), dict(g, x=15.0, t="b", word=1, cross=4), dict(g, x=40.0, t="c", word=2)]
        scene = {"lines": [{"id": 1, "x": 0.0, "y": 100.0, "w": 400.0, "h": 10.0, "d": 2.0}], "glyphs": glyphs}
        (line,) = page.page_units(scene)
        self.assertEqual([u.cross for u in line["units"]], [4, 0])

    def test_plus_makes_a_word_math(self):
        g = dict(y=100.0, w=5.0, h=7.0, d=0.0, size=16.0, xh=6.9, math=False, line=1, italic=False, bold=False, word=1)
        scene = {"lines": [{"id": 1, "x": 0.0, "y": 100.0, "w": 400.0, "h": 10.0, "d": 2.0}],
                 "glyphs": [dict(g, x=10.0, t="1"), dict(g, x=15.0, t="5"), dict(g, x=20.0, t="+")]}
        (line,) = page.page_units(scene)
        self.assertTrue(line["units"][0].math)

    def test_primes_are_drawn_as_ticks(self):
        self.assertEqual(page.PRIMES["″"], 2)

    def test_every_scribble_stays_near_the_words(self):
        for kind in ("scrub", "coil", "tangle"):
            for seed in range(20):
                rng = np.random.default_rng(seed)
                for p in page._scribble(kind, 100.0, 300.0, lambda x: 50.0 + 0 * np.atleast_1d(x), 20.0, rng):
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue((abs(p[:, 1] - 50) < 60).all() and (p[:, 0] > 40).all() and (p[:, 0] < 360).all(), kind)


class Options(unittest.TestCase):
    def test_numbers_with_a_leading_dot(self):
        from cacography_render import aspects
        for text in ("1", "0.8", ".8", "-15", "1."):
            self.assertTrue(aspects._number(text), text)
        self.assertFalse(aspects._number("lots"))


class Sloppiness(unittest.TestCase):
    def test_default_pages_stay_as_they_were_and_the_ends_differ(self):
        self.assertEqual(page.mess(0.5), 1.0)
        self.assertEqual(page.mess("0.5"), 1.0)  # as the document records it
        self.assertLess(page.mess(0), 0.5)
        self.assertGreater(page.mess(1), 1.5)
        self.assertEqual(page.mess(5), page.mess(1))


class Command(unittest.TestCase):
    def test_setup_finds_the_latex_files_and_their_version(self):
        from cacography_render import cli
        tex = cli._shipped_tex()
        self.assertIsNotNone(tex)
        self.assertTrue((tex / "cacography.lua").is_file())
        self.assertRegex(cli._sty_version(tex / "cacography.sty"), r"^\d+\.\d+\.\d+")

    def test_help_and_version_need_nothing_installed(self):
        from cacography_render import cli
        self.assertEqual(cli.main(["--help"]), 0)
        self.assertEqual(cli.main(["--version"]), 0)


if __name__ == "__main__":
    unittest.main()

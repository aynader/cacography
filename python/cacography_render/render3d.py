"""Photograph the sheet by tracing light through a 3D scene (Mitsuba 3), not by formula.

The sheet becomes a mesh bent by its folds, curl and lifted corners, its grain a normal
map, its holes and torn edge cut out. Paper and ink are rough glossy surfaces, ink the
glossier, so glare and shine come from the lamp's real reflection. The lamp is an area
light with a size, so shadows have soft edges where they should; the phone and hand are
a real object between it and the page; the room's light comes from all around and
bounces off the desk onto the paper. The camera is the same pinhole camera.py frames,
and the phone's processing (_develop) is the same too.

The desk is a real surface from surfaces.py, with its colour, normal and roughness maps
and a clear coat; cloth is a wrinkled mesh the sheet lies on. The room is an HDRI of a
real interior, so glossy desks, ink and paper reflect a real room.

Needs the `mitsuba` package and an NVIDIA GPU; camera.photograph is the fallback.
"""
import tempfile
from pathlib import Path

import numpy as np

from . import fabric, props, surfaces
from .camera import _desk, _develop, _frame, _normals, _occluder, desk_bounds, desk_gloss, footprint, with_flash
from .imaging import blur, sample

MESH_MM = 0.5  # finer than the bend, so punched holes can be cut out round
SAMPLES = 128  # paths per pixel: the photo's own noise hides what little is left
LAMP_MM = {"lamp": 90, "overhead": 220, "window": 1300, "soft": 1400, "raking": 260}  # the light's size

_mi = None


def available():
    """Whether Mitsuba and a CUDA GPU are there to render with."""
    global _mi
    if _mi is not None:
        return True
    try:
        import mitsuba as mi

        mi.set_variant("cuda_ad_rgb")
        _mi = mi
        return True
    except (ImportError, AttributeError, RuntimeError):
        return False


def _bitmap(a):
    return _mi.Bitmap(np.ascontiguousarray(a, dtype=np.float32))


def _texture(a):
    return {"type": "bitmap", "bitmap": _bitmap(a), "raw": True, "filter_type": "bilinear"}


def _sheet_mesh(sheet, path):
    """The sheet as triangles, a vertex every MESH_MM on its bend, saved as PLY."""
    cell = MESH_MM
    ny, nx = int(np.ceil(sheet.height_mm / cell)) + 1, int(np.ceil(sheet.width_mm / cell)) + 1
    v, u = np.mgrid[0:ny, 0:nx].astype(np.float32)
    x = np.minimum(u * cell, sheet.width_mm)
    y = np.minimum(v * cell, sheet.height_mm)
    z = sheet.lift(x, y) + 0.1  # paper is a tenth of a millimetre thick
    # Shadow rays can't pass through the opacity mask, so light would reach the desk
    # under a punched hole or past a torn edge only by chance hits on the lamp, which
    # leaves bright speckles. Triangles whose corners all miss the paper are left out
    # instead: a hole is round, so such a triangle lies wholly inside it.
    keep = sample(sheet.mask, x * sheet.ppmm, y * sheet.ppmm) > 0.01
    _grid_mesh("sheet", x, y, z, x / sheet.width_mm, y / sheet.height_mm, path, keep)


def _mesh(name, verts, faces, uv, path):
    """Triangles (with texture coordinates, if given) saved as PLY."""
    mi = _mi
    mesh = mi.Mesh(name, len(verts), len(faces), has_vertex_texcoords=uv is not None)
    p = mi.traverse(mesh)
    p["vertex_positions"] = mi.Float(np.asarray(verts, np.float32).ravel())
    if uv is not None:
        p["vertex_texcoords"] = mi.Float(np.asarray(uv, np.float32).ravel())
    p["faces"] = mi.UInt32(np.asarray(faces, np.uint32).ravel())
    p.update()
    mesh.write_ply(str(path))


def _prop_bsdf(material, verts, paint):
    """The surface of one prop (props.dress), and texture coordinates if it needs them."""
    kind = material["kind"]
    if kind == "steel":
        # Zinc-plated or chromed steel: a metal shine over a matte grey that catches the
        # room's light (mirror metal shows only the dark desk round it, and reads black).
        shine = {"type": "roughconductor", "material": "Al", "distribution": "ggx", "alpha": float(material["rough"])}
        matte = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.51, 0.52]}}
        return {"type": "blendbsdf", "weight": 0.35, "bsdf_0": shine, "bsdf_1": matte}, None
    if kind == "graphite":
        # Pencil lead: dark grey with a soft metallic sheen.
        return {"type": "principled", "base_color": {"type": "rgb", "value": [0.1, 0.1, 0.105]}, "metallic": 0.45,
                "roughness": 0.42, "specular": 0.5}, None
    if kind == "chrome":
        # Chromed metal or metal-look plastic: mostly mirror, a little grey haze.
        shine = {"type": "roughconductor", "material": "Al", "distribution": "ggx", "alpha": float(material.get("rough", 0.06))}
        matte = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.4, 0.41, 0.42]}}
        return {"type": "blendbsdf", "weight": 0.25, "bsdf_0": shine, "bsdf_1": matte}, None
    if kind == "metal":
        return {"type": "roughconductor", "material": "Al", "distribution": "ggx", "alpha": float(material["rough"])}, None
    if kind == "matte":
        return {"type": "diffuse", "reflectance": {"type": "rgb", "value": material["colour"]}}, None
    if kind == "textured":
        # Lacquer or wood with its own texture (props give the texture coordinates), and
        # on top, where the mask says, stamped foil (gold or silver) or printed ink.
        base = {"type": "roughplastic", "distribution": "ggx", "alpha": float(material["rough"]), "nonlinear": True,
                "diffuse_reflectance": _texture(material["albedo"]), "specular_reflectance": float(material.get("specular", 0.4))}
        foil = material.get("foil")
        if foil is not None and float(foil.max()) > 0:
            if material.get("metal") in ("Au", "Ag"):
                top = {"type": "roughconductor", "material": material["metal"], "distribution": "ggx", "alpha": 0.2}
            else:
                top = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.03, 0.2, 0.07]}}
            base = {"type": "blendbsdf", "weight": _texture(np.ascontiguousarray(foil[..., None])), "bsdf_0": base, "bsdf_1": top}
        return base, "own"
    if kind == "metalpaint":
        # Metallic paint under a clear coat: a soft metal sheen, no glitter.
        return {"type": "principled", "base_color": {"type": "rgb", "value": material["colour"]}, "metallic": 0.3,
                "roughness": 0.45, "specular": 0.5, "clearcoat": 0.6, "clearcoat_gloss": 0.8}, None
    if kind == "pplastic":
        # Moulded plastic: glossy, its gloss uneven where the maps say (mould seams, scuffs,
        # finger marks), printing on it, and stamped foil where there is foil.
        if "albedo" in material:
            base = {"type": "principled", "base_color": _texture(material["albedo"]),
                    "roughness": _texture(np.ascontiguousarray(material["rough_map"][..., None])), "specular": 0.5}
            foil = material.get("foil")
            if foil is not None and float(foil.max()) > 0:
                top = {"type": "roughconductor", "material": material.get("metal", "Ag"), "distribution": "ggx", "alpha": 0.15}
                base = {"type": "blendbsdf", "weight": _texture(np.ascontiguousarray(foil[..., None])), "bsdf_0": base, "bsdf_1": top}
            return base, "own"
        return {"type": "principled", "base_color": {"type": "rgb", "value": material["colour"]},
                "roughness": float(material["rough"]), "specular": 0.5}, None
    if kind == "brass":
        shine = {"type": "roughconductor", "material": "Au", "distribution": "ggx", "alpha": float(material.get("rough", 0.2))}
        matte = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.45, 0.36, 0.16]}}
        return {"type": "blendbsdf", "weight": 0.4, "bsdf_0": shine, "bsdf_1": matte}, None
    if kind == "rubber":
        # Soft rubber: matte, with a faint broad sheen.
        return {"type": "roughplastic", "distribution": "ggx", "alpha": 0.55, "specular_reflectance": 0.25,
                "diffuse_reflectance": {"type": "rgb", "value": material["colour"]}, "nonlinear": True}, None
    if kind == "clear":
        # A clear or tinted plastic barrel. Refracting it for real leaves what is inside lit
        # only by paths that happen to find the lamp through it: a glitter of noise. Instead
        # light passes straight through (a null surface, which volpath lets the lamp's light
        # through too), under a plastic's own reflection; a tinted or frosted barrel scatters
        # some of the light in its colour.
        tint = material["tint"]
        cloud = float(material.get("cloud", 0.06 if min(tint) > 0.9 else 0.14))
        body = {"type": "blendbsdf", "weight": cloud, "bsdf_0": {"type": "null"},
                "bsdf_1": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.85 * c for c in tint]}}}
        shine = {"type": "roughconductor", "material": "none", "distribution": "ggx", "alpha": 0.04}
        return {"type": "blendbsdf", "weight": 0.1, "bsdf_0": body, "bsdf_1": shine}, None
    if kind == "plastic":
        return {"type": "roughplastic", "distribution": "ggx", "alpha": float(material["rough"]),
                "diffuse_reflectance": {"type": "rgb", "value": material["colour"]}, "nonlinear": True}, None
    if kind == "paper":
        return {"type": "twosided", "bsdf": {"type": "roughplastic", "distribution": "ggx", "alpha": 0.5,
                                             "diffuse_reflectance": _texture(material["albedo"]), "specular_reflectance": 0.3}}, "own"
    if kind == "masking tape":
        # The same surface model as the tape painted on the paper, so the two halves match.
        return {"type": "twosided", "bsdf": {"type": "roughplastic", "distribution": "ggx", "alpha": 0.5, "nonlinear": True,
                                             "diffuse_reflectance": {"type": "rgb", "value": material["colour"]},
                                             "specular_reflectance": 0.3}}, None
    # Clear tape: the desk's own texture seen through a smooth glossy film.
    tex, uv_of = paint
    u, v = uv_of(verts[:, 0], verts[:, 1])
    bsdf = {"type": "principled", "base_color": tex, "roughness": 0.4, "specular": 0.5, "clearcoat": 1.0, "clearcoat_gloss": 0.92}
    # Tape scatters a little light of its own: a faint pale haze over the desk.
    haze = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.85, 0.85, 0.82]}}
    bsdf = {"type": "blendbsdf", "weight": 0.1, "bsdf_0": bsdf, "bsdf_1": haze}
    return {"type": "twosided", "bsdf": bsdf}, np.column_stack([u, v]).astype(np.float32)


def _grid_mesh(name, x, y, z, u, v, path, keep=None):
    """A height field as triangles, one vertex per grid point, with texture coordinates; saved as PLY."""
    mi = _mi
    ny, nx = x.shape
    verts = np.stack([x, y, z], -1).reshape(-1, 3)
    uv = np.stack([u, v], -1).reshape(-1, 2)
    i = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    if keep is not None:
        faces = faces[keep.ravel()[faces].any(1)]
    mesh = mi.Mesh(name, len(verts), len(faces), has_vertex_texcoords=True)
    p = mi.traverse(mesh)
    p["vertex_positions"] = mi.Float(verts.astype(np.float32).ravel())
    p["vertex_texcoords"] = mi.Float(uv.astype(np.float32).ravel())
    p["faces"] = mi.UInt32(faces.astype(np.uint32).ravel())
    p.update()
    mesh.write_ply(str(path))


def _sheet_bsdf(sheet, st):
    """Paper: diffuse with a faint broad sheen. Ink: the same colour, with a tighter shine."""
    ppmm = sheet.ppmm
    n = _normals(sheet.relief, ppmm)
    # Tangent space follows the texture: u along x, v along y (down the page), normal up.
    normals = np.stack([n[0], n[1], n[2]], -1) * 0.5 + 0.5
    # Paper is nearly matte: its fibres scatter light every way, with only a faint broad
    # sheen (a glossy coat would read as plastic). The flash still finds a soft hot spot.
    # Matte paper reflects a low lamp's light as fully as overhead light (Lambert); a coated
    # plastic model sends much of it off in its coat's reflection at a slant, so the room
    # filled every shadow a pen cast. The sheen is a separate faint lobe on top.
    albedo = _texture(sheet.albedo)
    coat = {"type": "roughplastic", "distribution": "ggx", "alpha": float(np.clip(0.85 - st.sheen_power / 90, 0.5, 0.8)),
            "diffuse_reflectance": {"type": "rgb", "value": [0.0, 0.0, 0.0]}, "specular_reflectance": 1.0}
    paper = {"type": "blendbsdf", "weight": float(np.clip(st.sheen * 2.5, 0.05, 0.3)),
             "bsdf_0": {"type": "diffuse", "reflectance": albedo}, "bsdf_1": coat}
    surface = paper
    if sheet.sheen is not None:
        ink = {"type": "roughplastic", "distribution": "ggx", "alpha": 0.14, "diffuse_reflectance": albedo,
               "specular_reflectance": 1.0, "nonlinear": True}
        surface = {"type": "blendbsdf", "weight": _texture(np.clip(sheet.sheen, 0, 1)[..., None]), "bsdf_0": paper, "bsdf_1": ink}
    bumped = {"type": "normalmap", "normalmap": {"type": "bitmap", "bitmap": _bitmap(normals), "raw": True}, "bsdf": surface}
    return {"type": "mask", "opacity": _texture(sheet.mask[..., None]), "bsdf": {"type": "twosided", "bsdf": bumped}}


def _look_at(origin, target, up):
    return _mi.ScalarTransform4f().look_at(origin=[float(v) for v in origin], target=[float(v) for v in target], up=[float(v) for v in up])


def _rect(centre, normal, size, extra):
    """A rectangle of `size` mm centred at `centre`, facing along `normal`."""
    T = _mi.ScalarTransform4f
    up = [0, 0, 1] if abs(normal[2]) < 0.9 else [0, 1, 0]
    # A rectangle faces +z in its own frame; look_at points +z at the target.
    frame = T().look_at(origin=[float(v) for v in centre], target=[float(v) for v in np.asarray(centre) + normal], up=up)
    return dict({"type": "rectangle", "to_world": frame @ T().scale([size[0] / 2, size[1] / 2, 1])}, **extra)


def _desk_scene(st, surface, aspects, x0, x1, y0, y1, tmp):
    """The surface under the sheet: a textured mesh (wrinkled, for cloth), or the procedural desk."""
    T = _mi.ScalarTransform4f
    if not surface.textured:
        res = 6.0
        desk, _, _ = _desk(st, st.desk, x0, y0, x1, y1, res, aspects.rng("background"), surface)
        gloss = desk_gloss(st, surface)
        if surface.procedural == "glass":
            bsdf = {"type": "principled", "base_color": _texture(desk), "roughness": 0.03, "specular": 0.5}
        elif gloss > 0:
            bsdf = {"type": "roughplastic", "distribution": "ggx", "alpha": float(np.clip(0.45 - gloss, 0.08, 0.45)),
                    "diffuse_reflectance": _texture(desk), "nonlinear": True}
        else:
            bsdf = {"type": "diffuse", "reflectance": _texture(desk)}
        # The desk texture's first row is at y0, so v runs with y.
        paint = (_texture(desk), lambda X, Y: ((X - x0) / (x1 - x0), (Y - y0) / (y1 - y0)))
        return {"type": "rectangle", "bsdf": bsdf,
                "to_world": T().translate([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]) @ T().scale([(x1 - x0) / 2, (y1 - y0) / 2, 1])}, paint
    wrap = "clamp" if surface.cover is not None else "repeat"

    def tex(a):
        return {"type": "bitmap", "bitmap": _bitmap(a), "raw": True, "filter_type": "bilinear", "wrap_mode": wrap}

    bsdf = {"type": "principled", "base_color": tex(surface.colour), "roughness": tex(surface.rough[..., None]),
            "specular": 0.18 if surface.height is not None else 0.4,  # cloth fibres: a dull sheen, no glints
            "clearcoat": float(surface.coat), "clearcoat_gloss": float(surface.coat_gloss),
            "sheen": float(surface.sheen)}
    normal = surface.normal
    if surface.height is not None and not surface.true_normals:
        # Cloth maps are made for close-ups: from a phone's distance the weave is soft.
        normal = normal * np.array([0.5, 0.5, 1.0], np.float32)
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    bsdf = {"type": "normalmap", "normalmap": tex(normal * 0.5 + 0.5), "bsdf": bsdf}
    cell = (surfaces.FINE if surface.crinkle is not None else surfaces.CELL) if surface.height is not None else max(x1 - x0, y1 - y0)
    xs = np.r_[np.arange(x0, x1, cell), x1].astype(np.float32)
    ys = np.r_[np.arange(y0, y1, cell), y1].astype(np.float32)
    X, Y = np.meshgrid(xs, ys)
    U, V = surface.texcoords(X, Y)
    path = Path(tmp) / "desk.ply"
    _grid_mesh("desk", X, Y, surface.surface_at(X, Y).astype(np.float32), U.astype(np.float32), V.astype(np.float32), path)
    return {"type": "ply", "filename": str(path), "face_normals": False, "bsdf": bsdf}, (tex(surface.colour), surface.texcoords)


def _room(st, rng):
    """The room around the desk from its HDRI, scaled to give the light the room should; None
    offline. Its brightest spots are tamed, so the main light stays the lamp, and it is
    turned so its own bright side lies the same way as the lamp."""
    path = surfaces.room(st.room) if st.room else None
    if path is None:
        return None, st.ambient_tint
    L = np.array(_mi.Bitmap(str(path))).astype(np.float32)[..., :3]
    h, w = L.shape[:2]
    theta = (np.arange(h) + 0.5) / h * np.pi  # from straight up (row 0) to straight down
    upper = np.clip(np.cos(theta), 0, None) * np.sin(theta)  # cosine-weighted, onto an upward page
    lum = L.mean(-1)
    typical = (lum * np.sin(theta)[:, None]).sum() / (np.sin(theta).sum() * w)
    L = L * np.minimum(1.0, 8 * typical / np.maximum(lum, 1e-6))[..., None]
    # Phones balance out the room's colour as well as the lamp's: what lies in shade turns
    # a little cool or warm, not brown. Keep half the room's colour.
    grey = L.mean(-1, keepdims=True)
    L = grey + 0.5 * (L - grey)
    E = (L * upper[:, None, None]).sum((0, 1)) * (np.pi / h) * (2 * np.pi / w)
    L *= (1 - st.key) / E.mean()
    # Mitsuba's latlong puts column u at phi = 2 pi u, which, turned z-up, faces azimuth pi/2 - phi.
    bright = (blur(L.mean(-1), 4.0) * upper[:, None]).sum(0)
    phi = 2 * np.pi * (np.argmax(bright) + 0.5) / w
    toward = np.arctan2(st.light[1], st.light[0])
    turn = np.degrees(toward - (np.pi / 2 - phi) + rng.normal(0, 0.3))
    T = _mi.ScalarTransform4f
    env = {"type": "envmap", "bitmap": _bitmap(L), "to_world": T().rotate([0, 0, 1], float(turn)) @ T().rotate([1, 0, 0], 90.0)}
    return env, (E / E.mean()).astype(np.float32)


def _lights(st, wmm, hmm, occ, room=None, camera_z=300.0):
    """The main light and the room, scaled so the page centre gets about the irradiance _light gives."""
    centre = np.array([wmm / 2, hmm / 2, 0.0])
    lamp = centre + st.light
    to_page = centre - lamp
    r = np.linalg.norm(to_page)
    size = 8.0 if st.kind == "flash" else LAMP_MM["raking"] if st.kind == "raking" else LAMP_MM["lamp"] if r < 1000 and st.key > 0.55 else LAMP_MM["overhead"] if st.light[2] > 1100 and st.key > 0.4 else LAMP_MM["window"]
    if size == LAMP_MM["overhead"]:
        # Ceiling lights run from a bare bulb to a wide shade: 9 to 26 cm (higher ones bigger),
        # so a hand's shadow is now readable, now a soft blur.
        size *= 0.4 + 0.8 * float(np.clip((st.light[2] - 1200) / 600, 0, 1))
    cos_page = st.light[2] / r  # light arriving at the page centre, off vertical
    area = size * size
    # Irradiance E = L * A * cos(light) * cos(page) / r^2 for a small source facing the page.
    radiance = st.key / (area * cos_page / r**2) * st.key_tint
    scene = {
        "lamp": _rect(lamp, to_page / r, (size, size), {"emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance.tolist()}}}),
        # The rest of the room: E = pi * L from a whole sky of it.
        "room": room or {"type": "constant", "radiance": {"type": "rgb", "value": ((1 - st.key) / np.pi * st.ambient_tint).tolist()}},
    }
    if occ is not None and lamp[2] > camera_z + 60:
        # The phone and hand that took the picture: just above the lens, so never in the
        # picture themselves, on the line from the lamp to where camera._occluder put their
        # shadow. Sized so the shadow is as large as one cast from 16 cm up would be.
        height = camera_z + 45.0  # the hand reaches a little below the phone: keep it all above the lens
        target = np.array([occ["cx"], occ["cy"], 0.0])
        where = target + (lamp - target) * (height / lamp[2])
        T = _mi.ScalarTransform4f
        grow = (lamp[2] - height) / (lamp[2] - 160.0)
        hx, hy = occ["half"][0] * grow * 0.8, occ["half"][1] * grow * 0.8
        place = T().translate([float(v) for v in where]) @ T().rotate([0, 0, 1], float(np.degrees(occ["angle"])))
        dark = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.04, 0.04, 0.045]}}
        scene["phone"] = {"type": "cube", "to_world": place @ T().scale([float(hx), float(hy), 4.0]), "bsdf": dark}
        # Its rounded corners: a cylinder at each, so the shadow's corners are round too.
        rc = min(0.18 * hx, 12.0)
        scene["phone"]["to_world"] = place @ T().scale([float(hx - rc), float(hy), 4.0])
        scene["phone_w"] = {"type": "cube", "to_world": place @ T().scale([float(hx), float(hy - rc), 4.0]), "bsdf": dark}
        for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            scene[f"phone_c{k}"] = {"type": "cylinder", "radius": float(rc), "p0": [0.0, 0.0, -4.0], "p1": [0.0, 0.0, 4.0], "bsdf": dark,
                                    "to_world": place @ T().translate([float(sx * (hx - rc)), float(sy * (hy - rc)), 0.0])}
        # The hand: ellipsoids as meshes (Mitsuba's spheres can't be stretched), written out
        # with the props by photograph.
        skin = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.3, 0.2, 0.16]}}
        th, ph = np.meshgrid(np.linspace(0, np.pi, 13), np.linspace(0, 2 * np.pi, 25)[:-1], indexing="ij")
        unit = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
        meshes = []
        for k, (ex, ey, rx, ry, turn, z, rz) in enumerate(occ.get("hand", [])):
            M = np.array((place @ T().translate([float(ex * hx), float(ey * hy), float(z)]) @ T().rotate([0, 0, 1], float(np.degrees(turn)))
                          @ T().scale([float(rx * hx), float(ry * hy), float(rz)])).matrix, np.float64)
            pts = unit.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            v, f = _grid_faces(pts.reshape(unit.shape))
            meshes.append((f"hand{k}", v, f, skin))
        scene["_meshes"] = meshes
    return scene


def _grid_faces(P):
    """A closed lat-long grid of points (rows pole to pole, columns round): (verts, faces)."""
    n, m, _ = P.shape
    verts = P.reshape(-1, 3).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            faces += [(a, c, b), (b, c, d)]
    return verts, np.array(faces, np.uint32)


def _camera(shot, w, h):
    """Mitsuba's perspective camera for camera.Shot: the same pinhole, focal length and centre.

    Mitsuba's camera frame is mirrored left to right against camera.py's and counts its
    principal point upwards; photograph flips the image back. Checked by overlap: the
    sheet lands on the same pixels in both, to 99.5%.
    """
    fov = float(np.degrees(2 * np.arctan(w / 2 / shot.scale)))
    return {
        "type": "perspective", "fov": fov, "fov_axis": "x",
        "principal_point_offset_x": float((shot.offset[0] - w / 2) / w),
        "principal_point_offset_y": float(-(shot.offset[1] - h / 2) / h),
        "to_world": _look_at(shot.centre, shot.centre + shot.forward, -shot.down),
        "film": {"type": "hdrfilm", "width": w, "height": h, "pixel_format": "rgb", "rfilter": {"type": "gaussian"}},
        "sampler": {"type": "independent", "sample_count": SAMPLES},
    }


def _no_fireflies(image, k=4.0):
    """Rare light paths leave single pixels far brighter than anything round them (light
    leaking under a lifted sheet onto a glossy desk); hold each to k times its neighbours'
    median, so the denoiser doesn't spread them into speckles."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(image).permute(2, 0, 1)[None]
    patches = F.unfold(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3)  # 1 x C*9 x H*W
    c = t.shape[1]
    median = patches.view(1, c, 9, -1).median(2).values.view_as(t)
    lum = t.mean(1, keepdim=True)
    limit = k * median.mean(1, keepdim=True) + 0.02
    out = torch.where(lum > limit, median, t)
    # They also come in clumps of a few pixels, which a 3x3 median keeps. Anything bright
    # and narrower than 5 pixels is held to k times what is left once it is taken away
    # (a morphological opening); a sheet's corner is far wider, and stays.
    lum = out.mean(1, keepdim=True)
    opened = F.max_pool2d(-F.max_pool2d(-lum, 5, 1, 2), 5, 1, 2)
    out = out * torch.clamp((k * opened + 0.02) / lum.clamp_min(1e-6), max=1.0)
    return out[0].permute(1, 2, 0).contiguous().numpy()


def photograph(sheet, st, aspects, page):
    """The phone photo of `sheet`, traced in 3D. Same inputs and output as camera.photograph."""
    mi = _mi
    rng = aspects.rng("camera", page)
    h, w = sheet.mask.shape
    wmm, hmm = sheet.width_mm, sheet.height_mm
    surface = surfaces.prepare(st, aspects, page, wmm, hmm)
    sheet = surface.drape(sheet)
    things = []
    kinds = props.choose(aspects.value("props"), sheet.wear, aspects.rng("props", page))
    if kinds:
        sheet, things = props.dress(sheet, kinds, aspects.rng("props", page, part="dress"), surface.surface_at)
    if sheet.erasures:
        things += props.crumbs(sheet, aspects.rng("props", page, part="crumbs"))
    occ = _occluder(aspects.rng("shadow", page), wmm, hmm, aspects.value("shadow"))
    shot = _frame(sheet, w, h, aspects.rng("angle", page), "phone", aspects.value("angle"),
                  aspects.value("framing"), aspects.rng("framing", page))
    if st.kind == "flash":
        st, occ = with_flash(st, shot, wmm, hmm), None
    sx, sy, ex, ey, focus = footprint(sheet, shot, "phone", w, h)
    on_paper = sample(sheet.mask, sx * sheet.ppmm, sy * sheet.ppmm, fill=0.0)

    x0, x1, y0, y1 = desk_bounds(ex, ey, wmm, hmm)
    if surface.true_normals and surface.height is not None:
        # Loose threads and lint on the cloth, where the camera sees them.
        # Not under the sheet: there it would be pressed flat and hidden, not poke up through it.
        for stray in fabric.strays(surface, aspects.rng("background", part=f"strays {page}"), (x0, x1, y0, y1),
                                   surface.colour.reshape(-1, 3).mean(0)):
            v = stray["verts"]
            if not (sample(sheet.mask, v[:, 0] * sheet.ppmm, v[:, 1] * sheet.ppmm, fill=0.0) > 0.01).any():
                things.append(stray)
    x0, x1, y0, y1 = x0 - 300, x1 + 300, y0 - 300, y1 + 300  # past the frame, for what reflects and bounces
    room, room_tint = _room(st, aspects.rng("light", part="room"))
    scene = {
        "type": "scene",
        "integrator": {"type": "volpath", "max_depth": 6},  # volpath: lamp light passes null surfaces (clear barrels, cut-outs)
        "sensor": _camera(shot, w, h),
        **_lights(st, wmm, hmm, occ, room, float(shot.centre[2])),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, v, f, bsdf in scene.pop("_meshes", []):
            path = Path(tmp) / f"{name}.ply"
            _mesh(name, v, f, None, path)
            scene[name] = {"type": "ply", "filename": str(path), "face_normals": False, "bsdf": {"type": "twosided", "bsdf": bsdf}}
        scene["desk"], paint = _desk_scene(st, surface, aspects, x0, x1, y0, y1, tmp)
        for i, thing in enumerate(things):
            bsdf, uv = _prop_bsdf(thing["material"], thing["verts"], paint)
            if isinstance(uv, str):
                uv = thing["uv"]
            path = Path(tmp) / f"prop{i}.ply"
            _mesh(f"prop{i}", thing["verts"], thing["faces"], uv, path)
            if thing.get("twosided"):
                bsdf = {"type": "twosided", "bsdf": bsdf}
            scene[f"prop{i}"] = {"type": "ply", "filename": str(path), "face_normals": bool(thing.get("flat")), "bsdf": bsdf}
        ply = Path(tmp) / "sheet.ply"
        _sheet_mesh(sheet, ply)
        scene["sheet"] = {"type": "ply", "filename": str(ply), "face_normals": False, "bsdf": _sheet_bsdf(sheet, st)}
        loaded = mi.load_dict(scene)
    traced = mi.render(loaded, seed=int(rng.integers(2**31)))
    traced = mi.TensorXf(_no_fireflies(np.array(traced).astype(np.float32)))
    try:
        # What noise the paths leave sits mostly in the shade; the phone adds its own later.
        traced = mi.OptixDenoiser(input_size=traced.shape[:2][::-1])(traced)
    except (AttributeError, RuntimeError):
        pass
    image = np.array(traced).astype(np.float32)[:, ::-1].copy()
    cast = (st.key * st.key_tint + (1 - st.key) * room_tint).astype(np.float32)
    return _develop(image, on_paper, st, "phone", rng, w, h, focus, cast)

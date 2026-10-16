"""
Steps 9-12 (first version) - turn one scan into a room profile.

Input : data/scans/<id>/ (scan.json, keyframes, SegFormer masks cached by walls.py)
Output: data/scans/<id>/room_profile.json + room_topdown.png + free_grid.npy

  9  geometry   floor height (ARCore floor plane); wall feet = wall/floor boundary of the
                segmentation, ray -> floor plane; Manhattan assumption: find the dominant
                direction, then one wall per side (strongest line of wall feet on that side).
                A side without enough evidence gets a PREDICTED wall (beyond the last floor seen).
                Doors/windows: their pixels' rays hit the wall planes -> position, width, height.
 10  objects    floor-standing furniture from the segmentation: per image column, the lowest pixel
                of the object -> ray -> floor = where it stands, the top pixel -> its height.
                Only the front edge is seen, so the depth is PREDICTED: up to the wall if the object
                is against one, else a typical depth for that kind of furniture.
 11  semantics  Gemini (vision) on 4 keyframes; rule-based fallback (objects -> room type,
                pixel colours of walls/floor).
 12  free space 5 cm occupancy grid (objects + door swing), distance to walls, then the free
                rectangles the recommender (step 7) uses: strips along each wall + open floor areas.
                Unseen floor is assumed free and the zones on it are marked as predicted.

Everything is in the coordinates of the AR session that made the scan (Unity world, metres),
plus a room frame: origin at one corner, x along one wall, z along the other.
Usage: python backend/space/room_profile.py <scan_id>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import walls as W  # noqa: E402  (also puts backend/server on sys.path)

ROOT = W.ROOT
RUG, DOOR, WINDOW, TV, PAINTING = 28, 14, 8, 89, 22
# ADE20K class -> (our label, step-7 category, typical depth m, typical height m)
FURNITURE = {
    23: ("sofa", "sofa", 0.90, 0.80), 30: ("armchair", "armchair", 0.80, 0.85),
    10: ("cabinet", "sideboard", 0.45, 0.80), 24: ("shelf", "bookcase_shelf", 0.35, 1.60),
    15: ("table", "dining_table", 0.80, 0.75), 64: ("coffee table", "coffee_side_table", 0.60, 0.45),
    19: ("chair", "dining_chair", 0.50, 0.85), 7: ("bed", "bed", 2.00, 0.60),
    33: ("desk", "desk", 0.60, 0.75), 35: ("wardrobe", "wardrobe", 0.60, 2.00),
    44: ("chest of drawers", "chest_of_drawers", 0.45, 0.90), 36: ("lamp", "floor_lamp", 0.40, 1.50),
    62: ("bookcase", "bookcase_shelf", 0.35, 1.80), 69: ("bench", "sideboard", 0.40, 0.50),
    31: ("seat", "armchair", 0.80, 0.85),
}
SEATING = {"sofa", "armchair", "seat"}
LOW_STORAGE = {"cabinet", "table", "coffee table", "bench", "chest of drawers"}
GRID = 0.05


# ---------------------------------------------------------------- loading
def load(scan_id):
    import app as server
    folder = W.SCANS / scan_id
    bundle = server.ScanBundle.model_validate_json((folder / "scan.json").read_text(encoding="utf-8"))
    masks = [(kf, W.segment(folder / "keyframes" / kf.file, folder / "seg" / kf.file.replace(".jpg", ".png")))
             for kf in bundle.keyframes if (folder / "keyframes" / kf.file).exists()]
    floor_y, _ = W.floor_geometry(bundle)
    return folder, bundle, masks, floor_y, W.DEFAULT_MAPPING


def pixel_rays(kf, mask, xs, ys, mapping):
    return W.rays(kf, np.asarray(xs, float), np.asarray(ys, float), mapping, mask.shape)


def floor_points(kf, mask, mapping, floor_y, step=8):
    """Floor (and rug) pixels -> where they are on the floor: the floor area actually seen."""
    sub = mask[::step, ::step]
    ys, xs = np.nonzero((sub == W.FLOOR) | (sub == RUG))
    if not len(xs):
        return np.zeros((0, 2))
    d, o = pixel_rays(kf, mask, xs * step, ys * step, mapping)
    p = W.hit_floor(o, d, floor_y)
    return p[~np.isnan(p[:, 0])]


# ---------------------------------------------------------------- step 9: Manhattan walls
def rotation(theta_deg):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # world xz -> room-aligned xz


def manhattan_angle(pts):
    """Direction (0-90 deg) that makes the wall feet line up best with the axes: the 5 cm histograms
    of both rotated coordinates are most 'peaky' (sum of squared bin shares is largest)."""
    best = (0.0, -1.0)
    for th in np.arange(0, 90, 0.5):
        q = pts @ rotation(th).T
        s = 0.0
        for a in (0, 1):
            h, _ = np.histogram(q[:, a], bins=np.arange(q[:, a].min(), q[:, a].max() + GRID, GRID))
            s += ((h / len(q)) ** 2).sum()
        if s > best[1]:
            best = (th, s)
    return best[0]


def side_walls(c, floor_c, min_support=40):
    """One wall on the low side and one on the high side of one axis (rotated coordinates).
    The wall = the strongest 5 cm line of wall feet in that half (drift can draw the same wall twice;
    the strongest copy wins). Weak evidence -> predicted just beyond the farthest floor seen."""
    from scipy.ndimage import uniform_filter1d
    edges = np.arange(c.min() - GRID, c.max() + 2 * GRID, GRID)
    h, _ = np.histogram(c, bins=edges)
    h = uniform_filter1d(h.astype(float), 3)
    centers = (edges[:-1] + edges[1:]) / 2
    mid = np.median(floor_c) if len(floor_c) else np.median(c)
    out = []
    for side, sel in (("low", centers < mid), ("high", centers >= mid)):
        k = np.argmax(np.where(sel, h, -1))
        pos = centers[k]
        support = int(np.sum(np.abs(c - pos) < 0.10))
        if support >= min_support:
            out.append({"pos": float(pos), "support": support, "predicted": False})
        else:
            far = (np.percentile(floor_c, 1) - 0.1) if side == "low" else (np.percentile(floor_c, 99) + 0.1)
            out.append({"pos": float(far), "support": support, "predicted": True})
    return out


def room_frame(foot, floor):
    theta = manhattan_angle(foot)
    R = rotation(theta)
    fr, flr = foot @ R.T, floor @ R.T
    xs, zs = side_walls(fr[:, 0], flr[:, 0]), side_walls(fr[:, 1], flr[:, 1])
    # a room is at least 2.2 m on a side: if a predicted wall makes it smaller, push it out
    for ws in (xs, zs):
        if ws[1]["pos"] - ws[0]["pos"] < 2.2:
            k = 1 if ws[1]["predicted"] else 0
            ws[k]["pos"] = ws[1 - k]["pos"] + (2.2 if k == 1 else -2.2)
    return theta, R, xs, zs


class Frame:
    """room-aligned frame: origin at the (x_min, z_min) corner, metres."""
    def __init__(self, theta, R, xs, zs):
        self.theta, self.R = theta, R
        self.x0, self.x1, self.z0, self.z1 = xs[0]["pos"], xs[1]["pos"], zs[0]["pos"], zs[1]["pos"]
        self.width, self.depth = self.x1 - self.x0, self.z1 - self.z0
        self.walls = {"left": xs[0], "right": xs[1], "bottom": zs[0], "top": zs[1]}

    def to_room(self, xz):
        q = np.atleast_2d(xz) @ self.R.T
        return np.stack([q[:, 0] - self.x0, q[:, 1] - self.z0], 1)

    def to_world(self, rxz):
        q = np.atleast_2d(rxz) + [self.x0, self.z0]
        return q @ self.R                  # R is orthonormal: inverse = transpose

    def world_dir(self, rdir):
        return np.atleast_2d(rdir) @ self.R


def wall_hits(o, d, fr):
    """Rays (world) -> first hit with one of the 4 walls. -> list of (wall, u along wall, height y)."""
    oo = fr.to_room(o[[0, 2]])[0]
    dd = (d[:, [0, 2]] @ fr.R.T)
    res = []
    for k in range(len(d)):
        best = None
        for wall, axis, val in (("left", 0, 0.0), ("right", 0, fr.width), ("bottom", 1, 0.0), ("top", 1, fr.depth)):
            if abs(dd[k, axis]) < 1e-6:
                continue
            t = (val - oo[axis]) / dd[k, axis]
            if t <= 0:
                continue
            p = oo + t * dd[k]
            u = p[1 - axis]
            lim = fr.depth if axis == 0 else fr.width
            if -0.1 <= u <= lim + 0.1 and (best is None or t < best[0]):
                best = (t, wall, float(u), float(o[1] + t * d[k, 1]))
        if best:
            res.append(best[1:])
    return res


def openings(masks, mapping, fr, floor_y, step=6):
    found = []
    for cls, kind in ((DOOR, "door"), (WINDOW, "window")):
        hits = []
        for kf, mask in masks:
            sub = mask[::step, ::step]
            ys, xs = np.nonzero(sub == cls)
            if len(xs) < 20:
                continue
            d, o = pixel_rays(kf, mask, xs * step, ys * step, mapping)
            hits += wall_hits(o, d, fr)
        for wall in ("left", "right", "bottom", "top"):
            u = np.array([h[1] for h in hits if h[0] == wall])
            y = np.array([h[2] for h in hits if h[0] == wall]) - floor_y
            if len(u) < 60:
                continue
            order = np.argsort(u)
            u, y = u[order], y[order]
            cuts = np.where(np.diff(u) > 0.3)[0] + 1        # separate openings on one wall
            for gu, gy in zip(np.split(u, cuts), np.split(y, cuts)):
                if len(gu) < 60:
                    continue
                a, b = np.percentile(gu, 3), np.percentile(gu, 97)
                bottom, top = float(np.percentile(gy, 3)), float(np.percentile(gy, 97))
                if b - a < 0.4:
                    continue
                found.append({"type": kind, "wall": wall, "u0": round(float(a), 2), "u1": round(float(b), 2),
                              "width": round(float(b - a), 2),
                              "bottom": round(max(0.0, bottom), 2) if kind == "window" else 0.0,
                              "top": round(top, 2), "evidence_px": int(len(gu))})
    return found


# ---------------------------------------------------------------- step 10: furniture
def furniture_points(masks, mapping, floor_y, step=4):
    """Per class: foot points (world xz) + heights from the columns of each object's pixels."""
    acc = {}
    for kf, mask in masks:
        H, Wd = mask.shape
        for cls in FURNITURE:
            m = mask == cls
            if m.sum() < 300:
                continue
            cols = [x for x in range(0, Wd, step) if m[:, x].any()]
            xb, yb, yt = [], [], []
            for x in cols:
                ys = np.nonzero(m[:, x])[0]
                bottom, top = ys.max(), ys.min()
                if bottom >= H - 3:                         # foot cut off by the image border
                    continue
                xb.append(x)
                yb.append(bottom + 1)
                yt.append(top if top > 2 else -1)
            if len(xb) < 5:
                continue
            d, o = pixel_rays(kf, mask, xb, yb, mapping)
            foot = W.hit_floor(o, d, floor_y)
            ok = ~np.isnan(foot[:, 0])
            heights = np.full(len(xb), np.nan)
            top_ok = np.array(yt) >= 0
            if top_ok.any():
                dt, _ = pixel_rays(kf, mask, np.array(xb)[top_ok], np.array(yt)[top_ok], mapping)
                fo = foot[top_ok]
                horiz = np.linalg.norm(fo - o[[0, 2]], axis=1)
                t = horiz / np.maximum(np.linalg.norm(dt[:, [0, 2]], axis=1), 1e-6)
                heights[top_ok] = o[1] + t * dt[:, 1] - floor_y
            a = acc.setdefault(cls, {"foot": [], "h": []})
            a["foot"].append(foot[ok])
            a["h"].append(heights[ok])
    return {c: (np.vstack(v["foot"]), np.concatenate(v["h"])) for c, v in acc.items() if v["foot"]}


def objects_from_points(fp, fr, min_points=40):
    from scipy.ndimage import binary_dilation, label
    objs = []
    for cls, (foot, hts) in fp.items():
        name, cat, typ_d, typ_h = FURNITURE[cls]
        r = fr.to_room(foot)
        inside = (r[:, 0] > -0.2) & (r[:, 0] < fr.width + 0.2) & (r[:, 1] > -0.2) & (r[:, 1] < fr.depth + 0.2)
        r, hts = r[inside], hts[inside]
        if len(r) < min_points:
            continue
        g = np.floor(r / 0.10).astype(int)
        g -= g.min(0)
        occ = np.zeros(g.max(0) + 1, int)
        np.add.at(occ, (g[:, 0], g[:, 1]), 1)
        lab, n = label(binary_dilation(occ >= 2, iterations=2))
        for k in range(1, n + 1):
            sel = lab[g[:, 0], g[:, 1]] == k
            if sel.sum() < min_points:
                continue
            p = r[sel]
            x0, x1 = np.percentile(p[:, 0], [5, 95])
            z0, z1 = np.percentile(p[:, 1], [5, 95])
            # only the front edge is seen: the thin side is the depth we have to predict
            depth_seen_x, depth_seen_z = x1 - x0, z1 - z0
            along_x = depth_seen_x >= depth_seen_z
            predicted = False
            # snap to the nearest wall behind it if close, else typical depth
            if along_x:
                dist_b, dist_t = z0, fr.depth - z1
                if min(dist_b, dist_t) < 1.0:
                    if dist_b < dist_t:
                        z0 = 0.0
                        z1 = max(z1, 0.8 * typ_d)
                    else:
                        z1 = fr.depth
                        z0 = min(z0, fr.depth - 0.8 * typ_d)
                elif z1 - z0 < typ_d * 0.7:
                    z1 = z0 + typ_d
                predicted = True
            else:
                dist_l, dist_r = x0, fr.width - x1
                if min(dist_l, dist_r) < 1.0:
                    if dist_l < dist_r:
                        x0 = 0.0
                        x1 = max(x1, 0.8 * typ_d)
                    else:
                        x1 = fr.width
                        x0 = min(x0, fr.width - 0.8 * typ_d)
                elif x1 - x0 < typ_d * 0.7:
                    x1 = x0 + typ_d
                predicted = True
            # never deeper than ~1.6 typical depth from the seen edge
            if along_x and z1 - z0 > 1.6 * typ_d:
                z0, z1 = (z1 - typ_d, z1) if z1 >= fr.depth - 0.01 else (z0, z0 + typ_d)
            if not along_x and x1 - x0 > 1.6 * typ_d:
                x0, x1 = (x1 - typ_d, x1) if x1 >= fr.width - 0.01 else (x0, x0 + typ_d)
            h = hts[sel]
            h = h[~np.isnan(h) & (h > 0.05) & (h < 2.6)]
            height = float(np.median(h)) if len(h) >= 10 else typ_h
            objs.append({"label": name, "category": cat,
                         "x0": round(max(0.0, float(x0)), 2), "z0": round(max(0.0, float(z0)), 2),
                         "x1": round(min(fr.width, float(x1)), 2), "z1": round(min(fr.depth, float(z1)), 2),
                         "height": round(height, 2), "height_predicted": len(h) < 10,
                         "depth_predicted": predicted, "evidence_points": int(sel.sum())})
    return objs


def merge_objects(objs, tv_spots):
    """One real object is often split in several classes (TV bench = cabinet + table + coffee table,
    a sofa = sofa + armchair pieces). Merge boxes that overlap (>= 40% of the smaller) or touch
    (gap < 15 cm) inside the same family; keep the label with the most evidence. A low storage piece
    under a TV becomes a TV bench."""
    def family(o):
        return "seat" if o["label"] in SEATING else "low" if o["label"] in LOW_STORAGE else o["label"]

    def close(a, b):
        ix = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
        iz = min(a["z1"], b["z1"]) - max(a["z0"], b["z0"])
        area = lambda o: max(1e-6, (o["x1"] - o["x0"]) * (o["z1"] - o["z0"]))
        if ix > 0 and iz > 0 and ix * iz >= 0.4 * min(area(a), area(b)):
            return True
        return family(a) == family(b) and ix > -0.15 and iz > -0.15

    groups = []
    for o in sorted(objs, key=lambda o: -o["evidence_points"]):
        for g in groups:
            if any(close(o, m) for m in g):
                g.append(o)
                break
        else:
            groups.append([o])
    out = []
    for g in groups:
        main = g[0]
        m = dict(main)
        for k, f in (("x0", min), ("z0", min), ("x1", max), ("z1", max)):
            m[k] = round(f(o[k] for o in g), 2)
        m["evidence_points"] = sum(o["evidence_points"] for o in g)
        if len(g) > 1:
            m["merged_from"] = [o["label"] for o in g]
        if family(main) == "seat" and (m["x1"] - m["x0"] > 1.2 or m["z1"] - m["z0"] > 1.2):
            m["label"], m["category"] = "sofa", "sofa"
        if family(main) == "low" and any(m["x0"] - 0.2 <= x <= m["x1"] + 0.2 and m["z0"] - 0.2 <= z <= m["z1"] + 0.2
                                         for x, z in tv_spots):
            m["label"], m["category"] = "tv bench", "tv_bench"
            m["has_tv"] = True
        out.append(m)
    return out


def tv_spots(masks, mapping, fr):
    """Where TVs are (room xz, on the nearest wall hit by their pixels' rays)."""
    pts = []
    for kf, mask in masks:
        sub = mask[::6, ::6]
        ys, xs = np.nonzero(sub == TV)
        if len(xs) < 15:
            continue
        d, o = pixel_rays(kf, mask, xs * 6, ys * 6, mapping)
        for wall, u, _ in wall_hits(o, d, fr):
            pts.append({"left": (0.0, u), "right": (fr.width, u), "bottom": (u, 0.0), "top": (u, fr.depth)}[wall])
    if len(pts) < 30:
        return []
    a = np.array(pts)
    return [tuple(np.median(a, 0))]


def rug_area(masks, mapping, floor_y, fr):
    pts = []
    for kf, mask in masks:
        sub = mask[::6, ::6]
        ys, xs = np.nonzero(sub == RUG)
        if len(xs) < 30:
            continue
        d, o = pixel_rays(kf, mask, xs * 6, ys * 6, mapping)
        p = W.hit_floor(o, d, floor_y)
        pts.append(p[~np.isnan(p[:, 0])])
    if not pts:
        return None
    r = fr.to_room(np.vstack(pts))
    if len(r) < 200:
        return None
    x0, x1 = np.percentile(r[:, 0], [3, 97])
    z0, z1 = np.percentile(r[:, 1], [3, 97])
    return {"label": "rug", "category": "rug", "x0": round(float(x0), 2), "z0": round(float(z0), 2),
            "x1": round(float(x1), 2), "z1": round(float(z1), 2), "height": 0.01, "walkable": True}


# ---------------------------------------------------------------- step 11: semantics
SEM_SYSTEM = ("You describe a room from phone photos for an interior-design app. Answer in JSON. "
              "room_type one of living_room, bedroom, dining_room, office, kitchen, other. "
              "observations: up to 3 short Vietnamese sentences useful for choosing new furniture "
              "(free corners, light, style).")


def semantics(folder, masks, objects):
    from PIL import Image
    from pydantic import BaseModel
    from typing import List

    class Sem(BaseModel):
        room_type: str
        style: str
        dominant_colors: List[str]
        floor_material: str
        lighting: str
        existing_furniture: List[str]
        observations: List[str]

    picks = [masks[i][0] for i in np.linspace(0, len(masks) - 1, 4).astype(int)]
    try:
        from google.genai import types
        sys.path.insert(0, str(ROOT / "backend"))
        import llm
        parts = []
        for kf in picks:
            im = W.upright(Image.open(folder / "keyframes" / kf.file).convert("RGB"))
            import io
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        parts.append("Objects measured by the app: " + ", ".join(o["label"] for o in objects))
        cfg = types.GenerateContentConfig(system_instruction=SEM_SYSTEM, response_mime_type="application/json",
                                          response_schema=Sem, temperature=0.0)
        last = None
        for model in llm.FALLBACK_MODELS:
            try:
                r = llm.client().models.generate_content(model=model, contents=parts, config=cfg)
                out = Sem.model_validate_json(r.text).model_dump()
                out["source"] = model
                return out
            except Exception as e:                     # quota / overload / timeout -> next model
                last = e
        raise RuntimeError(str(last)[:160])
    except Exception as e:
        return rule_semantics(folder, masks, objects, f"Gemini unavailable: {str(e)[:120]}")


def rule_semantics(folder, masks, objects, why):
    from PIL import Image
    labels = {o["label"] for o in objects}
    room_type = ("bedroom" if "bed" in labels else "living_room" if labels & {"sofa", "armchair"} else
                 "office" if "desk" in labels else "dining_room" if "table" in labels else "other")
    cols = {}
    for name, cls in (("wall", W.WALL), ("floor", W.FLOOR)):
        px = []
        for kf, mask in masks[::4]:
            im = np.array(W.upright(Image.open(folder / "keyframes" / kf.file).convert("RGB")))
            px.append(im[mask == cls])
        px = np.vstack(px) if px else np.zeros((0, 3))
        cols[name] = [int(v) for v in np.median(px, 0)] if len(px) else None
    return {"room_type": room_type, "style": "unknown", "dominant_colors_rgb": cols,
            "floor_material": "unknown", "lighting": "unknown",
            "existing_furniture": sorted(labels), "observations": [], "source": "rules", "note": why}


# ---------------------------------------------------------------- step 12: free space + zones
def free_space(fr, objects, ops, seen_floor_room):
    from scipy.ndimage import distance_transform_edt
    nx, nz = int(np.ceil(fr.width / GRID)), int(np.ceil(fr.depth / GRID))
    occ = np.zeros((nx, nz), bool)
    for o in objects:
        if o.get("walkable"):
            continue
        a = np.clip(np.floor(np.array([o["x0"], o["z0"]]) / GRID).astype(int) - 1, 0, [nx - 1, nz - 1])
        b = np.clip(np.ceil(np.array([o["x1"], o["z1"]]) / GRID).astype(int) + 1, 0, [nx, nz])
        occ[a[0]:b[0], a[1]:b[1]] = True
    keep = np.zeros_like(occ)                 # door swing: keep 0.8 m in front of doors clear
    for op in ops:
        if op["type"] != "door":
            continue
        u0, u1 = op["u0"] - 0.05, op["u1"] + 0.05
        if op["wall"] in ("bottom", "top"):
            xs = slice(max(0, int(u0 / GRID)), min(nx, int(u1 / GRID) + 1))
            zs = slice(0, int(0.8 / GRID)) if op["wall"] == "bottom" else slice(max(0, nz - int(0.8 / GRID)), nz)
        else:
            zs = slice(max(0, int(u0 / GRID)), min(nz, int(u1 / GRID) + 1))
            xs = slice(0, int(0.8 / GRID)) if op["wall"] == "left" else slice(max(0, nx - int(0.8 / GRID)), nx)
        keep[xs, zs] = True
    seen = np.zeros_like(occ)
    if len(seen_floor_room):
        g = np.floor(seen_floor_room / GRID).astype(int)
        ok = (g[:, 0] >= 0) & (g[:, 0] < nx) & (g[:, 1] >= 0) & (g[:, 1] < nz)
        seen[g[ok, 0], g[ok, 1]] = True
        from scipy.ndimage import binary_closing
        seen = binary_closing(seen, iterations=3)
    free = ~occ & ~keep
    dist_obj = distance_transform_edt(~occ) * GRID
    return free, occ, keep, seen, dist_obj


def largest_rect(mask):
    """Largest all-True axis-aligned rectangle (histogram method). -> (i0, j0, i1, j1) or None"""
    n, m = mask.shape
    h = np.zeros(m, int)
    best = (0, None)
    for i in range(n):
        h = np.where(mask[i], h + 1, 0)
        stack = []
        for j in range(m + 1):
            cur = h[j] if j < m else 0
            start = j
            while stack and stack[-1][1] >= cur:
                start, hh = stack.pop()
                area = hh * (j - start)
                if area > best[0]:
                    best = (area, (i - hh + 1, start, i + 1, j))
            stack.append((start, cur))
    return best[1]


def zones(fr, free, seen, strip=0.9, min_len=0.5):
    nx, nz = free.shape
    out = []
    s = int(strip / GRID)
    # strips along each wall
    for wall in ("bottom", "top", "left", "right"):
        if wall in ("bottom", "top"):
            band = free[:, :s] if wall == "bottom" else free[:, nz - s:]
            ok = band.mean(1) > 0.95
            along = nx
        else:
            band = free[:s, :] if wall == "left" else free[nx - s:, :]
            ok = band.mean(0) > 0.95
            along = nz
        k = 0
        while k < along:
            if not ok[k]:
                k += 1
                continue
            e = k
            while e < along and ok[e]:
                e += 1
            if (e - k) * GRID >= min_len:
                if wall in ("bottom", "top"):
                    x, y, w, h = k * GRID, (0 if wall == "bottom" else fr.depth - strip), (e - k) * GRID, strip
                    sv = seen[k:e, :s].mean() if wall == "bottom" else seen[k:e, nz - s:].mean()
                else:
                    x, y, w, h = (0 if wall == "left" else fr.width - strip), k * GRID, strip, (e - k) * GRID
                    sv = seen[:s, k:e].mean() if wall == "left" else seen[nx - s:, k:e].mean()
                out.append({"wall": wall, "x": x, "y": y, "w": w, "h": h, "seen": float(sv)})
            k = e
    # open floor (away from the walls: the strips above cover those)
    inner = free.copy()
    inner[:s, :] = inner[nx - s:, :] = False
    inner[:, :s] = inner[:, nz - s:] = False
    for _ in range(2):
        r = largest_rect(inner)
        if r is None:
            break
        i0, j0, i1, j1 = r
        if (i1 - i0) * GRID < 0.6 or (j1 - j0) * GRID < 0.6:
            break
        out.append({"wall": None, "x": i0 * GRID, "y": j0 * GRID, "w": (i1 - i0) * GRID, "h": (j1 - j0) * GRID,
                    "seen": float(seen[i0:i1, j0:j1].mean())})
        inner[i0:i1, j0:j1] = False
    names = {"bottom": "dọc tường dưới", "top": "dọc tường trên", "left": "dọc tường trái", "right": "dọc tường phải", None: "giữa phòng"}
    for k, z in enumerate(out):
        z["id"] = f"z{k + 1}"
        z["label"] = names[z["wall"]] + ("" if z["seen"] >= 0.5 else " (dự đoán, chưa quét tới)")
    return out


# ---------------------------------------------------------------- main
def build(scan_id):
    t0 = time.time()
    folder, bundle, masks, floor_y, mapping = load(scan_id)
    foot = np.vstack([p for p in (W.boundary_points(kf, m, mapping, floor_y) for kf, m in masks) if len(p)])
    floor = np.vstack([p for p in (floor_points(kf, m, mapping, floor_y) for kf, m in masks) if len(p)])
    theta, R, xs, zs = room_frame(foot, floor)
    fr = Frame(theta, R, xs, zs)
    ops = openings(masks, mapping, fr, floor_y)
    objs = merge_objects(objects_from_points(furniture_points(masks, mapping, floor_y), fr), tv_spots(masks, mapping, fr))
    rug = rug_area(masks, mapping, floor_y, fr)
    all_objs = objs + ([rug] if rug else [])
    seen_room = fr.to_room(floor)
    free, occ, keep, seen, dist = free_space(fr, all_objs, ops, seen_room)
    zs_ = zones(fr, free, seen)
    sem = semantics(folder, masks, all_objs)
    cams = np.array([[k.pose[12], k.pose[14]] for k in bundle.keyframes])

    corners = fr.to_world(np.array([[0, 0], [fr.width, 0], [fr.width, fr.depth], [0, fr.depth]]))
    np.save(folder / "free_grid.npy", free)
    prof = {
        "scan_id": scan_id, "source": bundle.source, "floor_y": round(float(floor_y), 3),
        "size_m": [round(fr.width, 2), round(fr.depth, 2)], "area_m2": round(fr.width * fr.depth, 2),
        "frame": {"theta_deg": fr.theta, "origin_rotated": [fr.x0, fr.z0],
                  "note": "room (x, z) = R(theta) * world (x, z) - origin_rotated; R = [[cos, sin], [-sin, cos]]"},
        "floor_polygon_world": [[round(float(a), 3), round(float(b), 3)] for a, b in corners],
        "walls": [{"id": k, "predicted": v["predicted"], "evidence_points": v["support"]} for k, v in fr.walls.items()],
        "openings": ops,
        "objects": all_objs,
        "floor_seen_ratio": round(float(seen.mean()), 3),
        "free_grid": {"res": GRID, "file": "free_grid.npy", "free_ratio": round(float(free.mean()), 3)},
        "semantics": sem,
        # the room as step 7 (recommender) wants it: cm, origin bottom-left, y into the room
        "room": {"name": f"Phòng quét {scan_id} ({fr.width:.1f} × {fr.depth:.1f} m)",
                 "width_cm": round(fr.width * 100), "depth_cm": round(fr.depth * 100),
                 "zones": [{"id": z["id"], "x": round(z["x"] * 100), "y": round(z["y"] * 100),
                            "w": round(z["w"] * 100), "h": round(z["h"] * 100), "wall": z["wall"],
                            "label": z["label"]} for z in zs_]},
        "zones_seen": {z["id"]: round(z["seen"], 2) for z in zs_},
        "evidence": {"keyframes": len(masks), "wall_feet": int(len(foot)), "floor_points": int(len(floor))},
        "seconds": round(time.time() - t0, 1),
    }
    (folder / "room_profile.json").write_text(json.dumps(prof, indent=1, ensure_ascii=False), encoding="utf-8")
    draw(prof, fr, foot, floor, cams, free, seen, folder / "room_topdown.png")
    return prof


def draw(prof, fr, foot, floor, cams, free, seen, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.5, 8))
    W_, D_ = fr.width, fr.depth
    ax.imshow(np.where(seen, 1.0, 0.55).T, origin="lower", extent=(0, W_, 0, D_), cmap="Greys", vmin=0, vmax=1.6, alpha=0.5)
    fp, flr = fr.to_room(foot), fr.to_room(floor)
    ax.scatter(flr[::7, 0], flr[::7, 1], s=1, color="#9ec5f4", label="sàn đã thấy")
    ax.scatter(fp[:, 0], fp[:, 1], s=2, color="#5b4ea8", label="chân tường")
    for name, w in fr.walls.items():
        seg = {"left": ([0, 0], [0, D_]), "right": ([W_, W_], [0, D_]), "bottom": ([0, W_], [0, 0]), "top": ([0, W_], [D_, D_])}[name]
        ax.plot(*seg, color="#0b0b0b", lw=3, ls="--" if w["predicted"] else "-")
    for z in prof["room"]["zones"]:
        ax.add_patch(Rectangle((z["x"] / 100, z["y"] / 100), z["w"] / 100, z["h"] / 100, fill=False, ec="#2e8b57", lw=1.5, ls=":"))
        ax.text(z["x"] / 100 + 0.05, z["y"] / 100 + 0.05, z["id"], color="#2e8b57", fontsize=8)
    for o in prof["objects"]:
        ax.add_patch(Rectangle((o["x0"], o["z0"]), o["x1"] - o["x0"], o["z1"] - o["z0"],
                               color="#eb6834" if not o.get("walkable") else "#e9d8b8", alpha=0.55 if not o.get("walkable") else 0.4))
        ax.text((o["x0"] + o["x1"]) / 2, (o["z0"] + o["z1"]) / 2, f"{o['label']}\n{o['height']} m", fontsize=7, ha="center", va="center")
    for op in prof["openings"]:
        c = "#c0392b" if op["type"] == "door" else "#2980b9"
        if op["wall"] in ("bottom", "top"):
            y = 0 if op["wall"] == "bottom" else D_
            ax.plot([op["u0"], op["u1"]], [y, y], color=c, lw=6)
        else:
            x = 0 if op["wall"] == "left" else W_
            ax.plot([x, x], [op["u0"], op["u1"]], color=c, lw=6)
        ax.text(*(((op["u0"] + op["u1"]) / 2, y) if op["wall"] in ("bottom", "top") else (x, (op["u0"] + op["u1"]) / 2)),
                "cửa" if op["type"] == "door" else "cửa sổ", fontsize=8, color=c)
    cr = fr.to_room(cams)
    ax.plot(cr[:, 0], cr[:, 1], color="#eb6834", lw=1, label="camera")
    s = prof["semantics"]
    ax.set_title(f"Hồ sơ phòng {prof['scan_id']}: {W_:.2f} × {D_:.2f} m ({prof['area_m2']} m²) · "
                 f"{len(prof['objects'])} đồ vật · {len(prof['openings'])} cửa\n"
                 f"loại phòng: {s.get('room_type')} ({s.get('source')}) · sàn đã thấy {prof['floor_seen_ratio']:.0%} · "
                 f"nét đứt = tường dự đoán, ô xanh = vùng trống cho gợi ý", fontsize=8.5, loc="left")
    ax.set_xlim(-0.5, W_ + 0.5)
    ax.set_ylim(-0.5, D_ + 0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right", markerscale=4)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = build(sys.argv[1])
    print(json.dumps({k: p[k] for k in ("size_m", "area_m2", "walls", "openings", "objects", "floor_seen_ratio", "room", "zones_seen", "seconds")},
                     indent=1, ensure_ascii=False))
    print("semantics:", json.dumps(p["semantics"], ensure_ascii=False)[:600])

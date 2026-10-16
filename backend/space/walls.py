"""
Steps 9/10 (first version) - find the walls of a scanned room from its keyframes.

ARCore plane detection misses plain painted walls (2 real scans: 0 walls), so walls come from
semantic segmentation instead:
  1. SegFormer-B0 (ADE20K, 150 indoor/outdoor classes) labels every pixel of every keyframe
     (wall / floor / ceiling / door / window / bed / cabinet / sofa / ...). Keyframes are stored in
     camera-sensor orientation, so they are rotated upright first. Masks are cached as PNG.
  2. Pixel -> 3D ray. Which way the saved image maps onto the Unity camera is not documented
     (sensor rotation + the MirrorY used when saving), so 8 candidate mappings are tried and the
     ARCore's floor/table polygons are projected into the images and checked against the labels;
     (no swap, +x, +y) won on both A55 scans and is the default (re-checked every scan).
  3. Wall evidence:
     a. wall/floor boundary: in each image column, the lowest wall pixel that sits on floor
        pixels; its ray hits the floor plane at the foot of the wall (metres, no depth needed)
     b. ARCore depth (if the scan has it): every wall pixel with a confident depth -> 3D point
  4. Sequential RANSAC on the top-down (x, z) points -> wall line segments.

Input : data/scans/<id>/{scan.json, keyframes/*.jpg[, *_depth.bin, *_conf.bin]}
Output: data/scans/<id>/seg/*.png (label masks), walls.json, walls_topdown.png
Usage : python backend/space/walls.py <scan_id>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[2]
SCANS = ROOT / "data" / "scans"
sys.path.insert(0, str(ROOT / "backend" / "server"))

MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
WALL, FLOOR, CEILING = 0, 3, 5                         # ADE20K class ids
FURNITURE = {7: "bed", 10: "cabinet", 15: "table", 19: "chair", 23: "sofa", 24: "shelf",
             30: "armchair", 33: "desk", 35: "wardrobe", 36: "lamp", 44: "chest of drawers",
             28: "rug", 64: "coffee table", 14: "door", 8: "window"}
_model = None


# ------------------------------------------------------------------ segmentation
def segmenter():
    global _model
    if _model is None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        _model = (SegformerImageProcessor.from_pretrained(MODEL),
                  SegformerForSemanticSegmentation.from_pretrained(MODEL).eval())
    return _model


def upright(img):
    """Saved keyframes are in sensor orientation; rotating 90 deg counter-clockwise makes them
    upright for a phone held in portrait (checked on scan 20261015_200351: floor at the bottom)."""
    return img.rotate(90, expand=True)


def segment(path: Path, cache: Path) -> np.ndarray:
    """Label mask (H x W, uint8 ADE20K ids) of the UPRIGHT image. Cached as PNG."""
    from PIL import Image
    if cache.exists():
        return np.array(Image.open(cache))
    import torch
    proc, model = segmenter()
    im = upright(Image.open(path).convert("RGB"))
    with torch.no_grad():
        logits = model(**proc(images=im, return_tensors="pt")).logits
    mask = torch.nn.functional.interpolate(logits, size=im.size[::-1], mode="bilinear").argmax(1)[0]
    mask = mask.numpy().astype(np.uint8)
    cache.parent.mkdir(exist_ok=True)
    Image.fromarray(mask).save(cache)
    return mask


# ------------------------------------------------------------------ geometry
def pose_matrix(kf):
    return np.array(kf.pose, dtype=float).reshape(4, 4).T      # column-major -> row-major


# 8 ways the upright image could map onto the Unity camera (x right, y up, z forward):
# swap image axes or not, and a sign on each axis. Picked per scan by floor agreement.
MAPPINGS = [(swap, sx, sy) for swap in (False, True) for sx in (1, -1) for sy in (1, -1)]


def rays(kf, xs, ys, mapping, shape):
    """Upright-image pixels -> unit ray directions in world space."""
    H, W = shape                      # upright image size (= sensor height, sensor width)
    # intrinsics are given for the sensor image (width x height); in the upright image the
    # axes are swapped: fx_u = fy, fy_u = fx, cx_u = cy, cy_u = (W_s - 1) - cx
    fx_u, fy_u = kf.fy, kf.fx
    cx_u, cy_u = kf.cy, (kf.width - 1) - kf.cx
    swap, sx, sy = mapping
    a = (xs - cx_u) / fx_u
    b = -(ys - cy_u) / fy_u           # image y grows downward, camera y upward
    if swap:
        a, b = b, a
    cam = np.stack([sx * a, sy * b, np.ones_like(a)], axis=1)
    M = pose_matrix(kf)
    d = cam @ M[:3, :3].T
    return d / np.linalg.norm(d, axis=1, keepdims=True), M[:3, 3]


def hit_floor(origin, dirs, floor_y):
    """Intersections of rays with the plane y = floor_y (NaN where the ray does not go down)."""
    t = (floor_y - origin[1]) / dirs[:, 1]
    t[(dirs[:, 1] >= -1e-3) | (t > 12)] = np.nan          # upward rays / beyond 12 m
    return origin[[0, 2]] + dirs[:, [0, 2]] * t[:, None]


def floor_geometry(bundle):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import app as server
    ups = server.unique([p for p in bundle.planes if p.alignment == "HorizontalUp" and len(p.boundary) >= 9])
    floor_y = server.find_floor_y(ups, bundle.keyframes)
    floors = [p for p in ups if floor_y is not None and abs(p.center[1] - floor_y) < 0.10]
    geom = unary_union([Polygon(server.polygon_xz(p)).buffer(0) for p in floors]) if floors else None
    return floor_y, geom


# Verified on both Galaxy A55 scans by projecting ARCore's floor/table polygons into the images:
# (no swap, +x, +y) matches (table outline on the wooden table top). It is a property of the app +
# phone, not of the room, so it is fixed; calibrate() only re-checks it on every scan.
DEFAULT_MAPPING = (False, 1, 1)
FLOORISH = {FLOOR, 28}                                   # floor, rug
TOPS = {7, 10, 15, 24, 33, 35, 44, 64}                   # bed, cabinet, table, shelf, desk, wardrobe, chest, coffee table


def project(kf, P, mapping):
    """World points -> upright-image pixels (inverse of rays); points behind the camera dropped."""
    M = pose_matrix(kf)
    c = (P - M[:3, 3]) @ M[:3, :3]
    c = c[c[:, 2] > 0.1]
    if len(c) < 3:
        return None
    swap, sx, sy = mapping
    a, b = sx * c[:, 0] / c[:, 2], sy * c[:, 1] / c[:, 2]
    if swap:
        a, b = b, a
    return list(zip((a * kf.fy + kf.cy).tolist(), (-b * kf.fx + (kf.width - 1) - kf.cx).tolist()))


def calibrate(bundle, masks, floor_y):
    """Project every ARCore up-facing plane into every image; score = share of the covered pixels
    that the segmenter labels as expected (floor/rug for the floor, furniture tops for higher planes).
    Keeps DEFAULT_MAPPING unless another mapping is clearly (> 0.05) better."""
    from PIL import Image, ImageDraw
    planes = [p for p in bundle.planes if p.alignment == "HorizontalUp" and len(p.boundary) >= 9]
    scores = {}
    for m in MAPPINGS:
        hit = tot = 0
        for kf, mask in masks:
            for p in planes:
                poly = project(kf, np.array(p.boundary, float).reshape(-1, 3), m)
                if not poly:
                    continue
                im = Image.new("L", mask.shape[::-1])
                ImageDraw.Draw(im).polygon(poly, fill=1)
                r = np.array(im, bool)
                if r.sum() < 200:
                    continue
                want = FLOORISH if abs(p.center[1] - floor_y) < 0.10 else TOPS
                hit += int(np.isin(mask[r], list(want)).sum())
                tot += int(r.sum())
        scores[m] = hit / max(tot, 1)
    best = max(scores, key=scores.get)
    return (best if scores[best] > scores[DEFAULT_MAPPING] + 0.05 else DEFAULT_MAPPING), scores


def boundary_points(kf, mask, mapping, floor_y, step=4):
    """Feet of walls: per column, the lowest wall pixel with floor right below it."""
    H, W = mask.shape
    xs, ys = [], []
    for x in range(0, W, step):
        col = mask[:, x]
        for y in range(H - 4, 0, -1):          # from the bottom up
            if col[y] == WALL and (col[y + 1:y + 4] == FLOOR).all():
                xs.append(x)
                ys.append(y + 1)
                break
    if not xs:
        return np.zeros((0, 2))
    d, o = rays(kf, np.array(xs, float), np.array(ys, float), mapping, mask.shape)
    p = hit_floor(o, d, floor_y)
    return p[~np.isnan(p[:, 0])]


def depth_wall_points(kf, mask, mapping, folder, min_conf=128, step=2):
    """Wall pixels with a confident ARCore depth -> world (x, z)."""
    if not kf.depth_file or not (folder / "keyframes" / kf.depth_file).exists():
        return np.zeros((0, 2))
    dw, dh = kf.depth_width, kf.depth_height
    depth = np.frombuffer((folder / "keyframes" / kf.depth_file).read_bytes(), dtype="<u2").reshape(dh, dw) / 1000.0
    conf = None
    if kf.conf_file and (folder / "keyframes" / kf.conf_file).exists():
        conf = np.frombuffer((folder / "keyframes" / kf.conf_file).read_bytes(), dtype=np.uint8).reshape(dh, dw)
    # depth is in sensor orientation like the saved image -> rotate it the same way as the mask
    depth_u = np.rot90(depth)                  # 90 deg counter-clockwise, as upright()
    conf_u = np.rot90(conf) if conf is not None else None
    Hu, Wu = mask.shape
    sy, sx = Hu / depth_u.shape[0], Wu / depth_u.shape[1]
    yy, xx = np.mgrid[0:depth_u.shape[0]:step, 0:depth_u.shape[1]:step]
    lab = mask[(yy * sy).astype(int).clip(0, Hu - 1), (xx * sx).astype(int).clip(0, Wu - 1)]
    z = depth_u[yy, xx]
    ok = (lab == WALL) & (z > 0.2) & (z < 8)
    if conf_u is not None:
        ok &= conf_u[yy, xx] >= min_conf
    if not ok.any():
        return np.zeros((0, 2))
    d, o = rays(kf, xx[ok] * sx, yy[ok] * sy, mapping, mask.shape)
    # ARCore depth is distance along the optical axis -> scale the ray to that z
    M = pose_matrix(kf)
    fwd = M[:3, 2] / np.linalg.norm(M[:3, 2])
    t = z[ok] / np.clip(d @ fwd, 1e-3, None)
    pts = o + d * t[:, None]
    return pts[:, [0, 2]]


def ransac_lines(pts, thresh=0.06, min_inliers=25, max_lines=8, iters=400, seed=0):
    """Sequential RANSAC: repeatedly take the line with most inliers, remove them."""
    rng = np.random.default_rng(seed)
    lines, rest = [], pts.copy()
    while len(rest) >= min_inliers and len(lines) < max_lines:
        best = None
        for _ in range(iters):
            a, b = rest[rng.choice(len(rest), 2, replace=False)]
            v = b - a
            n = np.linalg.norm(v)
            if n < 0.2:
                continue
            nrm = np.array([-v[1], v[0]]) / n
            dist = np.abs((rest - a) @ nrm)
            inl = dist < thresh
            if best is None or inl.sum() > best[0].sum():
                best = (inl, a, v / n)
        if best is None or best[0].sum() < min_inliers:
            break
        inl, a, u = best
        P = rest[inl]
        c = P.mean(0)                          # refit direction by PCA on the inliers
        u = np.linalg.svd(P - c)[2][0]
        s = (P - c) @ u
        lo, hi = np.percentile(s, [3, 97])
        lines.append({"p0": (c + lo * u).round(3).tolist(), "p1": (c + hi * u).round(3).tolist(),
                      "length_m": round(float(hi - lo), 2), "points": int(inl.sum())})
        rest = rest[~inl]
    return lines


def furniture_summary(masks):
    tot = sum(m.size for _, m in masks)
    out = {}
    for cid, name in FURNITURE.items():
        n = sum(int((m == cid).sum()) for _, m in masks)
        if n / tot > 0.005:
            out[name] = round(n / tot, 3)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def analyse(scan_id: str) -> dict:
    import app as server
    folder = SCANS / scan_id
    bundle = server.ScanBundle.model_validate_json((folder / "scan.json").read_text(encoding="utf-8"))
    t0 = time.time()
    masks = [(kf, segment(folder / "keyframes" / kf.file, folder / "seg" / kf.file.replace(".jpg", ".png")))
             for kf in bundle.keyframes if (folder / "keyframes" / kf.file).exists()]
    t_seg = time.time() - t0
    floor_y, floor_geom = floor_geometry(bundle)
    if floor_y is None or floor_geom is None or floor_geom.is_empty:
        return {"scan_id": scan_id, "error": "no floor plane in the scan"}
    mapping, scores = calibrate(bundle, masks, floor_y)

    foot = [boundary_points(kf, m, mapping, floor_y) for kf, m in masks]
    dpts = [depth_wall_points(kf, m, mapping, folder) for kf, m in masks]
    foot = np.vstack([p for p in foot if len(p)]) if any(len(p) for p in foot) else np.zeros((0, 2))
    dpts = np.vstack([p for p in dpts if len(p)]) if any(len(p) for p in dpts) else np.zeros((0, 2))
    # depth gives many points per frame: thin them so they do not drown the boundary evidence
    if len(dpts) > 4000:
        dpts = dpts[np.random.default_rng(0).choice(len(dpts), 4000, replace=False)]
    allp = np.vstack([foot, dpts]) if len(dpts) else foot
    lines = ransac_lines(allp) if len(allp) else []

    res = {
        "scan_id": scan_id, "keyframes": len(masks), "segmentation_s": round(t_seg, 1),
        "floor_y": round(float(floor_y), 3),
        "mapping": {"swap_axes": mapping[0], "sign_x": mapping[1], "sign_y": mapping[2]},
        "mapping_label_agreement": round(scores[mapping], 3),
        "mapping_scores": {str(k): round(v, 3) for k, v in scores.items()},
        "wall_foot_points": int(len(foot)), "depth_wall_points": int(len(dpts)),
        "walls": lines,
        "pixel_share": {"wall": round(sum(int((m == WALL).sum()) for _, m in masks) / sum(m.size for _, m in masks), 3),
                        "floor": round(sum(int((m == FLOOR).sum()) for _, m in masks) / sum(m.size for _, m in masks), 3),
                        "ceiling": round(sum(int((m == CEILING).sum()) for _, m in masks) / sum(m.size for _, m in masks), 3)},
        "furniture_seen": furniture_summary(masks),
    }
    (folder / "walls.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    draw(bundle, floor_y, foot, dpts, lines, res, folder / "walls_topdown.png")
    return res


def draw(bundle, floor_y, foot, dpts, lines, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    import app as server
    SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
    FLOORC, WALLC, CAM, PTS, DEP = "#9ec5f4", "#0b0b0b", "#eb6834", "#4a3aa7", "#1baf7a"
    fig, ax = plt.subplots(figsize=(7, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for p in bundle.planes:
        if p.alignment == "HorizontalUp" and abs(p.center[1] - floor_y) < 0.10 and len(p.boundary) >= 9:
            ax.add_patch(MplPolygon(server.polygon_xz(p), closed=True, fc=FLOORC, ec=SURFACE, alpha=0.8))
    if len(dpts):
        ax.scatter(dpts[:, 0], dpts[:, 1], s=1, color=DEP, alpha=0.25, label="điểm tường từ độ sâu")
    if len(foot):
        ax.scatter(foot[:, 0], foot[:, 1], s=3, color=PTS, alpha=0.5, label="chân tường (tường/sàn trong ảnh)")
    for ln in lines:
        (x0, z0), (x1, z1) = ln["p0"], ln["p1"]
        ax.plot([x0, x1], [z0, z1], color=WALLC, lw=3, solid_capstyle="round")
        ax.text((x0 + x1) / 2, (z0 + z1) / 2, f" {ln['length_m']} m", fontsize=7.5, color=INK)
    xs = [k.pose[12] for k in bundle.keyframes]
    zs = [k.pose[14] for k in bundle.keyframes]
    ax.plot(xs, zs, color=CAM, lw=1, label="đường đi camera")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.grid(color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title(f"Tường từ segmentation · scan {res['scan_id']}\n{len(lines)} tường · "
                 f"{res['wall_foot_points']} điểm chân tường · {res['depth_wall_points']} điểm độ sâu · "
                 f"phép chiếu khớp nhãn {res['mapping_label_agreement']:.0%}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    out = analyse(sys.argv[1])
    print(json.dumps({k: v for k, v in out.items() if k != "mapping_scores"}, indent=1, ensure_ascii=False))
    print("mapping scores:", out.get("mapping_scores"))

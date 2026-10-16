"""
Step 2 - 3D model processing (Phase 3.2).

Checks every crawled IKEA .glb so the AR app can trust it:
  * real-world bounding box in metres (from the glTF POSITION accessor min/max
    transformed by the node hierarchy - no Draco decode needed for this)
  * compares the model's size with the dimensions IKEA lists on the product page
  * origin check: is the model's bottom at y=0 and centred on x/z?
  * mesh complexity (triangles, vertices, textures, file size) for mobile
  * optional preview PNG (front / side / top views) decoded with DracoPy,
    used to verify by eye which way the model faces (glTF convention: +Z = front)

Input : data/catalog.jsonl, data/models/<item_no>.glb
Output: data/model_meta.jsonl   one record per model
        data/model_report.csv   same, flat, for a spreadsheet
        data/previews/<item_no>.png  (with --previews)

Usage:
  python pipeline/process_models.py
  python pipeline/process_models.py --previews        # also render preview PNGs
  python pipeline/process_models.py --exclude         # move non-usable models to data/excluded/
"""

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np

TOLERANCE = 0.10          # 10 % size mismatch allowed vs. IKEA's stated dimensions
MINOR_TOLERANCE = 0.20    # 10-20 %: accepted, usually handles/knobs/cushions
USABLE_STATUS = ("ok", "ok_minor", "no_reference")
ORIGIN_TOL_M = 0.01       # 1 cm
MOBILE_MAX_TRIS = 100_000
MOBILE_MAX_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------- glTF reading
def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    off, gltf, binary = 12, None, b""
    while off < len(data):
        length, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + length]
        if ctype == 0x4E4F534A:      # JSON
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:    # BIN
            binary = chunk
        off += 8 + length
    return gltf, binary


def node_matrix(node) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T   # glTF is column-major
    t = np.array(node.get("translation", [0, 0, 0]), dtype=float)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=float)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


def mesh_instances(gltf):
    """Yield (mesh_index, world_matrix) for every mesh node in the default scene."""
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]

    def walk(idx, parent):
        node = gltf["nodes"][idx]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            yield node["mesh"], world
        for c in node.get("children", []):
            yield from walk(c, world)

    for root in scene.get("nodes", []):
        yield from walk(root, np.eye(4))


def transform(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return points @ m[:3, :3].T + m[:3, 3]


def accessor_bbox(gltf):
    """World AABB from POSITION accessor min/max (required by the glTF spec)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh_idx, world in mesh_instances(gltf):
        for prim in gltf["meshes"][mesh_idx]["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            mn, mx = np.array(acc["min"]), np.array(acc["max"])
            corners = np.array([[a, b, c] for a in (mn[0], mx[0])
                                for b in (mn[1], mx[1]) for c in (mn[2], mx[2])])
            w = transform(corners, world)
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def mesh_stats(gltf):
    tris = verts = 0
    for mesh_idx, _ in mesh_instances(gltf):
        for prim in gltf["meshes"][mesh_idx]["primitives"]:
            verts += gltf["accessors"][prim["attributes"]["POSITION"]]["count"]
            if "indices" in prim:
                tris += gltf["accessors"][prim["indices"]]["count"] // 3
            else:
                tris += gltf["accessors"][prim["attributes"]["POSITION"]]["count"] // 3
    return tris, verts


COMPONENT_DTYPE = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_SIZE = {"SCALAR": 1, "VEC3": 3}


def read_accessor(gltf, binary, idx):
    """Read a plain (non-Draco) SCALAR/VEC3 accessor from the BIN chunk."""
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPE[acc["componentType"]])
    n = TYPE_SIZE[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or dtype.itemsize * n
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (acc["count"] - 1) + dtype.itemsize * n, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(acc["count"], dtype.itemsize * n), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(acc["count"], n)


def decode_triangles(gltf, binary):
    """Primitives -> world-space triangles (N,3,3), decoding Draco when used. Previews only."""
    out = []
    for mesh_idx, world in mesh_instances(gltf):
        for prim in gltf["meshes"][mesh_idx]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue  # triangles only
            ext = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
            if ext:
                import DracoPy
                bv = gltf["bufferViews"][ext["bufferView"]]
                start = bv.get("byteOffset", 0)
                mesh = DracoPy.decode(binary[start:start + bv["byteLength"]])
                pts = np.asarray(mesh.points, dtype=float).reshape(-1, 3)
                faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
            else:
                pts = read_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(float)
                if "indices" in prim:
                    faces = read_accessor(gltf, binary, prim["indices"]).astype(np.int64).reshape(-1, 3)
                else:
                    faces = np.arange(len(pts)).reshape(-1, 3)
            out.append(transform(pts, world)[faces])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


# ------------------------------------------------------------- dimension check
def stated_dims(product):
    """IKEA's assembled dimensions (cm) -> horizontal sizes (sorted desc) + height."""
    d = product.get("dimensions_cm") or {}
    horiz = [d[k] for k in ("width", "depth", "length") if k in d]
    if "diameter" in d:
        horiz = [d["diameter"], d["diameter"]]
    return sorted(horiz, reverse=True), d.get("height")


def compare(product, size_cm):
    """Ratio model/stated for each dimension IKEA lists. Horizontal sizes are matched
    largest-to-largest because IKEA's width/length/depth naming is not consistent."""
    horiz, height = stated_dims(product)
    model_h = sorted([size_cm[0], size_cm[2]], reverse=True)
    ratios = {}
    if len(horiz) == 1:
        # only one horizontal size listed (e.g. "Width: 80" of an 120x80 table): compare to the closest axis
        ratios["horiz"] = min((m / horiz[0] for m in model_h), key=lambda r: abs(r - 1))
    for i, v in enumerate(horiz[:2] if len(horiz) > 1 else []):
        if v:
            ratios["horiz_major" if i == 0 else "horiz_minor"] = model_h[i] / v
    if height:
        ratios["height"] = size_cm[1] / height
    return ratios


def classify(ratios, origin_ok):
    if not ratios:
        return "no_reference"
    worst = max(abs(r - 1) for r in ratios.values())
    if all(abs(r - 100) < 10 or abs(r - 0.01) < 0.001 for r in ratios.values()):
        return "unit_error"
    if not origin_ok:
        return "origin_offset"
    if worst <= TOLERANCE:
        return "ok"
    if worst <= MINOR_TOLERANCE:
        # handles, knobs, cushions, skirting not counted in IKEA's numbers - harmless in AR
        return "ok_minor"
    # >20 %: either IKEA lists a partial size (sofa without chaise, lamp with its cord)
    # or the 3D model is another size variant. Excluded either way.
    return "size_mismatch"


# ------------------------------------------------------------------- previews
def render_preview(tris, title, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if len(tris) > 60_000:
        tris = tris[np.random.default_rng(0).choice(len(tris), 60_000, replace=False)]
    views = [  # (title, horizontal axis, vertical axis, depth axis towards viewer, flip horizontal)
        ("FRONT (camera at +Z)", 0, 1, 2, False),
        ("SIDE (camera at +X)", 2, 1, 0, True),
        ("TOP (+Z = down)", 0, 2, 1, False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (name, h, v, dpt, flip) in zip(axes, views):
        depth = tris[:, :, dpt].mean(1)
        order = np.argsort(depth)                    # painter's algorithm: far first
        xy = tris[order][:, :, [h, v]].copy()
        if flip:
            xy[:, :, 0] *= -1
        shade = (depth[order] - depth.min()) / (np.ptp(depth) + 1e-9)
        pc = PolyCollection(xy, array=shade, cmap="viridis", edgecolors="none")
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        if name.startswith("TOP"):
            ax.invert_yaxis()                        # +Z (front) at the bottom of the image
            ax.annotate("front (+Z)", xy=(0.5, 0.02), xycoords="axes fraction", ha="center", color="red")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def process(product, data_dir: Path, previews: bool, force: bool = False):
    path = data_dir / product["model_file"]
    gltf, binary = read_glb(path)
    lo, hi = accessor_bbox(gltf)
    size_m = hi - lo
    size_cm = size_m * 100
    center = (lo + hi) / 2
    origin_ok = abs(lo[1]) <= ORIGIN_TOL_M and abs(center[0]) <= ORIGIN_TOL_M and abs(center[2]) <= ORIGIN_TOL_M
    ratios = compare(product, size_cm)
    tris, verts = mesh_stats(gltf)
    size_bytes = path.stat().st_size

    rec = {
        "item_no": product["item_no"],
        "name": product.get("name"),
        "type_name": product.get("type_name"),
        "category": product.get("crawl_category"),
        "model_file": product["model_file"],
        # model size = authoritative footprint for AR placement (x = width, z = depth, y = height)
        "bbox_m": {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()},
        "dims_cm": {"w": round(size_cm[0], 1), "d": round(size_cm[2], 1), "h": round(size_cm[1], 1)},
        "stated_dims_cm": product.get("dimensions_cm"),
        "ratios": {k: round(v, 3) for k, v in ratios.items()},
        "max_error_pct": round(max((abs(r - 1) for r in ratios.values()), default=0) * 100, 1),
        "origin_bottom_y": round(float(lo[1]), 4),
        "origin_center_xz": [round(float(center[0]), 4), round(float(center[2]), 4)],
        "origin_ok": bool(origin_ok),
        "triangles": tris,
        "vertices": verts,
        "textures": len(gltf.get("textures", [])),
        "file_bytes": size_bytes,
        "draco": "KHR_draco_mesh_compression" in gltf.get("extensionsUsed", []),
        "mobile_ok": tris <= MOBILE_MAX_TRIS and size_bytes <= MOBILE_MAX_BYTES,
        "front_axis": "+Z (glTF convention, verify via preview)",
    }
    rec["status"] = classify(ratios, origin_ok)
    reasons = []
    if rec["status"] not in USABLE_STATUS:
        reasons.append(f"{rec['status']} (max error {rec['max_error_pct']}%)")
    if tris > MOBILE_MAX_TRIS:
        reasons.append(f"heavy: {tris:,} triangles > {MOBILE_MAX_TRIS:,}")
    if size_bytes > MOBILE_MAX_BYTES:
        reasons.append(f"heavy: {size_bytes / 1e6:.1f} MB > {MOBILE_MAX_BYTES / 1e6:.0f} MB")
    rec["ar_usable"] = not reasons
    rec["exclude_reason"] = "; ".join(reasons) or None

    preview = data_dir / "previews" / f"{rec['item_no']}.png"
    if previews and (force or not preview.exists()):
        prev_dir = data_dir / "previews"
        prev_dir.mkdir(exist_ok=True)
        tri_world = decode_triangles(gltf, binary)
        d = rec["dims_cm"]
        render_preview(tri_world, f"{rec['name']} {rec['type_name']} ({rec['item_no']})  "
                                  f"model {d['w']}x{d['d']}x{d['h']} cm  status={rec['status']}",
                       preview)
    if preview.exists():
        rec["preview_file"] = f"previews/{rec['item_no']}.png"
    return rec


def main():
    ap = argparse.ArgumentParser(description="Validate IKEA 3D models for AR")
    ap.add_argument("--data", default="data")
    ap.add_argument("--previews", action="store_true", help="render missing preview PNGs (needs DracoPy + matplotlib)")
    ap.add_argument("--force-previews", action="store_true", help="re-render existing previews too")
    ap.add_argument("--exclude", action="store_true",
                    help="move non-usable models to data/excluded/ and drop them from catalog + meta")
    args = ap.parse_args()
    data_dir = Path(args.data)

    products = [json.loads(l) for l in open(data_dir / "catalog.jsonl", encoding="utf-8")]
    records = []
    for p in products:
        try:
            records.append(process(p, data_dir, args.previews or args.force_previews, args.force_previews))
        except Exception as e:
            records.append({"item_no": p["item_no"], "name": p.get("name"), "status": "error", "error": str(e)})
        r = records[-1]
        print(f"{r['item_no']}  {r['status']:<14} {str(r.get('name')):<16} {str(r.get('type_name'))[:28]:<28} "
              f"model={r.get('dims_cm')} stated={r.get('stated_dims_cm')} err={r.get('max_error_pct')}%")

    from collections import Counter
    print("\nstatus:", dict(Counter(r["status"] for r in records)))
    print("mobile_ok:", sum(bool(r.get("mobile_ok")) for r in records), "/", len(records))
    print("ar_usable:", sum(bool(r.get("ar_usable")) for r in records), "/", len(records))

    if args.exclude:
        records = exclude(records, products, data_dir)

    with open(data_dir / "model_meta.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = ["item_no", "name", "type_name", "category", "status", "ar_usable", "exclude_reason", "max_error_pct",
            "origin_ok", "triangles", "file_bytes", "mobile_ok"]
    with open(data_dir / "model_report.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["model_w_cm", "model_d_cm", "model_h_cm", "stated"])
        for r in records:
            d = r.get("dims_cm") or {}
            w.writerow([r.get(c) for c in cols] + [d.get("w"), d.get("d"), d.get("h"), json.dumps(r.get("stated_dims_cm"))])


def exclude(records, products, data_dir: Path):
    """Move every non-usable product's files to data/excluded/<sub>/, record why in
    data/excluded.json (the crawler skips these item numbers), and rewrite catalog.jsonl
    without them. Files are moved, not deleted, so the step can be undone by hand."""
    out = data_dir / "excluded"
    reg_path = data_dir / "excluded.json"
    registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    bad = {r["item_no"]: r for r in records if not r.get("ar_usable")}
    for item, r in bad.items():
        for sub, ext in (("models", ".glb"), ("images", ".jpg"), ("products", ".json"), ("previews", ".png")):
            src = data_dir / sub / f"{item}{ext}"
            if src.exists():
                (out / sub).mkdir(parents=True, exist_ok=True)
                src.replace(out / sub / src.name)
        registry[item] = {"name": r.get("name"), "type_name": r.get("type_name"), "category": r.get("category"),
                          "reason": r.get("exclude_reason") or r.get("error")}
        print(f"  excluded {item} {r.get('name')} {r.get('type_name')}: {registry[item]['reason']}")
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(data_dir / "catalog.jsonl", "w", encoding="utf-8") as f:
        for p in products:
            if p["item_no"] not in bad:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nexcluded {len(bad)} -> {out}; catalog now {len(products) - len(bad)} products")
    return [r for r in records if r["item_no"] not in bad]


if __name__ == "__main__":
    main()

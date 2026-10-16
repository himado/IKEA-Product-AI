"""
DoHa backend (step 8 + first part of step 14): receives room scans from the phone.

The Unity app (ARRoomScanner.cs) posts one multipart request per scan:
  scan      JSON (ScanBundle below): AR planes with world-space boundaries, keyframe camera
            poses + intrinsics, device info. Unity world space: metres, y up, left-handed.
  kf_000..  JPEG keyframes (camera CPU image, sensor orientation)

Stored as data/scans/<scan_id>/{scan.json, keyframes/kf_XXX.jpg, topdown.png}.
topdown.png is a quick visual check (floor/walls/camera path) - real room geometry is step 9.

Run:   python backend/server/app.py            (http://0.0.0.0:8000)
Phone: adb reverse tcp:8000 tcp:8000  -> the app reaches the PC at http://127.0.0.1:8000 over USB
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
SCANS = ROOT / "data" / "scans"
sys.path.insert(0, str(ROOT / "backend"))


class Plane(BaseModel):
    id: str
    alignment: str                 # HorizontalUp / HorizontalDown / Vertical / NotAxisAligned
    center: List[float]            # world xyz (m)
    normal: List[float]
    size: List[float]              # plane extents (m)
    boundary: List[float]          # world-space polygon, flat [x0,y0,z0, x1,y1,z1, ...]
    epoch: int = 0                 # tracking epoch (ARCore re-initialised -> +1); coordinates may jump between epochs
    last_seen_t: float = 0.0       # scan time of the last valid boundary


class Keyframe(BaseModel):
    file: str
    t: float                       # seconds since scan start
    pose: List[float]              # camera localToWorld 4x4, column-major (Unity Matrix4x4 order)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int                     # image size the intrinsics refer to
    height: int
    depth_file: Optional[str] = None   # ARCore environment depth, raw little-endian uint16 millimetres
    conf_file: Optional[str] = None    # depth confidence, raw uint8 (0-255)
    depth_format: Optional[str] = None
    depth_width: int = 0
    depth_height: int = 0
    epoch: int = 0


class ScanBundle(BaseModel):
    app_version: str
    device: str
    started_at: str
    duration_s: float
    screen_orientation: str
    planes: List[Plane]
    keyframes: List[Keyframe]
    tracking_lost_count: int = 0
    depth_supported: bool = False
    source: str = "live"             # live | arcore_playback:<file> (step 8b virtual room)
    tracking_resets: List[float] = []  # scan times where tracking came back after a loss (epoch boundaries)
    image_note: Optional[str] = None


app = FastAPI(title="DoHa backend", version="0.1")


@app.get("/health")
def health():
    return {"ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/scans")
async def upload_scan(scan: str = Form(...), files: List[UploadFile] = File(default=[])):
    try:
        bundle = ScanBundle.model_validate_json(scan)
    except Exception as e:
        raise HTTPException(400, f"invalid scan JSON: {e}")
    scan_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    folder = SCANS / scan_id
    (folder / "keyframes").mkdir(parents=True)
    names = {n for k in bundle.keyframes for n in (k.file, k.depth_file, k.conf_file) if n}
    saved = 0
    for f in files:
        name = Path(f.filename or "").name
        if name not in names:          # only files the bundle refers to
            continue
        (folder / "keyframes" / name).write_bytes(await f.read())
        saved += 1
    (folder / "scan.json").write_text(bundle.model_dump_json(indent=1), encoding="utf-8")
    summary = summarize(bundle)
    summary.update({"scan_id": scan_id, "keyframes_saved": saved})
    try:
        draw_topdown(bundle, folder / "topdown.png", scan_id)
        summary["preview"] = f"/scans/{scan_id}/topdown.png"
    except Exception as e:             # preview is a convenience, never fail the upload for it
        summary["preview_error"] = str(e)
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[scan] {scan_id}: {summary}", flush=True)
    return summary


@app.get("/scans")
def list_scans():
    return sorted((p.name for p in SCANS.glob("*") if p.is_dir()), reverse=True) if SCANS.exists() else []


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str):
    f = SCANS / Path(scan_id).name / "summary.json"
    if not f.exists():
        raise HTTPException(404, "scan not found")
    return json.loads(f.read_text(encoding="utf-8"))


@app.get("/scans/{scan_id}/topdown.png")
def get_preview(scan_id: str):
    f = SCANS / Path(scan_id).name / "topdown.png"
    if not f.exists():
        raise HTTPException(404, "no preview")
    return FileResponse(f)


# --------------------------------------------------------------------------- step 15: picks for AR
@app.get("/scans/{scan_id}/recommendation")
def get_recommendation(scan_id: str):
    f = SCANS / Path(scan_id).name / "recommendation.json"
    if not f.exists():
        raise HTTPException(404, "no recommendation for this scan (run backend/recommend/recommend_scan.py)")
    return with_model_urls(json.loads(f.read_text(encoding="utf-8")))


@app.get("/sim/recommendation")
def sim_recommendation(playback: str):
    """Newest recommendation made for a scan of this ARCore playback file (step 8b virtual room).
    Its coordinates belong to that playback session, so the phone shows them while replaying it."""
    want = f"arcore_playback:{Path(playback).name}"
    best = None
    for d in SCANS.iterdir():
        rec, sj = d / "recommendation.json", d / "scan.json"
        if not (rec.exists() and sj.exists()):
            continue
        if json.loads(sj.read_text(encoding="utf-8")).get("source") == want:
            if best is None or rec.stat().st_mtime > best.stat().st_mtime:
                best = rec
    if best is None:
        raise HTTPException(404, f"no recommendation for {want}")
    return with_model_urls(json.loads(best.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- step 14: /recommend
class RecommendIn(BaseModel):
    request: str                     # Vietnamese request, e.g. "thêm ghế bành đọc sách và đèn cây, 8 triệu"
    rerank: bool = True              # Gemini picks the bundle (step 7 part 2); False = by score only
    playback: Optional[str] = None   # /sim/recommend only: the ARCore playback file the scan came from


_index = None   # ProductIndex (embedding model + vectors)
_warm = {"state": "not started"}


def warm_up():
    """Import sentence-transformers and load the embedding model in the background at startup:
    done on the first request it took 140-160 s on this PC (slow imports when RAM is short),
    while the Gemini calls of a request take only 20-35 s."""
    global _index
    t0 = time.time()
    _warm["state"] = "loading"
    try:
        for sub in ("backend/space", "backend/recommend"):
            if str(ROOT / sub) not in sys.path:
                sys.path.insert(0, str(ROOT / sub))
        from retrieve import ProductIndex
        idx = ProductIndex(ROOT / "data")
        idx.embed_query("warm up")
        import recommend_scan  # noqa: F401  (intent parser, Gemini client, recommender)
        _index = idx
        _warm.update(state="ready", seconds=round(time.time() - t0, 1))
    except Exception as e:                         # the request path loads it again if this failed
        _warm.update(state=f"failed: {e}")


@app.on_event("startup")
def start_warm_up():
    import threading
    threading.Thread(target=warm_up, daemon=True).start()


@app.get("/warm")
def warm_state():
    return _warm


def run_recommend(scan_id: str, body: RecommendIn) -> dict:
    """Steps 9-12 (room profile, built once per scan) -> 6 (intent) -> 7 (bundle) -> 13 (placement)
    -> AR coordinates. Takes 20-60 s (Gemini calls; the first profile of a scan ~1 min more)."""
    global _index
    for sub in ("backend/space", "backend/recommend"):
        if str(ROOT / sub) not in sys.path:
            sys.path.insert(0, str(ROOT / sub))
    if not (SCANS / scan_id / "room_profile.json").exists():
        import room_profile
        room_profile.build(scan_id)
    import recommend_scan
    from retrieve import ProductIndex
    t_wait = time.time()
    while _warm["state"] == "loading" and time.time() - t_wait < 300:   # warm-up still running: wait for it
        time.sleep(0.5)
    if _index is None:
        _index = ProductIndex(ROOT / "data")
    t0 = time.time()
    out = recommend_scan.run(scan_id, body.request, rerank=body.rerank, index=_index)
    out["seconds"] = round(time.time() - t0, 1)
    return with_model_urls(out)


@app.post("/scans/{scan_id}/recommend")
def post_recommend(scan_id: str, body: RecommendIn):
    scan_id = Path(scan_id).name
    if not (SCANS / scan_id / "scan.json").exists():
        raise HTTPException(404, "unknown scan")
    return run_recommend(scan_id, body)


@app.post("/sim/recommend")
def sim_recommend(body: RecommendIn):
    """Same, for the newest scan made from an ARCore playback file (step 8b): prefers a scan that
    already has its room profile."""
    want = f"arcore_playback:{Path(body.playback or '').name}"
    cands = []
    for d in SCANS.iterdir():
        sj = d / "scan.json"
        if sj.exists() and json.loads(sj.read_text(encoding="utf-8")).get("source") == want:
            cands.append(((d / "room_profile.json").exists(), sj.stat().st_mtime, d.name))
    if not cands:
        raise HTTPException(404, f"no scan for {want}")
    return run_recommend(max(cands)[2], body)


def with_model_urls(rec: dict) -> dict:
    for p in rec.get("picks", []):
        p["model_url"] = f"/models/{p['item_no']}.glb"
    return rec


@app.get("/models/{item_no}.glb")
def get_model(item_no: str):
    """IKEA model made loadable by glTFast (WebP textures -> PNG, see pipeline/export_unity_models.py)."""
    if not item_no.isdigit():
        raise HTTPException(400, "bad item number")
    src = ROOT / "data" / "models" / f"{item_no}.glb"
    dst = ROOT / "data" / "models_unity" / f"{item_no}.glb"
    if not src.exists():
        raise HTTPException(404, "no model")
    if not dst.exists():
        sys.path.insert(0, str(ROOT / "pipeline"))
        from export_unity_models import make_unity_compatible
        dst.parent.mkdir(parents=True, exist_ok=True)
        make_unity_compatible(src, dst)
    return FileResponse(dst, media_type="model/gltf-binary")


# --------------------------------------------------------------------------- quick geometry
def polygon_xz(plane: Plane):
    b = plane.boundary
    return [(b[i], b[i + 2]) for i in range(0, len(b) - 2, 3)]


WALL_MIN_TOP = 1.2   # a wall reaches at least 1.2 m above the floor; lower vertical faces are furniture


def top_above(p, floor_y) -> float:
    ys = p.boundary[1::3]
    return max(ys) - floor_y if ys else 0.0


def split_vertical(planes, floor_y):
    """-> (walls, low vertical faces). Without a floor every vertical plane counts as a wall."""
    walls, low = [], []
    for p in planes:
        (walls if floor_y is None or top_above(p, floor_y) >= WALL_MIN_TOP else low).append(p)
    return walls, low


def unique(planes):
    """ARCore can report the same plane twice (an old plane subsumed by a new one): drop repeats."""
    seen, out = set(), []
    for p in planes:
        key = (round(p.center[0], 2), round(p.center[1], 2), round(p.center[2], 2), round(p.size[0], 2), round(p.size[1], 2))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def plane_area(p) -> float:
    from shapely.geometry import Polygon
    return Polygon(polygon_xz(p)).buffer(0).area


def find_floor_y(ups, keyframes):
    """Floor = the largest upward plane lying 1.0-2.0 m below the average camera height (a phone is
    held at ~1.2-1.6 m). The first version took the LOWEST plane, which on the first real scan was a
    2.5 m patch 25 cm below the floor the user walked on. Falls back to the largest plane overall."""
    if not ups:
        return None
    cam_y = sum(k.pose[13] for k in keyframes) / len(keyframes) if keyframes else None
    cands = [p for p in ups if cam_y is not None and 1.0 <= cam_y - p.center[1] <= 2.0] or ups
    return max(cands, key=plane_area).center[1]


def summarize(b: ScanBundle) -> dict:
    """First look at the scan: is there a floor, how big, how many walls. (Step 9 does this properly.)"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    ups = unique([p for p in b.planes if p.alignment == "HorizontalUp" and len(p.boundary) >= 9])
    verticals = unique([p for p in b.planes if p.alignment == "Vertical" and len(p.boundary) >= 9])
    floor_y = find_floor_y(ups, b.keyframes)
    walls, low_vertical = split_vertical(verticals, floor_y)
    floor = [p for p in ups if floor_y is not None and abs(p.center[1] - floor_y) < 0.10]
    area = 0.0
    bbox = None
    if floor:
        geom = unary_union([Polygon(polygon_xz(p)).buffer(0) for p in floor])
        area = geom.area
        x0, z0, x1, z1 = geom.bounds
        bbox = [round(x1 - x0, 2), round(z1 - z0, 2)]
    path = 0.0
    prev = None
    for k in b.keyframes:
        pos = (k.pose[12], k.pose[13], k.pose[14])
        if prev:
            path += sum((a - c) ** 2 for a, c in zip(pos, prev)) ** 0.5
        prev = pos
    lower = [p for p in ups if floor_y is not None and p.center[1] < floor_y - 0.10]
    return {
        "planes": len(b.planes), "floor_planes": len(floor), "wall_planes": len(walls),
        "low_vertical": len(low_vertical),   # vertical faces lower than 1.2 m: furniture sides, skirting
        "wall_lengths_m": [round(Polygon(polygon_xz(w)).length / 2, 2) for w in walls],
        "lower_horizontal": len(lower),   # below the floor: a step down, another level, or AR drift
        "other_horizontal": len(ups) - len(floor) - len(lower),
        "floor_area_m2": round(area, 2), "floor_bbox_m": bbox,
        "floor_y": round(floor_y, 3) if floor_y is not None else None,
        "camera_height_above_floor_m": round(sum(k.pose[13] for k in b.keyframes) / len(b.keyframes) - floor_y, 2)
        if b.keyframes and floor_y is not None else None,
        "keyframes": len(b.keyframes), "camera_path_m": round(path, 2),
        "duration_s": round(b.duration_s, 1), "device": b.device,
        "tracking_lost_count": b.tracking_lost_count,
        "depth_supported": b.depth_supported,
        "source": b.source,
        "keyframes_with_depth": sum(1 for k in b.keyframes if k.depth_file),
        "depth_size": next(([k.depth_width, k.depth_height] for k in b.keyframes if k.depth_file), None),
    }


def draw_topdown(b: ScanBundle, path: Path, scan_id: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
    FLOOR, OTHER, WALL, CAM, LOWV = "#9ec5f4", "#e1e0d9", "#0b0b0b", "#eb6834", "#898781"
    s = summarize(b)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    floor_y = s["floor_y"]
    for p in b.planes:
        pts = polygon_xz(p)
        if len(pts) < 2:
            continue
        if p.alignment == "Vertical":       # seen from above: a line along the plane
            xs, zs = zip(*pts)
            is_wall = floor_y is None or top_above(p, floor_y) >= WALL_MIN_TOP
            ax.plot(xs, zs, color=WALL if is_wall else LOWV, lw=2.5 if is_wall else 1.8, solid_capstyle="round")
        elif p.alignment == "HorizontalUp":
            is_floor = floor_y is not None and abs(p.center[1] - floor_y) < 0.10
            ax.add_patch(MplPolygon(pts, closed=True, fc=FLOOR if is_floor else OTHER,
                                    ec=SURFACE, lw=1, alpha=0.9 if is_floor else 0.8))
    if b.keyframes:
        xs = [k.pose[12] for k in b.keyframes]
        zs = [k.pose[14] for k in b.keyframes]
        ax.plot(xs, zs, color=CAM, lw=1.2)
        ax.scatter(xs, zs, s=14, color=CAM, zorder=3)
        for k, x, z in zip(b.keyframes, xs, zs):       # viewing direction = camera forward (column 2)
            fx, fz = k.pose[8], k.pose[10]
            n = (fx * fx + fz * fz) ** 0.5 or 1
            ax.annotate("", xy=(x + 0.25 * fx / n, z + 0.25 * fz / n), xytext=(x, z),
                        arrowprops=dict(arrowstyle="->", color=CAM, lw=0.8))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.grid(color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("x (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("z (m)", color=MUTED, fontsize=8)
    ax.set_title(f"Scan {scan_id}\nsàn {s['floor_area_m2']} m² (bbox {s['floor_bbox_m']} m) · "
                 f"{s['wall_planes']} tường + {s['low_vertical']} mặt đứng thấp\n{s['keyframes']} ảnh · "
                 f"camera cao {s['camera_height_above_floor_m']} m, đi {s['camera_path_m']} m",
                 fontsize=9, color=INK, loc="left")
    ax.text(0.0, -0.14, "Xanh = sàn · xám = mặt ngang khác (bàn, giường…) · đen = tường (cao ≥ 1,2 m)\n"
                        "xám đậm = mặt đứng thấp (thành tủ, giường) · cam = camera (mũi tên = hướng nhìn)",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def reprocess(scan_id: str):
    """Recompute summary + preview of a stored scan (after the analysis code changed)."""
    folder = SCANS / scan_id
    bundle = ScanBundle.model_validate_json((folder / "scan.json").read_text(encoding="utf-8"))
    summary = summarize(bundle)
    summary.update({"scan_id": scan_id, "keyframes_saved": len(list((folder / "keyframes").glob("*.jpg")))})
    draw_topdown(bundle, folder / "topdown.png", scan_id)
    summary["preview"] = f"/scans/{scan_id}/topdown.png"
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--reprocess":
        print(json.dumps(reprocess(sys.argv[2]), indent=1))
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

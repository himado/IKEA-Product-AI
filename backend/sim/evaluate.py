"""
Step 8b - score an ARCore playback scan of the virtual room against its ground truth.

The ground truth (ground_truth.json of the recording) is used ONLY here, never by the scan pipeline.

1. Align coordinates: ARCore playback starts its own world frame (and may re-initialise after a
   tracking loss = new 'epoch'), so for every epoch the playback camera path (report timeline) is
   fitted to the recording's camera path (frames.jsonl, the frame the virtual room was placed in)
   with a yaw + translation (gravity is shared). One time offset for the whole playback.
2. Planes of the uploaded scan -> recording world (their epoch's fit) -> room frame.
3. Metrics: floor height error, floor coverage of the walkable area, furniture-top heights,
   walls found, planes outside the room, camera path error.

Usage: python backend/sim/evaluate.py <scan_id>      (scan uploaded by a playback)
Output: data/sim/eval/<scan_id>.json + .png
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data" / "sim"
SCANS = ROOT / "data" / "scans"


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def fit_yaw_translation(P, Q):
    """Q ~ R_y P + t (R about the vertical axis). -> (R 2x2 on xz, t 3, residuals)"""
    pc, qc = P.mean(0), Q.mean(0)
    X, Y = P - pc, Q - qc
    U, _, Vt = np.linalg.svd(X[:, [0, 2]].T @ Y[:, [0, 2]])
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    def apply(p):
        p = np.atleast_2d(p) - pc
        xz = p[:, [0, 2]] @ R.T
        return np.stack([xz[:, 0], p[:, 1], xz[:, 1]], 1) + qc
    res = np.linalg.norm(apply(P) - Q, axis=1)
    return apply, res


def align(timeline, rec_t, rec_p, min_samples=6):
    """Per-epoch transform playback world -> recording world, shared time offset."""
    rows = [r for r in timeline if r["session"] == "SessionTracking"]
    epochs = sorted({r.get("epoch", 0) for r in rows})
    best = None
    for off in np.arange(-3.0, 3.001, 0.05):
        total, n, fits = 0.0, 0, {}
        for e in epochs:
            er = [r for r in rows if r.get("epoch", 0) == e]
            if len(er) < min_samples:
                continue
            P = np.array([r["cam"] for r in er])
            Q = np.array([rec_p[np.argmin(np.abs(rec_t - (r["t"] + off)))] for r in er])
            fn, res = fit_yaw_translation(P, Q)
            fits[e] = (fn, res)
            total += res.sum()
            n += len(res)
        if n and (best is None or total / n < best[0]):
            best = (total / n, off, fits)
    return best


class Room:
    def __init__(self, gt):
        self.spec = gt["spec"]
        self.o = np.array(gt["room_origin"])
        y = np.radians(gt["room_yaw_deg"])
        self.R = np.array([[np.cos(y), np.sin(y)], [-np.sin(y), np.cos(y)]])   # Unity yaw on (x, z)

    def to_room(self, w):
        w = np.atleast_2d(w) - self.o
        xz = w[:, [0, 2]] @ self.R          # R^T applied to row vectors
        return np.stack([xz[:, 0], w[:, 1], xz[:, 1]], 1)


def footprints(gt):
    from shapely.geometry import box
    out = []
    for b, it in zip(gt["built"], gt["spec"]["furniture"]):
        w, h, d = b["size_m"] if b.get("size_m") else (it["w"], it["h"], it["d"])
        turned = abs(((it["yaw"] + 180) % 180) - 90) < 1
        fw, fd = (d, w) if turned else (w, d)
        out.append((it, h, box(it["x"] - fw / 2, it["z"] - fd / 2, it["x"] + fw / 2, it["z"] + fd / 2)))
    return out


def evaluate(scan_id):
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union

    scan = json.loads((SCANS / scan_id / "scan.json").read_text(encoding="utf-8"))
    src = scan.get("source", "")
    assert src.startswith("arcore_playback:"), f"not a playback scan: {src}"
    play_file = src.split(":", 1)[1]
    rec_id = "_".join(play_file.split("_")[:3])                    # rec_YYYYMMDD_HHMMSS
    rec = SIM / "recordings" / rec_id
    gt = json.loads((rec / "ground_truth.json").read_text(encoding="utf-8"))
    reports = sorted((SIM / "reports").glob(Path(play_file).stem + "_*.json"))
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    rows = load_jsonl(rec / "frames.jsonl")
    log = [r for r in rows if "ts_ns" in r]
    # virtual capture starts ~0.5 s before the ARCore dataset (SimRecorder); playback time 0 = dataset start
    t0 = next((r["t"] for r in rows if r.get("event") == "arcore_recording_started"), 0.0)
    rec_t = np.array([r["t"] for r in log]) - t0
    rec_p = np.array([r["pose"][12:15] for r in log])
    err, off, fits = align(report["timeline"], rec_t, rec_p)
    room = Room(gt)
    W, D = gt["spec"]["width"], gt["spec"]["depth"]
    floor_y = gt["floor_y"]

    planes = []
    for p in scan["planes"]:
        e = p.get("epoch", 0)
        if e not in fits or len(p["boundary"]) < 9:
            planes.append({"id": p["id"], "alignment": p["alignment"], "epoch": e, "aligned": False})
            continue
        pts = room.to_room(fits[e][0](np.array(p["boundary"]).reshape(-1, 3)))
        h = pts[:, 1] - (floor_y - room.o[1])           # height above the virtual floor
        planes.append({"id": p["id"], "alignment": p["alignment"], "epoch": e, "aligned": True,
                       "xz": pts[:, [0, 2]], "h": float(np.median(h))})

    walk = box(0, 0, W, D)
    feet = footprints(gt)
    for it, _, fp in feet:
        if it["category"] != "rug":
            walk = walk.difference(fp)
    floors = [p for p in planes if p.get("aligned") and p["alignment"] == "HorizontalUp" and abs(p["h"]) < 0.12]
    floor_union = unary_union([Polygon(p["xz"]).buffer(0) for p in floors]) if floors else Polygon()
    floor_area = sum(Polygon(p["xz"]).area for p in floors)
    res = {
        "scan_id": scan_id, "recording": rec_id, "playback_file": play_file,
        "time_offset_s": round(float(off), 2),
        "camera_path_error_m": {str(e): round(float(r.mean()), 3) for e, (_, r) in fits.items()},
        "epochs": sorted({p["epoch"] for p in planes}) if planes else [],
        "tracking_resets_s": scan.get("tracking_resets", []),
        "planes_total": len(planes),
        "planes_aligned": sum(1 for p in planes if p.get("aligned")),
        "floor": {
            "planes": len(floors),
            "height_error_cm": round(100 * float(np.average([p["h"] for p in floors],
                                     weights=[Polygon(p["xz"]).area for p in floors])), 1) if floors else None,
            "area_m2": round(floor_area, 2),
            "walkable_area_m2": round(walk.area, 2),
            "coverage_of_walkable": round(floor_union.intersection(walk).area / walk.area, 3),
            "outside_room_m2": round(floor_union.difference(box(-0.05, -0.05, W + 0.05, D + 0.05)).area, 2),
        },
        "tops": [], "walls": [], "outside_room": [],
    }
    for p in planes:
        if not p.get("aligned") or p in floors:
            continue
        c = Polygon(p["xz"]).buffer(0).centroid if len(p["xz"]) >= 3 else Point(p["xz"].mean(0))
        inside = -0.1 <= c.x <= W + 0.1 and -0.1 <= c.y <= D + 0.1
        if not inside:
            res["outside_room"].append({"id": p["id"], "alignment": p["alignment"], "x": round(c.x, 2), "z": round(c.y, 2), "h": round(p["h"], 2)})
            continue
        if p["alignment"] == "HorizontalUp":
            hit = [(it, h) for it, h, fp in feet if fp.buffer(0.1).contains(c)]
            res["tops"].append({"x": round(c.x, 2), "z": round(c.y, 2), "h_m": round(p["h"], 3),
                                "furniture": hit[0][0]["name"] if hit else None,
                                "furniture_h_m": round(hit[0][1], 3) if hit else None})
        elif p["alignment"] == "Vertical":
            dists = {"south": c.y, "north": D - c.y, "west": c.x, "east": W - c.x}
            wall = min(dists, key=dists.get)
            res["walls"].append({"wall": wall, "distance_m": round(dists[wall], 2), "h_m": round(p["h"], 2)})
    draw(res, gt, feet, planes, floors, fits, report, rec_p, room, SIM / "eval" / f"{scan_id}.png")
    out = SIM / "eval" / f"{scan_id}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def draw(res, gt, feet, planes, floors, fits, report, rec_p, room, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    W, D = gt["spec"]["width"], gt["spec"]["depth"]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.add_patch(Rectangle((0, 0), W, D, fill=False, lw=3, color="#0b0b0b"))
    for it, _, fp in feet:
        x, y = fp.exterior.xy
        ax.fill(x, y, color="#d9d7cf" if it["category"] != "rug" else "#efe9df", zorder=1)
        ax.text(fp.centroid.x, fp.centroid.y, it["name"].split()[0], fontsize=7, ha="center", color="#555")
    for p in planes:
        if not p.get("aligned"):
            continue
        xz = np.vstack([p["xz"], p["xz"][:1]])
        if p in floors:
            ax.fill(xz[:, 0], xz[:, 1], color="#4a90d9", alpha=0.35, zorder=2)
        elif p["alignment"] == "Vertical":
            ax.plot(xz[:, 0], xz[:, 1], color="#c0392b", lw=2.5, zorder=3)
        else:
            ax.fill(xz[:, 0], xz[:, 1], color="#eb6834", alpha=0.45, zorder=2)
    gt_path = room.to_room(rec_p)
    ax.plot(gt_path[:, 0], gt_path[:, 2], color="#2e8b57", lw=1.2, label="đường đi thật (ghi)")
    for e, (fn, _) in fits.items():
        cams = np.array([r["cam"] for r in report["timeline"] if r.get("epoch", 0) == e and r["session"] == "SessionTracking"])
        rp = room.to_room(fn(cams))
        ax.plot(rp[:, 0], rp[:, 2], "--", lw=1, label=f"ARCore phát lại, epoch {e}")
    f = res["floor"]
    ax.set_title(f"{res['scan_id']} - phòng ảo {W}×{D} m\nsàn: {f['planes']} mặt, lệch cao {f['height_error_cm']} cm, "
                 f"phủ {f['coverage_of_walkable']:.0%} chỗ đi được · mặt khác {len(res['tops'])} · tường {len(res['walls'])}",
                 fontsize=9, loc="left")
    ax.set_aspect("equal")
    ax.set_xlim(-1, W + 1)
    ax.set_ylim(-1, D + 1)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("x (m, đông)")
    ax.set_ylabel("z (m, bắc)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=1, ensure_ascii=False))

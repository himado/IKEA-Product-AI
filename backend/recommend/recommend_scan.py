"""
Steps 13-14 (first version) - recommend products for a SCANNED room and place them in AR coordinates.

room_profile.json (steps 9-12) gives the free zones in room cm; step 6 parses the request; step 7
picks the bundle and puts every product in a zone. Here the placements are turned back into the AR
session's world coordinates (position on the floor + yaw so the model's front, +Z, faces the room),
which is what the phone needs to show them (step 15).

Usage: python backend/recommend/recommend_scan.py <scan_id> "<yêu cầu>" [--rerank]
Output: data/scans/<id>/recommendation.json + recommend_topdown.png
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import ROOT, ProductIndex, recommend  # noqa: E402

sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "intent"))
from schemas import RoomProfile  # noqa: E402
from currency import fmt_vnd  # noqa: E402
from placement import FRONT_CLEAR, GROUP, place_group  # noqa: E402

FRONT = {"up": (0.0, 1.0), "down": (0.0, -1.0), "right": (1.0, 0.0), "left": (-1.0, 0.0)}   # room (x, z)


def world_pose(prof, pl):
    """Placement (room cm) -> world position (x, floor_y, z) + yaw (deg, Unity: +Z of the model)."""
    th = np.radians(prof["frame"]["theta_deg"])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    ox, oz = prof["frame"]["origin_rotated"]
    c = np.array([(pl.x + pl.w / 2) / 100 + ox, (pl.y + pl.h / 2) / 100 + oz]) @ R
    d = np.array(FRONT[pl.front]) @ R
    yaw = float(np.degrees(np.arctan2(d[0], d[1])))
    return [round(float(c[0]), 3), prof["floor_y"], round(float(c[1]), 3)], round(yaw, 1)


def run(scan_id, request, rerank=False, index=None):
    """index: a loaded ProductIndex (the server keeps one; loading the embedding model takes ~10 s)."""
    from parse import parse_intent
    folder = ROOT / "data" / "scans" / scan_id
    prof = json.loads((folder / "room_profile.json").read_text(encoding="utf-8"))
    room = RoomProfile(**prof["room"])
    res = parse_intent(request)
    if res.intent is None:
        return {"clarification": res.clarification_question}
    rec = recommend(index or ProductIndex(ROOT / "data"), res.intent, room, scenario=scan_id, rerank=rerank)

    # step 13: re-place the reading group together (placement.py); others keep step 7's zone
    notes = []
    members = {p.category: p for p in rec.picks if p.category in GROUP and p.placement}
    if len(members) >= 2:
        free = np.load(folder / prof["free_grid"]["file"])
        layout, notes = place_group(prof, free, {c: (p.size_cm[0], p.size_cm[1]) for c, p in members.items()})
        if layout:
            for c, p in members.items():
                r = layout[c]
                x, y, w, h = r["x0"] * 100, r["z0"] * 100, (r["x1"] - r["x0"]) * 100, (r["z1"] - r["z0"]) * 100
                fc = FRONT_CLEAR[c] * 100
                dx, dy = FRONT[r["front"]]
                cx0 = x + min(0, dx) * fc
                cy0 = y + min(0, dy) * fc
                p.placement = p.placement.model_copy(update=dict(
                    zone_id=f"góc đọc ({r['wall']})", x=x, y=y, w=w, h=h, front=r["front"],
                    clear_x=cx0, clear_y=cy0, clear_w=w + abs(dx) * fc, clear_h=h + abs(dy) * fc))
        else:
            notes.append("không tìm được chỗ đặt cả nhóm cùng nhau, giữ cách đặt của bước 7")

    picks = []
    for p in rec.picks:
        item = {"category": p.category, "item_no": p.item_no, "name": p.name, "type_name": p.type_name,
                "price_vnd": p.price_vnd, "qty": p.qty, "size_cm": p.size_cm, "model_file": p.model_file,
                "reason_vi": p.reason_vi, "fit_vi": p.fit_vi}
        if p.placement:
            pos, yaw = world_pose(prof, p.placement)
            item.update(zone=p.placement.zone_id, room_cm=[p.placement.x, p.placement.y, p.placement.w, p.placement.h],
                        front=p.placement.front, world_position=pos, world_yaw_deg=yaw)
        picks.append(item)
    out = {"scan_id": scan_id, "request": request, "intent_source": res.source,
           "room_size_m": prof["size_m"], "semantics": prof["semantics"].get("room_type"),
           "budget_vnd": rec.budget_vnd, "total_vnd": rec.total_vnd, "ranker": rec.ranker,
           "comment_vi": rec.comment_vi, "unmet": rec.unmet, "picks": picks,
           "rule_violations": rec.rule_violations, "placement_notes": notes}
    (folder / "recommendation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    draw(prof, rec, folder / "recommend_topdown.png")
    return out


def draw(prof, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    Wd, Dp = prof["size_m"]
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.add_patch(Rectangle((0, 0), Wd, Dp, fill=False, lw=3, color="#0b0b0b"))
    for o in prof["objects"]:
        ax.add_patch(Rectangle((o["x0"], o["z0"]), o["x1"] - o["x0"], o["z1"] - o["z0"],
                               color="#e9d8b8" if o.get("walkable") else "#c9c6bd", alpha=0.6))
        ax.text((o["x0"] + o["x1"]) / 2, (o["z0"] + o["z1"]) / 2, "có sẵn: " + o["label"], fontsize=7, ha="center", color="#555")
    for op in prof["openings"]:
        c = "#c0392b" if op["type"] == "door" else "#2980b9"
        if op["wall"] in ("bottom", "top"):
            y = 0 if op["wall"] == "bottom" else Dp
            ax.plot([op["u0"], op["u1"]], [y, y], color=c, lw=6)
        else:
            x = 0 if op["wall"] == "left" else Wd
            ax.plot([x, x], [op["u0"], op["u1"]], color=c, lw=6)
    for z in prof["room"]["zones"]:
        ax.add_patch(Rectangle((z["x"] / 100, z["y"] / 100), z["w"] / 100, z["h"] / 100, fill=False, ec="#2e8b57", ls=":", lw=1))
    for p in rec.picks:
        pl = p.placement
        if not pl:
            continue
        ax.add_patch(Rectangle((pl.clear_x / 100, pl.clear_y / 100), pl.clear_w / 100, pl.clear_h / 100, color="#eb6834", alpha=0.15))
        ax.add_patch(Rectangle((pl.x / 100, pl.y / 100), pl.w / 100, pl.h / 100, color="#eb6834", alpha=0.8))
        cx, cy = (pl.x + pl.w / 2) / 100, (pl.y + pl.h / 2) / 100
        dx, dy = FRONT[pl.front]
        ax.arrow(cx, cy, dx * 0.25, dy * 0.25, head_width=0.06, color="#0b0b0b")
        ax.text(cx, cy - 0.12, f"{p.name}\n{fmt_vnd(p.price_vnd)}", fontsize=7, ha="center", va="top")
    ax.set_title(f"Gợi ý cho phòng quét {prof['scan_id']} ({Wd} × {Dp} m)\n"
                 f"tổng {fmt_vnd(rec.total_vnd)} / ngân sách {fmt_vnd(rec.budget_vnd)} · {rec.ranker}", fontsize=9, loc="left")
    ax.set_xlim(-0.3, Wd + 0.3)
    ax.set_ylim(-0.3, Dp + 0.3)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    req = args[1] if len(args) > 1 else ("Phòng khách này muốn thêm một ghế bành để đọc sách, một đèn cây "
                                         "và một bàn trà nhỏ, tổng khoảng 8 triệu")
    print(json.dumps(run(args[0], req, rerank="--rerank" in sys.argv), indent=1, ensure_ascii=False))

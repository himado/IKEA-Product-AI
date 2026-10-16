"""
Step 16 - evaluation table (PLAN.md section 9) from the data the project already has.

  P1 room size error, big-furniture recall  : room_profile.json of a playback scan vs the virtual
                                              room's ground truth (used ONLY here)
  P2 slot extraction                         : backend/intent/eval_intent.py results (30 cases)
  P4 budget / size violations                : every recommendation made (7 scenarios + scanned room)
  P4 'reasonable' picks                      : user ratings (human)
  P5 FPS, model load, drift                  : step 5 phone test + step 8b trajectory fit
  latency                                    : /recommend timings

Usage: python backend/eval/run_eval.py [scan_id]     -> eval/report.md + eval/report.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "sim"))
import evaluate as EV  # noqa: E402

SCAN = sys.argv[1] if len(sys.argv) > 1 else "20261016_174640_f741c4"
FAMILY = {"sofa": "seat", "armchair": "seat", "tv_bench": "low", "sideboard": "low", "bookcase_shelf": "shelf",
          "rug": "rug", "floor_lamp": "lamp", "coffee_side_table": "low"}


def room_metrics(scan_id):
    """Profile (playback frame) -> recording world (trajectory fit) -> ground-truth room frame."""
    scan = json.loads((ROOT / "data" / "scans" / scan_id / "scan.json").read_text(encoding="utf-8"))
    prof = json.loads((ROOT / "data" / "scans" / scan_id / "room_profile.json").read_text(encoding="utf-8"))
    play = scan["source"].split(":", 1)[1]
    rec_id = "_".join(play.split("_")[:3])
    rec = ROOT / "data" / "sim" / "recordings" / rec_id
    gt = json.loads((rec / "ground_truth.json").read_text(encoding="utf-8"))
    report = json.loads(sorted((ROOT / "data" / "sim" / "reports").glob(Path(play).stem + "_*.json"))[0].read_text(encoding="utf-8"))
    rows = EV.load_jsonl(rec / "frames.jsonl")
    t0 = next((r["t"] for r in rows if r.get("event") == "arcore_recording_started"), 0.0)
    log = [r for r in rows if "ts_ns" in r]
    err, _, fits = EV.align(report["timeline"], np.array([r["t"] for r in log]) - t0, np.array([r["pose"][12:15] for r in log]))
    fn = fits[0][0]
    room = EV.Room(gt)

    th = np.radians(prof["frame"]["theta_deg"])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    ox, oz = prof["frame"]["origin_rotated"]

    def prof_to_gt(rx, rz):
        w = (np.array([rx + ox, rz + oz]) @ R)
        p = fn(np.array([[w[0], prof["floor_y"], w[1]]]))
        return room.to_room(p)[0][[0, 2]]

    Wp, Dp = prof["size_m"]
    c = np.array([prof_to_gt(*xz) for xz in [(0, 0), (Wp, 0), (Wp, Dp), (0, Dp)]])
    ax_x = c[1] - c[0]
    along_gt_x = abs(ax_x[0]) > abs(ax_x[1])          # which GT axis the profile's x axis lies on
    Wg, Dg = gt["spec"]["width"], gt["spec"]["depth"]
    pairs = [(Wp, Wg if along_gt_x else Dg), (Dp, Dg if along_gt_x else Wg)]
    size_err = [abs(a - b) / b for a, b in pairs]

    gt_items = [(it, EV.footprints(gt)[k][2]) for k, it in enumerate(gt["spec"]["furniture"])]
    found = []
    for o in prof["objects"]:
        cx, cz = prof_to_gt((o["x0"] + o["x1"]) / 2, (o["z0"] + o["z1"]) / 2)
        found.append((o, np.array([cx, cz])))
    matches = []
    for it, fp in gt_items:
        fam = FAMILY.get(it["category"], it["category"])
        best = None
        for o, c2 in found:
            if FAMILY.get(o["category"], o["category"]) != fam:
                continue
            d = float(np.hypot(c2[0] - fp.centroid.x, c2[1] - fp.centroid.y))
            if best is None or d < best[1]:
                best = (o["label"], d)
        wrong = None
        if not (best and best[1] < 1.0):          # right place, wrong class?
            near = [(o["label"], float(np.hypot(c2[0] - fp.centroid.x, c2[1] - fp.centroid.y))) for o, c2 in found]
            near = [n for n in near if n[1] < 1.0]   # same tolerance as the class match (ARCore drift ~0.3-0.6 m)
            wrong = min(near, key=lambda n: n[1]) if near else None
        matches.append({"gt": it["name"], "category": it["category"], "found": best[0] if best and best[1] < 1.0 else None,
                        "centre_error_m": round(best[1], 2) if best and best[1] < 1.0 else (round(wrong[1], 2) if wrong else None),
                        "wrong_class": wrong[0] if wrong else None})
    big = [m for m in matches if m["category"] in ("sofa", "tv_bench", "bookcase_shelf", "bed", "wardrobe", "dining_table")]
    located = sum(1 for m in big if m["found"] or m.get("wrong_class"))
    return {"scan": scan_id, "profile_size_m": [Wp, Dp], "truth_size_m": [p[1] for p in pairs],
            "size_error_pct": [round(100 * e, 1) for e in size_err], "trajectory_error_m": round(float(err), 3),
            "matches": matches, "big_recall": f"{sum(1 for m in big if m['found'])}/{len(big)} đúng loại, {located}/{len(big)} đúng chỗ",
            "all_recall": f"{sum(1 for m in matches if m['found'])}/{len(matches)}"}


def violations():
    checked, bad = [], []
    files = list((ROOT / "eval" / "results_nl_rerank").glob("S*.json")) + list((ROOT / "data" / "scans").glob("*/recommendation.json"))
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        over = r.get("budget_vnd") is not None and (r.get("total_vnd") or 0) > r["budget_vnd"]
        rules = r.get("rule_violations") or []
        checked.append(f.parent.name if f.name == "recommendation.json" else f.stem)
        if over or rules:
            bad.append({"file": str(f.relative_to(ROOT)), "over_budget": over, "rules": rules})
    return {"bundles": len(checked), "violations": bad}


def main():
    rm = room_metrics(SCAN)
    vi = violations()
    rows = [
        ("P1", "Sai số kích thước phòng", "< 10%", f"{rm['size_error_pct'][0]}% và {rm['size_error_pct'][1]}% "
         f"({rm['profile_size_m'][0]} × {rm['profile_size_m'][1]} m so với {rm['truth_size_m'][0]} × {rm['truth_size_m'][1]} m)",
         "phòng ảo, 1 bản quét"),
        ("P1", "Recall đồ vật lớn", "≥ 80%", f"{rm['big_recall']} (mọi đồ: {rm['all_recall']})", "đèn cây không nhận ra"),
        ("P2", "Trích xuất slot (danh mục, ngân sách…)", "≥ 90%", "30/30 câu (Gemini); 23/30 khi chỉ dùng luật dự phòng",
         "30 câu thay vì 50"),
        ("P4", "Vi phạm ngân sách / tỷ lệ giá", "0%", f"{len(vi['violations'])}/{vi['bundles']} bộ gợi ý", "7 kịch bản + phòng quét"),
        ("P4", "Gợi ý hợp lý (người chấm)", "≥ 70%", "lần chấm v1: 4/7 Tốt, 3/7 Tạm; đã sửa phân bổ ngân sách, chưa chấm lại",
         "cần bạn chấm lại"),
        ("P5", "Độ trôi vị trí", "< 5 cm", f"phòng ảo phát lại: {round(rm['trajectory_error_m'] * 100)} cm",
         "ảnh dựng khác ảnh camera thật; chưa đo trên phòng thật"),
        ("P5", "FPS / thời gian load model", "≥ 30 fps, < 3 s", "30 FPS (5 model); load 171–198 ms", "Galaxy A55, bước 5"),
        ("–", "Thời gian một lượt gợi ý (API)", "–", "≈ 9 s (trước là 90–113 s)", "server nạp sẵn mô hình khi khởi động (12–75 s); bỏ qua model Gemini đang hết hạn mức"),
    ]
    md = ["# Đánh giá DoHa (bước 16)", "", f"Bản quét dùng để đo: `{SCAN}` (phòng ảo, ARCore phát lại). Đáp án phòng ảo chỉ dùng ở file này.", "",
          "| Giai đoạn | Chỉ số | Mục tiêu | Kết quả | Ghi chú |", "|---|---|---|---|---|"]
    for r in rows:
        ok = ""
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    md += ["", "## Đồ vật: đáp án ↔ máy tìm được", "", "| Đồ thật | Máy nhận là | Lệch tâm |", "|---|---|---|"]
    for m in rm["matches"]:
        seen = m["found"] or (f"{m['wrong_class']} (đúng chỗ, sai loại)" if m.get("wrong_class") else "— (không thấy)")
        md.append(f"| {m['gt']} | {seen} | {m['centre_error_m'] if m['centre_error_m'] is not None else '—'} m |")
    (ROOT / "eval" / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "eval" / "report.json").write_text(json.dumps({"room": rm, "violations": vi}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

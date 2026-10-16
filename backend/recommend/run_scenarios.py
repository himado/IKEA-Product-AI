"""
Run every scenario in eval/scenarios.json through retrieve.recommend(), check the hard
guarantees independently, and draw a top-down plan per scenario.

Checks (must hold for every pick, otherwise the scenario FAILS):
  budget    total price (incl. qty) <= budget
  category  pick is in the requested category
  fit       product + clearance lies inside its zone, inside the room
  overlap   no two floor items share a zone, and no product footprint overlaps another
            product or its usage clearance (rugs excepted - they lie underneath)

Output: eval/results/<id>.json, eval/results/<id>.png, eval/results/summary.md
Usage:  python backend/recommend/run_scenarios.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from retrieve import ROOT, SHARES_ZONE, ProductIndex, recommend  # noqa: E402
from currency import fmt_vnd  # noqa: E402
from schemas import Intent, RoomProfile  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PRODUCT, CLEARANCE = "#2a78d6", "#cde2fb"
EPS = 0.01


def check(rec, room, intent):
    errors = []
    if intent.budget is not None and rec.total_price > intent.budget:
        errors.append(f"budget: ¥{rec.total_price:,} > ¥{intent.budget:,}")
    if sum(p.price * p.qty for p in rec.picks) != rec.total_price:
        errors.append("total_price does not match picks")
    wanted = {it.category for it in intent.items}
    zones = {z.id: z for z in room.zones}
    used = []
    for p in rec.picks:
        if p.category not in wanted:
            errors.append(f"category: {p.name} is {p.category}")
        pl = p.placement
        if pl is None:
            continue
        z = zones[pl.zone_id]
        for (x, y, w, h, what) in [(pl.x, pl.y, pl.w, pl.h, "product"),
                                   (pl.clear_x, pl.clear_y, pl.clear_w, pl.clear_h, "clearance")]:
            if x < z.x - EPS or y < z.y - EPS or x + w > z.x + z.w + EPS or y + h > z.y + z.h + EPS:
                errors.append(f"fit: {p.name} {what} outside zone {z.id}")
            if x < -EPS or y < -EPS or x + w > room.width_cm + EPS or y + h > room.depth_cm + EPS:
                errors.append(f"fit: {p.name} {what} outside room")
        if p.category not in SHARES_ZONE:
            if pl.zone_id in used:
                errors.append(f"overlap: zone {pl.zone_id} used twice")
            used.append(pl.zone_id)
    solid = [p for p in rec.picks if p.placement and p.category not in SHARES_ZONE]
    for i, a in enumerate(solid):
        for b in solid[i + 1:]:
            A, B = a.placement, b.placement
            if (intersects((A.x, A.y, A.w, A.h), (B.clear_x, B.clear_y, B.clear_w, B.clear_h))
                    or intersects((B.x, B.y, B.w, B.h), (A.clear_x, A.clear_y, A.clear_w, A.clear_h))):
                errors.append(f"overlap: {a.name} and {b.name}")
    return errors


def intersects(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return x1 < x2 + w2 - EPS and x2 < x1 + w1 - EPS and y1 < y2 + h2 - EPS and y2 < y1 + h1 - EPS


def draw(rec, room, path):
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * room.depth_cm / room.width_cm + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.add_patch(Rectangle((0, 0), room.width_cm, room.depth_cm, fill=False, ec=INK, lw=2.5))
    for z in room.zones:
        ax.add_patch(Rectangle((z.x, z.y), z.w, z.h, fill=False, ec=MUTED, lw=1))
        ax.text(z.x + 3, z.y + 3, f"{z.id} {z.label}", fontsize=7, color=MUTED, va="bottom")
    arrows = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    for p in sorted(rec.picks, key=lambda p: p.category not in SHARES_ZONE):  # rugs first (underneath)
        pl = p.placement
        if pl is None:
            continue
        if p.category in SHARES_ZONE:   # rug: outline only, so it is not mistaken for clearance
            ax.add_patch(Rectangle((pl.x, pl.y), pl.w, pl.h, fill=False, ec=PRODUCT, lw=2))
        else:
            ax.add_patch(Rectangle((pl.clear_x, pl.clear_y), pl.clear_w, pl.clear_h, fc=CLEARANCE, ec="none"))
            ax.add_patch(Rectangle((pl.x, pl.y), pl.w, pl.h, fc=PRODUCT, ec=SURFACE, lw=2))
        cx, cy = pl.x + pl.w / 2, pl.y + pl.h / 2
        dx, dy = arrows[pl.front]
        if p.category not in SHARES_ZONE:
            ax.annotate("", xy=(cx + dx * pl.w * 0.35 if dx else cx, cy + dy * pl.h * 0.35 if dy else cy),
                        xytext=(cx, cy), arrowprops=dict(arrowstyle="->", color=SURFACE, lw=1.5))
        ax.text(cx, pl.y + pl.h + 2 if p.category not in SHARES_ZONE else pl.y + 4,
                f"{p.name}\n{p.size_cm[0]:.0f}×{p.size_cm[1]:.0f} cm", fontsize=7.5, color=INK,
                ha="center", va="bottom")
    ax.set_xlim(-10, room.width_cm + 10)
    ax.set_ylim(-10, room.depth_cm + 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    off = [p for p in rec.picks if p.placement is None]
    sub = ("  ·  không cần chỗ sàn: " + ", ".join(f"{p.name}×{p.qty}" for p in off)) if off else ""
    budget = f" / {fmt_vnd(rec.budget_vnd)}" if rec.budget_vnd else ""
    by = "" if rec.ranker == "score" else f"  ·  chọn bởi {rec.ranker}"
    ax.set_title(f"{room.name}\n\"{rec.request}\"\nTổng {fmt_vnd(rec.total_vnd)}{budget}{sub}{by}",
                 fontsize=8.5, color=INK, loc="left")
    if rec.unmet:
        ax.text(room.width_cm / 2, room.depth_cm / 2, "\n".join(rec.unmet), ha="center", va="center",
                fontsize=8, color=INK2, wrap=True)
    ax.text(0, -8, "Xanh đậm = sản phẩm (mũi tên = hướng mặt trước) · xanh nhạt = khoảng trống để sử dụng · "
                   "viền xanh = thảm · khung xám = vùng trống của phòng", fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def fmt_set(rec):
    return "<br>".join(f"{p.name} {p.type_name} {fmt_vnd(p.price_vnd)}" + (f" ×{p.qty}" if p.qty > 1 else "")
                       for p in rec.picks) or "–"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", action="store_true",
                    help="end-to-end: parse each scenario's natural-language request with step 6 (Gemini) "
                         "instead of using the hand-written intent")
    ap.add_argument("--rerank", action="store_true",
                    help="step 7 stage 2: let Gemini pick the set from the shortlist; also writes compare.md "
                         "(score-ranked set vs. reranked set)")
    args = ap.parse_args()
    data = json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))
    name = "results" + ("_nl" if args.nl else "") + ("_rerank" if args.rerank else "")
    out = ROOT / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    if args.nl:
        sys.path.insert(0, str(ROOT / "backend" / "intent"))
        from parse import parse_intent
    index = ProductIndex(ROOT / "data")
    lines = ["| Kịch bản | Ngân sách | Tổng | Chọn | Chưa đáp ứng | Kiểm tra |", "|---|---|---|---|---|---|"]
    compare = ["# So sánh: bộ chọn theo điểm (retrieval) và bộ do Gemini rerank", ""]
    n_fail = n_changed = 0
    for sc in data["scenarios"]:
        room, intent = RoomProfile(**sc["room"]), Intent(**sc["intent"])
        if args.nl:
            hand = intent
            parsed = parse_intent(hand.request)
            if parsed.intent is None:
                print(f"\n=== {sc['id']}  asks: {parsed.clarification_question}")
                continue
            intent = parsed.intent
            same = (sorted(i.category for i in intent.items) == sorted(i.category for i in hand.items)
                    and intent.budget == hand.budget
                    and sorted(i.qty for i in intent.items) == sorted(i.qty for i in hand.items))
            print(f"\n[{parsed.source}] intent {'== hand-written' if same else 'DIFFERS'}: "
                  f"{[(i.category, i.qty, i.query) for i in intent.items]} budget {intent.budget}")
        base = recommend(index, intent, room, sc["id"])
        rec = recommend(index, intent, room, sc["id"], rerank=True) if args.rerank else base
        errs = check(rec, room, intent)
        n_fail += bool(errs)
        (out / f"{sc['id']}.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        draw(rec, room, out / f"{sc['id']}.png")

        print(f"\n=== {sc['id']}  {intent.request}   [ranker: {rec.ranker}]")
        for p in rec.picks:
            where = p.placement.zone_id if p.placement else "-"
            print(f"  {p.category:<18} {p.name:<16} {p.type_name[:30]:<30} {fmt_vnd(p.price_vnd):>13} x{p.qty}  "
                  f"{p.size_cm[0]:.0f}x{p.size_cm[1]:.0f}  zone {where}  score {p.score}")
            if rec.ranker != "score":
                print(f"      -> {p.reason_vi}")
        for u in rec.unmet:
            print(f"  UNMET: {u}")
        if rec.rerank_note:
            print(f"  NOTE: {rec.rerank_note}")
        print(f"  total {fmt_vnd(rec.total_vnd)} / budget {fmt_vnd(rec.budget_vnd) if rec.budget_vnd else '-'}   "
              f"checks: {'PASS' if not errs else errs}")
        lines.append(f"| {sc['id']} | {fmt_vnd(rec.budget_vnd) if rec.budget_vnd else '–'} | "
                     f"{fmt_vnd(rec.total_vnd)} | {fmt_set(rec)} | {'; '.join(rec.unmet) or '–'} | "
                     f"{'✅' if not errs else '❌ ' + '; '.join(errs)} |")

        if args.rerank:
            changed = [p.item_no for p in rec.picks] != [p.item_no for p in base.picks]
            n_changed += changed
            compare += [f"## {sc['id']}", f"> {intent.request}", "",
                        f"Ngân sách: {fmt_vnd(rec.budget_vnd) if rec.budget_vnd else '–'} · "
                        f"Gemini: **{'đổi bộ' if changed else 'giữ nguyên bộ'}** · người chọn: `{rec.ranker}`"
                        + (f" · ghi chú: {rec.rerank_note}" if rec.rerank_note else ""), "",
                        "| Món | Mục tiêu | Theo điểm (retrieval) | Gemini rerank | Lý do (Gemini) |", "|---|---|---|---|---|"]
            for bp, rp in zip(base.picks, rec.picks):
                mark = "" if bp.item_no == rp.item_no else " 🔁"
                tgt = fmt_vnd(rp.target_vnd) if rp.target_vnd else "–"
                compare.append(f"| {rp.category} | {tgt} | {bp.name} {bp.type_name} · {bp.size_cm[0]:.0f}×{bp.size_cm[1]:.0f} cm · "
                               f"{fmt_vnd(bp.price_vnd)} | **{rp.name} {rp.type_name}**{mark} · "
                               f"{rp.size_cm[0]:.0f}×{rp.size_cm[1]:.0f} cm · {fmt_vnd(rp.price_vnd)} | "
                               f"{rp.reason_vi if rec.ranker != 'score' else '–'} |")
            use = lambda r: f" ({r.total_vnd / r.budget_vnd:.0%} ngân sách)" if r.budget_vnd else ""
            compare += ["", f"Tổng: theo điểm {fmt_vnd(base.total_vnd)}{use(base)} → Gemini {fmt_vnd(rec.total_vnd)}{use(rec)}"
                        + (f"  ·  Nhận xét chung: {rec.comment_vi}" if rec.comment_vi else ""),
                        f"Luật tỉ lệ giá: {'; '.join(rec.price_rules) or '–'} → "
                        f"{'✅ đạt' if not rec.rule_violations else '❌ ' + '; '.join(rec.rule_violations)}"
                        f" · lệch phân bổ: theo điểm {base.allocation_deviation} → Gemini {rec.allocation_deviation}",
                        f"Kiểm tra cứng: {'✅ đạt' if not errs else '❌ ' + '; '.join(errs)}", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.rerank:
        (out / "compare.md").write_text("\n".join(compare) + "\n", encoding="utf-8")
        print(f"\nGemini changed the set in {n_changed} scenario(s) -> {out / 'compare.md'}")
    print(f"\n{len(data['scenarios']) - n_fail}/{len(data['scenarios'])} scenarios pass all checks -> {out}")


if __name__ == "__main__":
    main()

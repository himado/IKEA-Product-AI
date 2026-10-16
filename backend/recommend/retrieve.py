"""
Step 7 (retrieval part, no LLM): intent + room -> a product bundle that is guaranteed to
(1) match each requested category, (2) fit a free zone of the room including the space
needed to use it, and (3) stay within the total budget.

Pipeline per request:
  1. candidates  - ProductIndex.search(): hard filters (category, unit price) + semantic rank
  2. dedupe      - keep the best colour variant per product line
  3. fit         - keep candidates whose footprint + clearance fits at least one zone
  4. bundle      - exhaustive search over the top-N per item: maximise total similarity
                   subject to total price <= budget and one zone per floor item
                   (rugs may lie under other furniture)
The LLM rerank (PLAN §5 4.3) will later reorder/explain this shortlist; it never has to
check prices or sizes because nothing here can violate them.
"""

import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "backend"))

from currency import fmt_vnd, jpy_to_vnd  # noqa: E402
from allocation import allocation_deviation, ratio_violations, rules_for, targets_jpy  # noqa: E402
from schemas import Intent, IntentItem, Pick, Placement, Recommendation, RoomProfile, Zone  # noqa: E402
from search import ProductIndex  # noqa: E402

# Space needed to use a product, (front_cm, each_side_cm), and whether it normally stands
# against a wall. Hand-written defaults until step 3 (enrichment) provides per-product values.
USAGE = {
    "sofa": (60, 0, True), "armchair": (50, 0, False), "ottoman_pouffe": (0, 0, False),
    "dining_chair": (40, 0, False), "desk_chair": (50, 0, False),
    "coffee_side_table": (30, 0, False), "dining_table": (70, 70, False), "desk": (70, 0, True),
    "bed": (50, 50, True), "bedside_table": (0, 0, True), "wardrobe": (80, 0, True),
    "chest_of_drawers": (60, 0, True), "bookcase_shelf": (40, 0, True),
    "sideboard_cabinet": (50, 0, True), "tv_bench": (30, 0, True),
    "floor_lamp": (0, 0, False), "table_lamp": (0, 0, False), "rug": (0, 0, False),
}
# IKEA category pages also list accessories and neighbours (a swivel chair under "sofas", a
# floor protector under "desk chairs", a laptop tray under "desks"). Until step 3 gives a
# per-product category_norm, keep a candidate only if its type_name matches its category.
TYPE_RULES = {  # category: (type_name must contain one of, must not contain any of)
    "sofa": (["sofa"], ["cover", "bed"]),
    "armchair": (["armchair", "easy chair", "rocking", "wing chair", "lounge"], ["cover"]),
    "ottoman_pouffe": (["footstool", "pouffe", "ottoman", "stool"], ["cover"]),
    "dining_chair": (["chair"], ["cover", "pad", "cushion"]),
    "desk_chair": (["chair"], ["protector", "pad", "cushion"]),
    "coffee_side_table": (["coffee table", "side table", "nest of tables", "tray table", "storage table"], ["bedside"]),
    "dining_table": (["table"], ["and 4 chairs", "and 6 chairs", "and 2 chairs"]),
    "desk": (["desk"], ["tray", "stand", "organiser"]),
    "bed": (["bed"], ["bedside", "tray", "storage box", "underbed"]),
    "bedside_table": (["bedside", "nightstand", "chest", "table"], []),
    "wardrobe": (["wardrobe"], []),
    "chest_of_drawers": (["chest", "drawer"], []),
    "bookcase_shelf": (["bookcase", "shelving", "shelf"], []),
    "sideboard_cabinet": (["sideboard", "cabinet"], []),
    "tv_bench": (["tv"], []),
    "floor_lamp": (["floor"], []),
    "table_lamp": (["lamp"], ["floor"]),
    "rug": (["rug"], []),
}
NO_ZONE = {"table_lamp"}      # stands on furniture, not on the floor
SHARES_ZONE = {"rug"}         # lies under other furniture
TOP_N = 6                     # most similar candidates per item that enter the bundle search
ALLOC_PENALTY = 0.03              # relevance points lost per item when the spend split is completely off
EXTRA_NEAR_TARGET = 2         # + the remaining fitting candidates whose price is closest to the item's
                              #   budget target (allocation.py), so the shortlist offers options at the
                              #   right price level (user feedback 2026-10-15)
FACING = {"top": "down", "bottom": "up", "left": "right", "right": "left"}


def needs_zone(item: IntentItem) -> bool:
    return item.needs_zone if item.needs_zone is not None else item.category not in NO_ZONE


def place(zone: Zone, cat: str, w: float, d: float) -> Optional[Placement]:
    """Put a w x d (cm) product into a zone with its usage clearance. Against-wall zones:
    back to the wall, facing into the room. Free zones: centred, either orientation.
    Returns None if it does not fit."""
    front, side, _ = USAGE.get(cat, (0, 0, False))
    need_along, need_depth = w + 2 * side, d + front

    if zone.wall:
        along, depth = (zone.w, zone.h) if zone.wall in ("top", "bottom") else (zone.h, zone.w)
        if need_along > along or need_depth > depth:
            return None
        o = (along - w) / 2                      # centred along the wall
        co = (along - need_along) / 2
        if zone.wall == "top":
            r = (zone.x + o, zone.y + zone.h - d, w, d)
            c = (zone.x + co, zone.y + zone.h - need_depth, need_along, need_depth)
        elif zone.wall == "bottom":
            r = (zone.x + o, zone.y, w, d)
            c = (zone.x + co, zone.y, need_along, need_depth)
        elif zone.wall == "left":
            r = (zone.x, zone.y + o, d, w)
            c = (zone.x, zone.y + co, need_depth, need_along)
        else:
            r = (zone.x + zone.w - d, zone.y + o, d, w)
            c = (zone.x + zone.w - need_depth, zone.y + co, need_depth, need_along)
        front_dir = FACING[zone.wall]
    else:
        # free-standing: clearance on the front only (all sides for dining tables via `side`)
        fits_a = need_along <= zone.w and need_depth <= zone.h   # facing down/up the room
        fits_b = need_along <= zone.h and need_depth <= zone.w   # rotated 90 degrees
        if not (fits_a or fits_b):
            return None
        if fits_a:
            cw, ch, pw, ph, front_dir = need_along, need_depth, w, d, "down"
            r = (zone.x + (zone.w - pw) / 2, zone.y + (zone.h - ch) / 2 + front, pw, ph)
        else:
            cw, ch, pw, ph, front_dir = need_depth, need_along, d, w, "left"
            r = (zone.x + (zone.w - cw) / 2 + front, zone.y + (zone.h - ph) / 2, pw, ph)
        c = (zone.x + (zone.w - cw) / 2, zone.y + (zone.h - ch) / 2, cw, ch)
    return Placement(zone_id=zone.id, x=r[0], y=r[1], w=r[2], h=r[3],
                     clear_x=c[0], clear_y=c[1], clear_w=c[2], clear_h=c[3], front=front_dir)


def candidates(index: ProductIndex, item: IntentItem, room: RoomProfile, budget: Optional[int]):
    """Ranked, deduplicated candidates for one item, each with the zones it fits."""
    cap = item.max_price
    if budget is not None:
        cap = min(cap, budget // item.qty) if cap else budget // item.qty
    raw = index.search(item.query, k=60, category={item.category}, max_price=cap, max_w=item.max_width_cm)
    include, exclude = TYPE_RULES.get(item.category, ([], []))
    out, seen = [], set()
    for r in raw:
        if item.min_width_cm and (r["w_cm"] or 0) < item.min_width_cm:
            continue          # e.g. a single bed for "giường đôi"
        t = (r["type_name"] or "").lower()
        if (include and not any(k in t for k in include)) or any(k in t for k in exclude):
            continue
        line = (r["name"], r["type_name"])
        if line in seen:          # colour variant of something already listed
            continue
        seen.add(line)
        fits = {}
        if needs_zone(item):
            for z in room.zones:
                p = place(z, item.category, r["w_cm"], r["d_cm"])
                if p:
                    fits[z.id] = p
            if not fits:
                continue
        out.append((r, fits))
    return out


def shortlist(cands, target_unit_jpy=None):
    """Top TOP_N by similarity + the EXTRA_NEAR_TARGET of the rest priced closest to the item's
    target (or the most expensive ones when there is no budget). All already fit room + price cap."""
    head, rest = cands[:TOP_N], cands[TOP_N:]
    if target_unit_jpy:
        extra = sorted(rest, key=lambda c: abs(c[0]["price"] - target_unit_jpy))[:EXTRA_NEAR_TARGET]
    else:
        extra = sorted(rest, key=lambda c: -c[0]["price"])[:EXTRA_NEAR_TARGET]
    return head + extra


def check_bundle(intent: Intent, active, combo, zone_by_id):
    """Hard constraints for one candidate per item: total price within budget and one free zone
    per floor item. Returns (zones, pref, total) or None."""
    total = sum(r["price"] * intent.items[i].qty for i, (r, _) in zip(active, combo))
    if intent.budget is not None and total > intent.budget:
        return None
    assigned = assign_zones([(intent.items[i], fits) for i, (_, fits) in zip(active, combo)], zone_by_id)
    if assigned is None:
        return None
    zones, pref = assigned
    return zones, pref, total


def recommend(index: ProductIndex, intent: Intent, room: RoomProfile, scenario: str = "",
              rerank: bool = False) -> Recommendation:
    """Stage 1 (always): shortlist + best valid bundle by score.
    Stage 2 (rerank=True): Gemini picks the bundle from the same shortlist; used only if it
    passes the same hard checks, otherwise the score bundle is kept and the reason recorded."""
    per_item: List[List[Tuple[dict, Dict[str, Placement]]]] = []
    unmet = []
    targets = targets_jpy(intent.items, intent.budget)
    for k, it in enumerate(intent.items):
        c = candidates(index, it, room, intent.budget)
        if not c:
            unmet.append(f"{it.category}: không có sản phẩm nào vừa giá và vừa chỗ trong phòng")
        per_item.append(shortlist(c, targets[k] / it.qty if targets[k] else None))

    active = [i for i, c in enumerate(per_item) if c]
    zone_by_id = {z.id: z for z in room.zones}
    best, best_key = None, None
    valid = []                                   # every set that passes the hard checks
    for combo in itertools.product(*[per_item[i] for i in active]):
        checked = check_bundle(intent, active, combo, zone_by_id)
        if checked is None:
            continue
        zones, pref, total = checked
        viol = ratio_violations(intent.items, {i: r["price"] for i, (r, _) in zip(active, combo)})
        dev = allocation_deviation(intent.items, active, combo, intent.budget)
        valid.append((combo, zones, pref, total, viol, dev))
        # 1 price-ratio rules hold, 2 natural placement, 3 relevance with a mild penalty for a spend split
        # far from the targets (ALLOC_PENALTY per item at dev = 1; relevance must stay the main signal -
        # ranking allocation first picked a single bed for "giường đôi"), 4 uses more of the budget
        fit = sum(r["score"] for r, _ in combo) - ALLOC_PENALTY * len(active) * dev
        key = (not viol, pref, round(fit, 3), total)
        if best_key is None or key > best_key:
            best, best_key = (combo, zones, total), key

    ranker, note, comment, llm_reasons = "score", None, None, {}
    if rerank and best is not None and active:   # nothing to choose when no item has candidates
        from rerank import llm_rerank
        try:
            choice, comment, model = llm_rerank(intent, room, per_item, active)
            combo = tuple(next(c for c in per_item[i] if c[0]["item_no"] == choice[slot][0])
                          for slot, i in enumerate(active))
            checked = check_bundle(intent, active, combo, zone_by_id)
            llm_viol = ratio_violations(intent.items, {i: r["price"] for i, (r, _) in zip(active, combo)})
            any_ok = any(not v[4] for v in valid)
            if checked is None or (llm_viol and any_ok):
                total_llm = sum(r["price"] * intent.items[i].qty for i, (r, _) in zip(active, combo))
                why = (f"tổng {fmt_vnd(jpy_to_vnd(total_llm))} vượt ngân sách {fmt_vnd(jpy_to_vnd(intent.budget))}"
                       if intent.budget is not None and total_llm > intent.budget
                       else "các món không xếp được vào các vùng trống cùng lúc" if checked is None
                       else "sai tỉ lệ giá: " + "; ".join(llm_viol))
                # repair: the valid set that keeps the most of the LLM's picks (then placement, similarity, budget use)
                wanted = [c[0]["item_no"] for c in combo]
                same = lambda v: sum(c[0]["item_no"] == w for c, w in zip(v[0], wanted))
                pool = [v for v in valid if not v[4]] or valid      # respect the price-ratio rules if possible
                fix = max(pool, key=lambda v: (same(v), v[2],
                                               round(sum(c[0]["score"] for c in v[0]) - ALLOC_PENALTY * len(active) * v[5], 3), v[3]))
                kept = same(fix)
                if kept == 0:
                    note = f"{model} chọn bộ không hợp lệ ({why}) -> dùng bộ theo điểm"
                    comment = None
                else:
                    best = (fix[0], fix[1], fix[3])
                    ranker = f"{model}+repair"
                    comment = None   # the LLM's set comment described items that were replaced
                    note = (f"{model} chọn bộ không hợp lệ ({why}) -> giữ {kept}/{len(active)} món Gemini chọn, "
                            f"thay {len(active) - kept} món bằng lựa chọn hợp lệ tốt nhất")
                    llm_reasons = {i: choice[slot][1] for slot, (i, c) in enumerate(zip(active, fix[0]))
                                   if c[0]["item_no"] == choice[slot][0]}
            else:
                best = (combo,) + checked[:1] + checked[2:]
                ranker = model
                llm_reasons = {i: choice[slot][1] for slot, i in enumerate(active)}
        except Exception as e:   # LLMUnavailable, invalid item_no, ...
            note = f"rerank lỗi ({type(e).__name__}: {str(e)[:120]}) -> dùng bộ theo điểm"

    picks = []
    if best is None and active:
        unmet.append("không tìm được bộ sản phẩm nào vừa tổng ngân sách và đủ chỗ cho tất cả các món")
        total = 0
    elif best:
        combo, zones, total = best
        for i, (r, fits), zid in zip(active, combo, zones):
            it = intent.items[i]
            p = fits.get(zid) if zid else None
            alts = [f"{a['name']} {a['type_name']} {fmt_vnd(jpy_to_vnd(a['price']))}"
                    for a, _ in per_item[i] if a is not r][:3]
            facts = reason(it, r, p, room)
            k = i
            picks.append(Pick(
                category=it.category, item_no=r["item_no"], name=r["name"], type_name=r["type_name"],
                price=r["price"], price_vnd=jpy_to_vnd(r["price"]), qty=it.qty, score=round(r["score"], 4),
                size_cm=[r["w_cm"], r["d_cm"], r["h_cm"]], model_file=r["model_file"], placement=p,
                reason_vi=llm_reasons.get(i, facts), fit_vi=facts, alternatives=alts,
                target_vnd=jpy_to_vnd(targets[k]) if targets[k] else None))
    total = total if best else 0
    return Recommendation(scenario=scenario, request=intent.request, budget=intent.budget,
                          budget_vnd=jpy_to_vnd(intent.budget) if intent.budget else None,
                          total_price=total, total_vnd=sum(p.price_vnd * p.qty for p in picks),
                          picks=picks, unmet=unmet, ranker=ranker, rerank_note=note, comment_vi=comment,
                          price_rules=rules_for(intent.items),
                          rule_violations=ratio_violations(intent.items, {i: r["price"] for i, (r, _) in
                                                          zip(active, best[0])}) if best else [],
                          allocation_deviation=round(allocation_deviation(intent.items, active, best[0], intent.budget), 3)
                          if best and active else None)


def zone_preference(item: IntentItem, zone: Zone) -> int:
    """1 if the zone suits how the item is normally placed (wall items on a wall zone,
    free-standing items in an open zone), else 0."""
    against_wall = USAGE.get(item.category, (0, 0, False))[2]
    return int(against_wall == (zone.wall is not None))


def assign_zones(items_fits, zones: Dict[str, Zone]) -> Optional[Tuple[List[Optional[str]], int]]:
    """One zone per floor item; rugs and table-top items may share. Exhaustive (few items,
    few zones) - returns the assignment with the most 'natural' placements, or None."""
    best: List = [None, -1]
    result: List[Optional[str]] = [None] * len(items_fits)

    def go(k, used, pref):
        if k == len(items_fits):
            if pref > best[1]:
                best[0], best[1] = list(result), pref
            return
        item, fits = items_fits[k]
        if not needs_zone(item):
            result[k] = None
            go(k + 1, used, pref)
            return
        shared = item.category in SHARES_ZONE
        for zid in fits:
            if not shared and zid in used:
                continue
            result[k] = zid
            go(k + 1, used if shared else used | {zid}, pref + zone_preference(item, zones[zid]))
        result[k] = None

    go(0, frozenset(), 0)
    return (best[0], best[1]) if best[1] >= 0 else None


def reason(it: IntentItem, r: dict, p: Optional[Placement], room: RoomProfile) -> str:
    parts = [f"khớp \"{it.query}\" (độ giống {r['score']:.3f})"]
    if p:
        z = next(z for z in room.zones if z.id == p.zone_id)
        front, side, _ = USAGE.get(it.category, (0, 0, False))
        extra = f", chừa {front} cm phía trước" if front else ""
        extra += f", {side} cm mỗi bên" if side else ""
        parts.append(f"vừa vùng {z.id} ({z.label or 'trống'}, {z.w:.0f}×{z.h:.0f} cm){extra}")
    elif not needs_zone(it):
        parts.append("đặt trên đồ khác, không cần chỗ trên sàn")
    parts.append(f"giá {fmt_vnd(jpy_to_vnd(r['price']))}" + (f" × {it.qty}" if it.qty > 1 else ""))
    return "; ".join(parts)

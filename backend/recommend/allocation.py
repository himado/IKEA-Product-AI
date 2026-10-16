"""
Budget allocation across the items of one request (user request 2026-10-15: prices inside a set
must follow market logic - e.g. a dining chair cannot cost almost as much as the dining table).

Three pieces, all derived from two sources (see pipeline.md, step 7):
  A. market rules of thumb from interior-design / retailer budgeting guides
     - living room: sofa ~25-40%, accent chair ~10%, rug ~10%, coffee table ~8-9%, lamps in the rest
     - dining:      table ~45-55%, 4 chairs together ~30-40%
     - bedroom:     bed frame ~25-30% (no mattress), storage ~20%, dresser ~15%, nightstand cheapest
     - home office: chair >= desk (60/40 or 40/30 chair/desk)
  B. our own IKEA catalog (388 products): median price per category, e.g.
     dining chair = 0.31 x dining table, bedside table = 0.33 x bed, floor lamp = 0.18 x sofa,
     desk chair = 0.89 x desk

WEIGHTS   relative importance points per category (per unit). A request's target share for an
          item = weight x qty / sum over items.
RATIOS    hard-ish rules on UNIT prices between items of the same set (anchor vs. accessory).
TARGET_USE  aim to spend ~85% of the budget (range 60-100% is fine).
"""

from typing import Dict, List, Optional

WEIGHTS = {
    # living room - sofa is the anchor
    "sofa": 30, "armchair": 12, "ottoman_pouffe": 6, "coffee_side_table": 8, "tv_bench": 10,
    "rug": 10, "floor_lamp": 5, "table_lamp": 3, "bookcase_shelf": 8, "sideboard_cabinet": 10,
    # dining - table is the anchor; per chair = 0.2 x table (market ~0.18-0.2; IKEA median 0.31)
    "dining_table": 40, "dining_chair": 8,   # 4 chairs = 44%, table 56% (market: table largest, 45-55%)
    # bedroom - bed is the anchor, nightstand is where to save
    "bed": 28, "wardrobe": 20, "chest_of_drawers": 15, "bedside_table": 7,
    # home office - the chair deserves at least as much as the desk
    "desk_chair": 30, "desk": 25,
}
DEFAULT_WEIGHT = 8

# (item, anchor, max unit-price ratio item/anchor) - applied only when both are in the request
RATIOS = [
    ("dining_chair", "dining_table", 0.5),
    ("coffee_side_table", "sofa", 0.6),
    ("floor_lamp", "sofa", 0.5),
    ("floor_lamp", "armchair", 1.0),
    ("table_lamp", "desk", 0.5),
    ("bedside_table", "bed", 0.5),
    ("ottoman_pouffe", "sofa", 0.7),
    ("ottoman_pouffe", "armchair", 1.0),
    ("rug", "sofa", 1.0),
    ("rug", "armchair", 1.0),
    ("desk_chair", "desk", 2.0),
]
TARGET_USE = 0.85
RULE_TEXT = {
    ("dining_chair", "dining_table"): "mỗi ghế ăn ≤ 50% giá bàn ăn",
    ("coffee_side_table", "sofa"): "bàn trà ≤ 60% giá sofa",
    ("floor_lamp", "sofa"): "đèn cây ≤ 50% giá sofa",
    ("floor_lamp", "armchair"): "đèn cây ≤ giá ghế bành",
    ("table_lamp", "desk"): "đèn bàn ≤ 50% giá bàn làm việc",
    ("bedside_table", "bed"): "tủ đầu giường ≤ 50% giá giường",
    ("ottoman_pouffe", "sofa"): "đôn ≤ 70% giá sofa",
    ("ottoman_pouffe", "armchair"): "đôn ≤ giá ghế bành",
    ("rug", "sofa"): "thảm ≤ giá sofa",
    ("rug", "armchair"): "thảm ≤ giá ghế bành",
    ("desk_chair", "desk"): "ghế làm việc ≤ 2 lần giá bàn",
}


def target_shares(items) -> List[float]:
    """Share of the spend each requested item should take (sums to 1)."""
    w = [WEIGHTS.get(it.category, DEFAULT_WEIGHT) * it.qty for it in items]
    total = sum(w) or 1
    return [x / total for x in w]


def targets_jpy(items, budget: Optional[int]) -> List[Optional[int]]:
    """Target spend per item (all units together), or None without a budget."""
    if budget is None:
        return [None] * len(items)
    return [int(budget * TARGET_USE * s) for s in target_shares(items)]


def ratio_violations(items, prices: Dict[int, int]) -> List[str]:
    """items: IntentItem list; prices: {item index: UNIT price} for the chosen products."""
    cat_price = {}
    for i, price in prices.items():
        cat_price.setdefault(items[i].category, price)
    out = []
    for item, anchor, max_ratio in RATIOS:
        if item in cat_price and anchor in cat_price and cat_price[anchor] > 0:
            if cat_price[item] > max_ratio * cat_price[anchor]:
                out.append(f"{RULE_TEXT[(item, anchor)]} (đang {cat_price[item] / cat_price[anchor]:.0%})")
    return out


def rules_for(items) -> List[str]:
    cats = {it.category for it in items}
    return [RULE_TEXT[(a, b)] for a, b, _ in RATIOS if a in cats and b in cats]


def allocation_deviation(items, active, combo, budget: Optional[int] = None) -> float:
    """How far the chosen set's spend is from the per-item targets (0 = exactly on target).
    With a budget: sum |spend_i - target_i| / sum target_i - penalises a wrong split AND spending
    far too little/much (a first version compared shares only, so a cheap set with the right
    proportions looked perfect and budget use dropped). Without a budget: share distance."""
    sel = [items[i] for i in active]
    spend = [r["price"] * items[i].qty for i, (r, _) in zip(active, combo)]
    if budget:
        tgt = [budget * TARGET_USE * s for s in target_shares(sel)]
        return sum(abs(s - t) for s, t in zip(spend, tgt)) / (sum(tgt) or 1)
    total = sum(spend) or 1
    return 0.5 * sum(abs(s / total - t) for s, t in zip(spend, target_shares(sel)))

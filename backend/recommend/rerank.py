"""
Step 7 - LLM rerank (PLAN §5 4.3).

Two-stage recommendation:
  stage 1 (retrieve.py)  embedding + hard rules over all 388 products -> a SAFE shortlist:
                         ~6 candidates per requested item, each one already fits a free zone
                         and the price cap
  stage 2 (this file)    Gemini reads the request, the room and the shortlist TOGETHER and picks
                         one candidate per item - the set that best matches what the user meant
                         (size, use, style, colour harmony, value for money) - and explains each
                         choice in Vietnamese

Guard rails: the model may only answer with item numbers from the shortlist (checked here);
retrieve.py then re-checks the total budget and the zone assignment for the chosen set and
falls back to the score-ranked set if anything is off or Gemini is unavailable.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from currency import fmt_vnd, jpy_to_vnd  # noqa: E402
from allocation import rules_for, targets_jpy  # noqa: E402
from llm import structured  # noqa: E402
from schemas import Intent, RoomProfile  # noqa: E402


class SlotChoice(BaseModel):
    slot: int = Field(description="slot number from the input")
    item_no: str = Field(description="item_no of the chosen candidate - must be one of that slot's candidates")
    reason_vi: str = Field(description="1-2 Vietnamese sentences: why this product for this user and room, "
                                       "citing concrete facts from the input (size, price, material, colour)")


class RerankOut(BaseModel):
    choices: List[SlotChoice]
    comment_vi: str = Field(description="1-2 Vietnamese sentences about the set as a whole")


SYSTEM = """You are an interior-design assistant choosing IKEA furniture for a customer.
For every slot, pick exactly ONE candidate (by item_no) from that slot's list.
All candidates already fit the room and the price cap, so judge on:
1. what the customer actually asked for (size words like "nhỏ gọn"/"cho 4 người", use, seats, colour, style);
2. the set as a whole: colours, materials and style that go together;
3. price balance: each slot has target_vnd = its fair share of the budget (all units), derived
   from market practice - the anchor piece (sofa, dining table, bed, office chair) gets the largest
   share, accessories (lamps, bedside tables, side tables) small ones. Pick candidates priced near
   their target; and price_rules (unit-price ratios between items of the set, e.g. a dining chair
   at most 50% of the table) MUST hold.
4. budget: budget_vnd is what the customer is WILLING to spend, not a target to undercut.
   The TOTAL of price_vnd x qty over all slots must not exceed it, but a good set uses most of it
   (roughly 60-100%) on larger, better-built or better-matching items - e.g. a wider wardrobe or a
   proper coffee table instead of a small tray table when the room has space. A set that spends far
   below the budget is a weak answer unless the customer asked for something cheap or small, or the
   cheaper item clearly fits the request better. Balance the money across the items.
5. space: each free zone holds ONE floor item (a rug may lie under another item; items "not placed
   on the floor" need no zone). Check that every chosen floor item can get its own zone from its
   fits_zones - e.g. if two chosen items both fit only z1, the set is invalid.
retrieval_rank is only a text-similarity hint (1 = most similar wording); override it when the
facts say another candidate fits the request better.
Write reasons in natural Vietnamese for the customer. Use only facts given in the input;
never invent features, colours or measurements."""

_catalog: Dict[str, dict] = {}


def _product(item_no: str) -> dict:
    if not _catalog:
        for line in open(ROOT / "data" / "catalog.jsonl", encoding="utf-8"):
            p = json.loads(line)
            _catalog[p["item_no"]] = p
    return _catalog.get(item_no, {})


def _short_desc(item_no: str) -> str:
    p = _product(item_no)
    mats = "; ".join(f"{m.get('part', '')} {m.get('material', '')}".strip() for m in (p.get("materials") or [])[:2])
    return " ".join(x for x in [p.get("description") or "", (p.get("summary") or "")[:220], mats] if x)[:420]


def build_prompt(intent: Intent, room: RoomProfile, per_item, active) -> str:
    targets = targets_jpy(intent.items, intent.budget)
    slots = []
    for slot, i in enumerate(active):
        it = intent.items[i]
        slots.append({
            "slot": slot,
            "category": it.category,
            "qty": it.qty,
            "customer_words": it.query,
            "target_vnd": jpy_to_vnd(targets[i]) if targets[i] else None,
            "candidates": [{
                "item_no": r["item_no"],
                "name": f"{r['name']} {r['type_name']}",
                "price_vnd": jpy_to_vnd(r["price"]),
                "size_cm": f"{r['w_cm']:.0f} x {r['d_cm']:.0f} x {r['h_cm']:.0f} (W x D x H)",
                "fits_zones": sorted(fits) if fits else "not placed on the floor",
                "retrieval_rank": rank + 1,
                "description": _short_desc(r["item_no"]),
            } for rank, (r, fits) in enumerate(per_item[i])],
        })
    payload = {
        "request": intent.request,
        "budget_vnd": jpy_to_vnd(intent.budget) if intent.budget else None,
        "price_rules": rules_for(intent.items),
        "room": {"name": room.name, "size_cm": f"{room.width_cm:.0f} x {room.depth_cm:.0f}",
                 "free_zones": [{"id": z.id, "label": z.label, "size_cm": f"{z.w:.0f} x {z.h:.0f}",
                                 "against_wall": bool(z.wall)} for z in room.zones]},
        "slots": slots,
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def llm_rerank(intent: Intent, room: RoomProfile, per_item, active) -> Tuple[Dict[int, Tuple[str, str]], str, str]:
    """Returns ({slot: (item_no, reason_vi)}, comment_vi, model). Raises ValueError if the answer
    is not a valid choice from the shortlist (caller falls back)."""
    out, model = structured(build_prompt(intent, room, per_item, active), RerankOut, SYSTEM, thinking="MEDIUM")
    picks = {}
    for c in out.choices:
        if c.slot < 0 or c.slot >= len(active):
            raise ValueError(f"unknown slot {c.slot}")
        allowed = {r["item_no"] for r, _ in per_item[active[c.slot]]}
        if c.item_no not in allowed:
            raise ValueError(f"slot {c.slot}: {c.item_no} is not in the shortlist")
        picks[c.slot] = (c.item_no, c.reason_vi.strip())
    if len(picks) != len(active):
        raise ValueError(f"answered {len(picks)} of {len(active)} slots")
    return picks, out.comment_vi.strip(), model


def describe_total(total_jpy: int) -> str:
    return fmt_vnd(jpy_to_vnd(total_jpy))

"""
Step 6 - user intent understanding (Phase 2).

Vietnamese (or English) request  ->  Gemini 3.8 Flash (structured output)  ->  Intent
(backend/schemas.py), the exact input of step 7 (backend/recommend/retrieve.py).

Division of labour:
  LLM  : language - which of the 18 catalog categories each wanted item is, how many,
         an English search phrase per item, the budget number + currency as said
  code : arithmetic and rules - currency conversion to JPY, which chairs need their own
         floor zone, validation
If no item can be inferred, the result carries one clarification question instead.

Usage: python backend/intent/parse.py "phòng khách nhỏ, cần sofa 2 chỗ và đèn đọc sách, dưới 60 nghìn yên"
"""

import json
import sys
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "pipeline"))

from llm import LLMUnavailable, structured  # noqa: E402
from schemas import Intent, IntentItem  # noqa: E402

from currency import JPY_PER_UNIT  # noqa: E402  (fixed rates live in backend/currency.py)

CATEGORIES = {   # crawl category -> what it covers (shown to the LLM)
    "sofa": "sofa, ghế sofa, ghế băng dài (2-4 chỗ)",
    "armchair": "ghế bành, ghế thư giãn đơn, ghế đọc sách có tay vịn, ghế bập bênh",
    "ottoman_pouffe": "đôn, ghế đẩu thấp, ghế gác chân, pouffe",
    "dining_chair": "ghế ăn, ghế tựa bàn ăn",
    "desk_chair": "ghế văn phòng, ghế xoay, ghế làm việc, ghế gaming",
    "coffee_table": "bàn trà, bàn sofa, bàn phụ nhỏ cạnh ghế",
    "dining_table": "bàn ăn",
    "desk": "bàn làm việc, bàn học, bàn máy tính",
    "bed": "giường, khung giường",
    "bedside_table": "tủ đầu giường, bàn đầu giường",
    "wardrobe": "tủ quần áo",
    "chest_of_drawers": "tủ ngăn kéo",
    "bookcase_shelf": "kệ sách, giá sách, kệ để đồ",
    "sideboard_cabinet": "tủ thấp, tủ đựng đồ, tủ trưng bày, tủ buffet",
    "tv_bench": "kệ tivi, tủ tivi",
    "floor_lamp": "đèn cây, đèn đứng, đèn sàn",
    "table_lamp": "đèn bàn, đèn để bàn, đèn ngủ",
    "rug": "thảm, thảm trải sàn",
}
# the LLM sees "coffee_table" (clearer name); the catalog calls it coffee_side_table
TO_CATALOG = {"coffee_table": "coffee_side_table"}
Category = Literal[tuple(CATEGORIES)]


class LLMItem(BaseModel):
    category: Category
    user_words: str = Field(description="the part of the request about this item, in the user's own words")
    search_en: str = Field(description="short English product search phrase: type + size/seats/style/colour/use "
                                       "the user mentioned, e.g. 'compact 2-seat sofa, grey fabric'")
    qty: int = Field(description="how many of this item; 1 if not said")
    max_unit_price: Optional[float] = Field(description="price cap per unit if the user gave one for THIS item, "
                                                        "in budget_currency; null otherwise")
    min_width_cm: Optional[float] = Field(description="minimum product width in cm ONLY when the words imply it: "
                                                      "'giường đôi'/double bed -> 140, 'giường đơn' -> null, "
                                                      "'sofa 3 chỗ' -> 180, an explicit 'rộng ít nhất X' -> X; else null")
    max_width_cm: Optional[float] = Field(description="maximum product width in cm ONLY when the user states a limit: "
                                                      "'giường đơn' -> 120, 'lọt hốc rộng 50 cm' -> 50, "
                                                      "'không quá 1m2' -> 120; else null")


class LLMIntent(BaseModel):
    items: List[LLMItem] = Field(description="every furniture item the user asks for; empty if unclear")
    budget_amount: Optional[float] = Field(description="the total budget as a plain number, e.g. 60000 for "
                                                       "'60 nghìn', 1000000 for '1 triệu', 500000 for 'nửa triệu'; null if none")
    budget_currency: Optional[Literal["JPY", "VND", "USD"]] = Field(
        description="currency of the amounts: 'yên'/'yen'/'¥' -> JPY, 'đồng'/'triệu'/'nghìn' without yen -> VND, "
                    "'đô'/'$' -> USD; null if no money mentioned")
    room: Optional[str] = Field(description="room type if mentioned (living_room, bedroom, dining_room, "
                                            "home_office, kids_room, hallway), else null")
    clarification_question_vi: Optional[str] = Field(
        description="ONLY when no item can be inferred: one short Vietnamese question asking what furniture "
                    "they want. null otherwise")


SYSTEM = (
    "You turn a furniture shopping request (usually Vietnamese, sometimes English or without diacritics) "
    "into structured data for an IKEA room-planning app.\n"
    "Catalog categories (choose only from these):\n"
    + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items()) +
    "\nRules:\n"
    "- One item per distinct piece of furniture the user wants. Do not add items they did not ask for.\n"
    "- Map indirect descriptions to a category: 'chỗ ngồi xem phim cả nhà' -> sofa; 'đồ trải dưới bàn trà' "
    "-> rug; 'nơi cất quần áo' -> wardrobe; 'ánh sáng cho góc đọc' -> floor_lamp unless they say on a table/desk.\n"
    "- 'Bàn ăn kèm 4 ghế' = dining_table qty 1 + dining_chair qty 4.\n"
    "- Never convert currencies; report the amount and currency as said.\n"
    "- If the request names no furniture you can map (e.g. only 'làm đẹp phòng'), return no items and ask one "
    "short clarification question in Vietnamese. Products IKEA does not sell in these categories "
    "(fridge, TV, air conditioner) are not items."
)


class IntentResult(BaseModel):
    intent: Optional[Intent]
    clarification_question: Optional[str]
    raw: LLMIntent
    source: str = Field(description="model that produced it, or 'rules' when every Gemini model failed")


# ------------------------------------------------------------------ rule-based fallback
# Used only when the Gemini API is unavailable (it returned 503/504 for minutes during testing).
# Keywords are matched on diacritic-free lowercase text, longest first, so "tủ đầu giường"
# is not also read as "giường".
KEYWORDS = {
    "bedside_table": ["tu dau giuong", "ban dau giuong", "bedside", "nightstand"],
    "dining_table": ["ban an", "dining table"],
    "coffee_table": ["ban tra", "ban sofa", "ban phu", "coffee table", "side table"],
    "desk": ["ban lam viec", "ban hoc", "ban may tinh", "desk"],
    "dining_chair": ["ghe an", "dining chair"],
    "desk_chair": ["ghe van phong", "ghe xoay", "ghe lam viec", "ghe gaming", "office chair"],
    "armchair": ["ghe banh", "ghe thu gian", "ghe doc sach", "armchair", "ghe bap benh"],
    "ottoman_pouffe": ["ghe dau", "ghe gac chan", "pouffe", "footstool", "don "],
    "sofa": ["sofa", "ghe bang", "couch"],
    "bed": ["giuong", "bed"],
    "wardrobe": ["tu quan ao", "wardrobe", "cat quan ao"],
    "chest_of_drawers": ["tu ngan keo", "chest of drawers", "dresser"],
    "tv_bench": ["ke tivi", "ke tv", "tu tivi", "tu tv", "tv bench"],
    "bookcase_shelf": ["ke sach", "gia sach", "ke de do", "bookcase", "shelf"],
    "sideboard_cabinet": ["tu thap", "tu trung bay", "tu buffet", "sideboard", "cabinet"],
    "floor_lamp": ["den cay", "den dung", "den san", "den doc", "den de doc", "floor lamp", "reading lamp"],
    "table_lamp": ["den ban", "den de ban", "den ngu", "table lamp", "desk lamp"],
    "rug": ["tham", "rug", "carpet"],
}
NUM_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6}


def strip_accents(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def rule_based(text: str) -> LLMIntent:
    import re
    t = " " + strip_accents(text) + " "
    found = []   # (position, category, qty)
    for cat, words in sorted(KEYWORDS.items(), key=lambda kv: -max(len(w) for w in kv[1])):
        for w in sorted(words, key=len, reverse=True):
            i = t.find(w)
            if i < 0:
                continue
            before = t[max(0, i - 12):i].split()
            qty = 1
            if before:
                last = before[-1]
                qty = int(last) if last.isdigit() else NUM_WORDS.get(last, 1)
            found.append((i, cat, qty))
            t = t[:i] + " " * len(w) + t[i + len(w):]   # consume, so shorter keywords can't re-match
            break
    # a bare "ghế" (chair) left over: its kind depends on the other items ("bàn ăn kèm 4 ghế")
    cats = {c for _, c, _ in found}
    chair_cats = {"dining_chair", "desk_chair", "armchair", "sofa", "ottoman_pouffe"}
    m = re.search(r"(\d+|mot|hai|ba|bon|tu|nam|sau)?\s*(?:cai\s+|chiec\s+)?ghe\b", t)
    if m and not (cats & chair_cats):
        cat = "dining_chair" if "dining_table" in cats else "desk_chair" if "desk" in cats else "armchair"
        word = m.group(1) or ""
        found.append((m.start(), cat, int(word) if word.isdigit() else NUM_WORDS.get(word, 1)))
    items = [LLMItem(category=c, user_words=c, search_en=c.replace("_", " "), qty=q, max_unit_price=None,
                     min_width_cm=None, max_width_cm=None)
             for _, c, q in sorted(found)]

    amount, currency = None, None
    plain = strip_accents(text).replace("nua trieu", "0,5 trieu")
    money = re.compile(r"(\$)?\s*(\d{1,3}(?:\.\d{3})+|\d+(?:,\d+)?)\s*(k\b|nghin|ngan|trieu|tr\b)?\s*(yen|¥|jpy|dong|vnd|usd|do\b)?")
    for m in money.finditer(plain):
        dollar, num, mult, cur = m.groups()
        if not (dollar or mult or cur):
            continue          # a bare number is a quantity or a size ("1m6"), not money
        # "50.000" = fifty thousand (dot = thousands separator); "1,5" = one and a half
        amount = float(num.replace(".", "")) if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", num) else float(num.replace(",", "."))
        amount *= {"k": 1e3, "nghin": 1e3, "ngan": 1e3, "trieu": 1e6, "tr": 1e6}.get(mult or "", 1)
        cur = "usd" if dollar else (cur or "")
        currency = "JPY" if cur in ("yen", "¥", "jpy") else "USD" if cur in ("usd", "do") else "VND"
        break
    return LLMIntent(items=items, budget_amount=amount, budget_currency=currency, room=None,
                     clarification_question_vi=None if items else "Bạn muốn tìm món đồ nội thất nào?")


def to_jpy(amount: Optional[float], currency: Optional[str]) -> Optional[int]:
    if amount is None:
        return None
    return int(round(amount * JPY_PER_UNIT.get(currency or "JPY", 1.0)))


def parse_intent(text: str, use_cache: bool = True, allow_rules: bool = True) -> IntentResult:
    try:
        raw, source = structured(f"Request: {text}", LLMIntent, SYSTEM, use_cache=use_cache)
    except LLMUnavailable:
        if not allow_rules:
            raise
        raw, source = rule_based(text), "rules"
    if not raw.items:
        return IntentResult(intent=None, raw=raw, source=source,
                            clarification_question=raw.clarification_question_vi or "Bạn muốn tìm món đồ nội thất nào?")
    cats = {i.category for i in raw.items}
    items = []
    for it in raw.items:
        cat = TO_CATALOG.get(it.category, it.category)
        # chairs that go with a requested table/desk are placed around it, not in a zone of their own
        accompanying = (cat == "dining_chair" and "dining_table" in cats) or (cat == "desk_chair" and "desk" in cats)
        items.append(IntentItem(
            category=cat, query=it.search_en, qty=max(1, it.qty),
            max_price=to_jpy(it.max_unit_price, raw.budget_currency),
            needs_zone=False if accompanying else None,
            min_width_cm=it.min_width_cm, max_width_cm=it.max_width_cm))
    budget = to_jpy(raw.budget_amount, raw.budget_currency)
    if budget is None and len(items) == 1 and items[0].max_price:
        # "ghế gaming dưới 15000 yên": the LLM may file the cap under the item, not the total - same thing here
        budget = items[0].max_price * items[0].qty
    return IntentResult(intent=Intent(request=text, items=items, budget=budget),
                        clarification_question=None, raw=raw, source=source)


if __name__ == "__main__":
    res = parse_intent(" ".join(sys.argv[1:]) or "phòng khách nhỏ, cần sofa 2 chỗ và đèn đọc sách, dưới 60 nghìn yên")
    print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))

"""
Data contracts between pipeline stages (PLAN.md §4-§5). Pydantic so every stage
validates what it receives.

Until steps 1/6/12 exist, Intent and RoomProfile are written by hand
(eval/scenarios.json); later the LLM (step 6) and the room scanner (steps 8-12)
produce the same shapes.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Wall = Optional[Literal["top", "bottom", "left", "right"]]


class Zone(BaseModel):
    """A free floor rectangle in room coordinates (cm, origin bottom-left, y = into the room).
    `wall` = which side of the zone touches a wall (None = free-standing area)."""
    id: str
    x: float
    y: float
    w: float
    h: float
    wall: Wall = None
    label: str = ""


class RoomProfile(BaseModel):
    name: str
    width_cm: float
    depth_cm: float
    zones: List[Zone]


class IntentItem(BaseModel):
    category: str = Field(description="crawl category, e.g. sofa, rug, floor_lamp")
    query: str = Field(description="what the user wants, free text (any language)")
    qty: int = 1
    max_price: Optional[int] = Field(None, description="per-unit price cap (JPY)")
    needs_zone: Optional[bool] = Field(None, description="override: False for chairs that go around a table")
    min_width_cm: Optional[float] = Field(None, description="hard size filter on the model width (e.g. double bed >= 140)")
    max_width_cm: Optional[float] = Field(None, description="hard size filter on the model width (e.g. fits a 50 cm niche)")


class Intent(BaseModel):
    request: str = Field(description="the user's original sentence")
    items: List[IntentItem]
    budget: Optional[int] = Field(None, description="total budget for all items, in JPY (catalog currency; "
                                                    "users speak VND - see backend/currency.py)")


class Placement(BaseModel):
    zone_id: str
    x: float          # footprint rectangle in room cm (product itself)
    y: float
    w: float
    h: float
    clear_x: float    # rectangle incl. clearance
    clear_y: float
    clear_w: float
    clear_h: float
    front: Literal["up", "down", "left", "right"]   # direction the product faces (top-down)


class Pick(BaseModel):
    category: str
    item_no: str
    name: str
    type_name: str
    price: int                    # JPY (catalog)
    price_vnd: int                # shown to the user
    qty: int
    score: float                  # cosine similarity from retrieval (step 4)
    size_cm: List[float]          # w, d, h of the 3D model
    model_file: Optional[str]
    placement: Optional[Placement]
    reason_vi: str                # why this product (LLM rerank), or the fit facts when ranked by score
    fit_vi: str = ""              # facts checked by code: fits zone X with clearance, price
    target_vnd: Optional[int] = None   # budget share this item should take (allocation.py), all units
    alternatives: List[str] = []


class Recommendation(BaseModel):
    scenario: str
    request: str
    budget: Optional[int]         # JPY
    budget_vnd: Optional[int] = None
    total_price: int              # JPY
    total_vnd: int = 0
    picks: List[Pick]
    unmet: List[str] = []         # items that could not be satisfied, with why
    ranker: str = "score"         # "score" (retrieval order) or the Gemini model that reranked
    rerank_note: Optional[str] = None   # why the LLM choice was not used, if it was rejected
    comment_vi: Optional[str] = None    # LLM summary of the whole set
    price_rules: List[str] = []         # unit-price ratio rules that apply to this set (allocation.py)
    rule_violations: List[str] = []     # rules the final set still breaks (only if no set satisfies them)
    allocation_deviation: Optional[float] = None   # 0 = spend split exactly like the targets

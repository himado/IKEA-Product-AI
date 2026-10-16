"""
Step 3 - Product enrichment with Claude (Phase 3.3).

IKEA pages don't say *how* a product is placed in a room or what style it is,
but the recommender (step 7) and the placement engine (step 13) need exactly that.
For each product we send Claude the product facts + the main photo and get back a
fixed JSON schema (structured outputs, validated with Pydantic).

Input : data/catalog.jsonl, data/model_meta.jsonl (true size from the 3D model),
        data/images/<item_no>.jpg, ANTHROPIC_API_KEY in .env
Output: data/enriched/<item_no>.json   one file per product (cached: re-runs skip it)
        data/enriched.jsonl            all enriched products merged

Usage:
  python pipeline/enrich.py --limit 10            # test on 10 products, 1 per category
  python pipeline/enrich.py                       # all products
  python pipeline/enrich.py --items 60417637,10295025 --force
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import List, Literal, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

MODEL = "claude-haiku-5-5"
PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50)}   # input, output USD (prompts <= 100K tokens)

Category = Literal[
    "sofa", "sofa_bed", "armchair", "footstool_pouffe", "coffee_table", "side_table",
    "tv_bench", "media_storage", "dining_table", "dining_set", "dining_chair", "bar_stool",
    "desk", "desk_chair", "bed", "bunk_loft_bed", "bedside_table", "wardrobe",
    "chest_of_drawers", "bookcase", "shelving_unit", "sideboard", "cabinet",
    "display_cabinet", "floor_lamp", "table_lamp", "rug", "other",
]
Room = Literal["living_room", "bedroom", "dining_room", "home_office", "kids_room", "hallway", "kitchen"]
Style = Literal["scandinavian", "modern", "minimalist", "industrial", "rustic", "traditional",
                "mid_century", "boho", "japandi", "contemporary"]
Color = Literal["white", "black", "grey", "beige", "brown", "light_wood", "medium_wood", "dark_wood",
                "green", "blue", "red", "yellow", "pink", "orange", "metallic", "transparent", "multicolor"]
Material = Literal["wood", "wood_veneer", "particleboard", "metal", "fabric", "leather", "plastic",
                   "glass", "rattan", "bamboo", "wool", "cotton", "synthetic_fiber", "stone", "paper"]


class Enrichment(BaseModel):
    category_norm: Category = Field(description="Normalised product type. Use 'other' only if nothing fits.")
    placement_type: Literal["floor", "wall_mounted", "tabletop", "ceiling"] = Field(
        description="Where the product physically sits when in use.")
    against_wall: Literal["required", "preferred", "free"] = Field(
        description="required: must stand with its back to a wall (wardrobe, tall bookcase); "
                    "preferred: usually against a wall but not necessary (sofa, bed, TV bench); "
                    "free: commonly placed in open space (coffee table, rug, dining table).")
    front_clearance_cm: int = Field(
        description="Free floor space needed in front of the product to use it: open doors/drawers, "
                    "sit down, walk past. 0 for rugs.")
    side_clearance_cm: int = Field(description="Free space needed on each side (e.g. to get into a bed). Often 0.")
    seats: Optional[int] = Field(description="Number of people it seats/sleeps, or null if not applicable.")
    room_types: List[Room] = Field(description="Rooms this product is typically used in, most typical first.")
    styles: List[Style] = Field(description="1-3 interior styles it fits, best match first.")
    colors: List[Color] = Field(description="Visible colours, dominant first.")
    materials: List[Material] = Field(description="Main visible materials.")
    functions: List[str] = Field(description="2-6 short English tags about use, e.g. storage, reading, "
                                             "compact, extendable, stackable, kids, washable_cover.")
    summary_vi: str = Field(description="One natural Vietnamese sentence describing the product for a shopper.")


SYSTEM = """You label IKEA products for a room-planning AR app.
The app uses your labels to (1) filter products for a user's request and (2) decide where in a
scanned room the 3D model can be placed. Be practical and literal:
- Base answers on the product facts and the photo. Do not invent features.
- The 3D model size (model_dims_cm, x=width, z=depth, y=height) is the true size; IKEA's
  stated dimensions may describe only part of the product.
- Clearances are real-world centimetres a person needs to use the item (wardrobe doors,
  drawers pulled out, sitting down on a chair, getting into a bed).
- category_norm describes what the product actually is, even if IKEA filed it elsewhere
  (e.g. a laptop stand listed under desks is 'other')."""


def product_facts(p: dict, meta: dict) -> dict:
    return {
        "name": p.get("name"),
        "type_name": p.get("type_name"),
        "ikea_category_path": p.get("category_path"),
        "description": p.get("description"),
        "summary": p.get("summary"),
        "long_description": (p.get("long_description") or [])[:4],
        "materials": p.get("materials"),
        "good_to_know": (p.get("good_to_know") or [])[:5],
        "stated_measurements": p.get("measurements"),
        "model_dims_cm": meta.get("dims_cm"),
        "price": f'{p.get("price")} {p.get("currency")}',
    }


def enrich_one(client, p: dict, meta: dict, data_dir: Path, model: str) -> dict:
    content = []
    img = data_dir / (p.get("image_file") or f"images/{p['item_no']}.jpg")
    if img.exists():
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(img.read_bytes()).decode()}})
    content.append({"type": "text", "text": "Product facts:\n" + json.dumps(product_facts(p, meta), ensure_ascii=False)})

    t0 = time.time()
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},   # short labelling task
        messages=[{"role": "user", "content": content}],
        output_format=Enrichment,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused: {response.stop_details}")
    if response.parsed_output is None:
        raise RuntimeError(f"no parsed output (stop_reason={response.stop_reason})")
    u = response.usage
    return {
        "item_no": p["item_no"],
        "name": p.get("name"),
        "type_name": p.get("type_name"),
        "crawl_category": p.get("crawl_category"),
        **response.parsed_output.model_dump(),
        "_model": response.model,
        "_usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens},
        "_seconds": round(time.time() - t0, 1),
        "_request_id": response._request_id,
    }


def pick(products, limit, items):
    if items:
        wanted = set(items)
        return [p for p in products if p["item_no"] in wanted]
    if not limit:
        return products
    # round-robin over categories so a small test covers as many product types as possible
    by_cat = {}
    for p in products:
        by_cat.setdefault(p.get("crawl_category"), []).append(p)
    out, i = [], 0
    while len(out) < limit and any(len(v) > i for v in by_cat.values()):
        out += [v[i] for v in by_cat.values() if len(v) > i][: limit - len(out)]
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Enrich IKEA products with Claude")
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0, help="only N products (1 per category first)")
    ap.add_argument("--items", default="", help="comma-separated item numbers")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    load_dotenv()
    data_dir = Path(args.data)
    out_dir = data_dir / "enriched"
    out_dir.mkdir(exist_ok=True)
    products = [json.loads(l) for l in open(data_dir / "catalog.jsonl", encoding="utf-8")]
    meta = {m["item_no"]: m for m in map(json.loads, open(data_dir / "model_meta.jsonl", encoding="utf-8"))}
    todo = pick(products, args.limit, [i.strip() for i in args.items.split(",") if i.strip()])

    client = anthropic.Anthropic()
    tin = tout = done = failed = 0
    for n, p in enumerate(todo, 1):
        path = out_dir / f"{p['item_no']}.json"
        if path.exists() and not args.force:
            print(f"[{n}/{len(todo)}] cached {p['item_no']} {p.get('name')}")
            continue
        try:
            rec = enrich_one(client, p, meta.get(p["item_no"], {}), data_dir, args.model)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            print(f"STOP: API key problem - {e}")
            break
        except anthropic.BadRequestError as e:
            if "credit balance" in str(e):
                print("STOP: Anthropic account has no credit - add credits at console.anthropic.com > Billing")
                break
            failed += 1
            print(f"[{n}/{len(todo)}] FAILED {p['item_no']} {p.get('name')}: {e}")
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
            failed += 1
            print(f"[{n}/{len(todo)}] FAILED {p['item_no']} {p.get('name')}: {e}")
            continue
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        tin += rec["_usage"]["input_tokens"]
        tout += rec["_usage"]["output_tokens"]
        done += 1
        print(f"[{n}/{len(todo)}] {p['item_no']} {p.get('name')} {p.get('type_name')} -> "
              f"{rec['category_norm']}, {rec['placement_type']}, wall={rec['against_wall']}, "
              f"front={rec['front_clearance_cm']}cm, {rec['room_types'][:2]}, {rec['styles'][:2]} ({rec['_seconds']}s)")

    # merge everything enriched so far (including earlier runs)
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(out_dir.glob("*.json"))]
    with open(data_dir / "enriched.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pin, pout = PRICE_PER_MTOK.get(args.model, (0, 0))
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    print(f"\nthis run: {done} enriched, {failed} failed, {tin:,} input + {tout:,} output tokens, ~${cost:.4f}")
    if done:
        print(f"average per product: {tin // done:,} in / {tout // done:,} out tokens, ~${cost / done:.5f}")
    print(f"enriched.jsonl: {len(rows)} products")


if __name__ == "__main__":
    main()

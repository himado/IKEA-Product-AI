"""
Step 4 - Embedding + index (Phase 3.4).

Turns every product in the catalog into one text "document", embeds it with a
multilingual sentence-embedding model (so Vietnamese queries match English IKEA text),
and stores the vectors next to the structured fields the recommender filters on.

Storage: for ~400 products a brute-force cosine search over a numpy matrix takes <1 ms,
so the index is plain files (no database needed for the demo). The same files can be
loaded into Postgres + pgvector later without re-embedding.

Input : data/catalog.jsonl, data/model_meta.jsonl (true size from the 3D model),
        data/enriched.jsonl (optional - step 3, merged in when present)
Output: data/index/embeddings.npy   float32 [N, dim], L2-normalised
        data/index/products.jsonl   row i = metadata of embeddings[i] (filter fields + doc text)
        data/index/meta.json        model name, dim, count, build time

Usage:
  python pipeline/index.py
  python pipeline/index.py --model BAAI/bge-m3      # stronger, needs ~2.5 GB RAM
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")  # transformers would otherwise also load TensorFlow (~1 GB RAM)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

# e5 models are trained with these prefixes; other models get none
PREFIX = {"intfloat/": ("query: ", "passage: ")}


def prefixes(model_name: str):
    for k, v in PREFIX.items():
        if model_name.startswith(k):
            return v
    return "", ""


# Bilingual (Vietnamese + English) names per crawl category. IKEA text is English only and a
# small multilingual model matches Vietnamese queries to it poorly; putting the product type
# in both languages at the start of the document fixes most of that without any LLM call.
CATEGORY_TEXT = {
    "sofa": "ghế sofa, sofa, ghế băng dài phòng khách | sofa, couch",
    "armchair": "ghế bành, ghế thư giãn, ghế đơn có tay vịn | armchair, easy chair",
    "coffee_side_table": "bàn trà, bàn sofa, bàn phụ, bàn nhỏ cạnh ghế | coffee table, side table",
    "tv_bench": "kệ tivi, tủ tivi, kệ để TV | TV bench, media unit",
    "ottoman_pouffe": "đôn, ghế đẩu thấp, ghế gác chân | footstool, pouffe, ottoman",
    "dining_table": "bàn ăn | dining table",
    "dining_chair": "ghế ăn, ghế tựa | dining chair",
    "desk": "bàn làm việc, bàn học, bàn máy tính | desk, work table",
    "desk_chair": "ghế văn phòng, ghế xoay, ghế làm việc | office chair, desk chair, swivel chair",
    "bed": "giường, khung giường | bed, bed frame",
    "bedside_table": "tủ đầu giường, bàn đầu giường | bedside table, nightstand",
    "wardrobe": "tủ quần áo | wardrobe, closet",
    "chest_of_drawers": "tủ ngăn kéo, tủ nhiều ngăn kéo | chest of drawers, dresser",
    "bookcase_shelf": "kệ sách, giá sách, kệ để đồ | bookcase, shelving unit",
    "sideboard_cabinet": "tủ thấp, tủ đựng đồ, tủ trưng bày | sideboard, cabinet",
    "floor_lamp": "đèn cây, đèn đứng, đèn sàn | floor lamp",
    "table_lamp": "đèn bàn, đèn để bàn | table lamp, desk lamp",
    "rug": "thảm, thảm trải sàn | rug, carpet",
}


def doc_text(p: dict, meta: dict, enr: dict) -> str:
    """One searchable paragraph per product. Order = most important first
    (the model truncates long inputs at 512 tokens)."""
    d = meta.get("dims_cm") or {}
    parts = [
        CATEGORY_TEXT.get(p.get("crawl_category"), "") + ".",
        f"{p.get('name')} {p.get('type_name')}.",
        p.get("description") or "",
        p.get("summary") or "",
        "Category: " + " > ".join((p.get("category_path") or [])[1:]),
        f"Size: width {d.get('w')} cm, depth {d.get('d')} cm, height {d.get('h')} cm." if d else "",
        "Materials: " + "; ".join(f"{m.get('part', '')} {m.get('material', '')}".strip()
                                   for m in (p.get("materials") or [])[:4]),
    ]
    if enr:  # step 3 output, when available
        parts += [
            "Rooms: " + ", ".join(enr.get("room_types", [])),
            "Style: " + ", ".join(enr.get("styles", [])),
            "Colors: " + ", ".join(enr.get("colors", [])),
            "Use: " + ", ".join(enr.get("functions", [])),
            enr.get("summary_vi", ""),
        ]
    parts += (p.get("long_description") or [])[:2]
    return " ".join(x.strip() for x in parts if x and x.strip())


def row(p: dict, meta: dict, enr: dict, text: str) -> dict:
    d = meta.get("dims_cm") or {}
    return {
        "item_no": p["item_no"],
        "name": p.get("name"),
        "type_name": p.get("type_name"),
        "category": p.get("crawl_category"),
        "category_norm": enr.get("category_norm"),
        "placement_type": enr.get("placement_type"),
        "against_wall": enr.get("against_wall"),
        "price": p.get("price"),
        "currency": p.get("currency"),
        "w_cm": d.get("w"), "d_cm": d.get("d"), "h_cm": d.get("h"),
        "rating": p.get("rating"),
        "model_file": p.get("model_file"),
        "image_file": p.get("image_file"),
        "url": p.get("url"),
        "doc": text,
    }


def main():
    ap = argparse.ArgumentParser(description="Embed the product catalog")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=16, help="small batches keep RAM use low")
    args = ap.parse_args()
    data = Path(args.data)
    out = data / "index"
    out.mkdir(exist_ok=True)

    products = [json.loads(l) for l in open(data / "catalog.jsonl", encoding="utf-8")]
    meta = {m["item_no"]: m for m in map(json.loads, open(data / "model_meta.jsonl", encoding="utf-8"))}
    enriched = {}
    if (data / "enriched.jsonl").exists():
        enriched = {e["item_no"]: e for e in map(json.loads, open(data / "enriched.jsonl", encoding="utf-8"))}

    rows = []
    for p in products:
        m, e = meta.get(p["item_no"], {}), enriched.get(p["item_no"], {})
        rows.append(row(p, m, e, doc_text(p, m, e)))
    print(f"{len(rows)} products, {len(enriched)} with enrichment, model {args.model}")

    from sentence_transformers import SentenceTransformer  # heavy import, after cheap checks
    t0 = time.time()
    model = SentenceTransformer(args.model, device="cpu")
    _, doc_prefix = prefixes(args.model)
    emb = model.encode([doc_prefix + r["doc"] for r in rows], batch_size=args.batch,
                       normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    secs = time.time() - t0

    np.save(out / "embeddings.npy", emb)
    with open(out / "products.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "model": args.model, "dim": int(emb.shape[1]), "count": len(rows),
        "with_enrichment": len(enriched), "normalized": True,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "encode_seconds": round(secs, 1),
    }, indent=2), encoding="utf-8")
    print(f"embeddings {emb.shape} in {secs:.1f}s -> {out}")


if __name__ == "__main__":
    main()

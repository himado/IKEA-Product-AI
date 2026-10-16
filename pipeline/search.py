"""
Search the product index: hard filters first (exact), then rank by meaning (cosine).

This is the retrieval core that step 7 (recommendation) will call. Hard filters make
budget/size constraints exact - an embedding alone would happily return an 80,000 yen
sofa for "under 30,000 yen".

Usage:
  python pipeline/search.py "ghế đọc sách nhỏ gọn" --max-price 20000
  python pipeline/search.py "sofa 3 chỗ màu xám" --category sofa --max-width 220 -k 5
  python pipeline/search.py --eval          # run the built-in Vietnamese test queries
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")  # transformers would otherwise also load TensorFlow (~1 GB RAM)

from index import prefixes


class ProductIndex:
    def __init__(self, data_dir="data"):
        d = Path(data_dir) / "index"
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.emb = np.load(d / "embeddings.npy")
        self.rows = [json.loads(l) for l in open(d / "products.jsonl", encoding="utf-8")]
        self._model = None

    def embed_query(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.meta["model"], device="cpu")
        q_prefix, _ = prefixes(self.meta["model"])
        return self._model.encode([q_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]

    def search(self, query: str, k=5, category=None, max_price=None, min_price=None,
               max_w=None, max_d=None, max_h=None):
        mask = np.ones(len(self.rows), dtype=bool)
        for i, r in enumerate(self.rows):
            if category and r["category"] not in category:
                mask[i] = False
            elif max_price is not None and (r["price"] is None or r["price"] > max_price):
                mask[i] = False
            elif min_price is not None and (r["price"] is None or r["price"] < min_price):
                mask[i] = False
            elif max_w is not None and (r["w_cm"] or 0) > max_w:
                mask[i] = False
            elif max_d is not None and (r["d_cm"] or 0) > max_d:
                mask[i] = False
            elif max_h is not None and (r["h_cm"] or 0) > max_h:
                mask[i] = False
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        scores = self.emb[idx] @ self.embed_query(query)
        top = np.argsort(-scores)[:k]
        return [{**self.rows[idx[j]], "score": float(scores[j])} for j in top]


# (query, filters, categories that count as a correct hit) - used by --eval
EVAL = [
    ("ghế đọc sách nhỏ gọn", {"max_price": 20000}, {"armchair", "desk_chair", "dining_chair", "ottoman_pouffe"}),
    ("sofa cho phòng khách nhỏ", {}, {"sofa"}),
    ("giường đôi có ngăn chứa đồ", {}, {"bed"}),
    ("bàn làm việc cho laptop", {}, {"desk"}),
    ("tủ quần áo cửa trượt", {}, {"wardrobe"}),
    ("kệ sách bằng gỗ", {}, {"bookcase_shelf"}),
    ("đèn đọc sách đặt cạnh ghế", {}, {"floor_lamp", "table_lamp"}),
    ("thảm trải sàn phòng khách", {}, {"rug"}),
    ("bàn ăn cho 4 người", {}, {"dining_table"}),
    ("kệ để tivi", {}, {"tv_bench"}),
    ("tủ đầu giường", {}, {"bedside_table"}),
    ("tủ ngăn kéo cho phòng ngủ", {}, {"chest_of_drawers"}),
    ("bàn trà nhỏ", {}, {"coffee_side_table"}),
    ("ghế xoay văn phòng", {}, {"desk_chair"}),
    ("đèn bàn ánh sáng ấm", {}, {"table_lamp"}),
    # held out: phrased without the category words used in index.CATEGORY_TEXT
    ("chỗ ngồi êm ái để xem phim cả nhà", {}, {"sofa", "armchair"}),
    ("nơi cất quần áo cho phòng ngủ", {}, {"wardrobe", "chest_of_drawers"}),
    ("ánh sáng cho góc đọc sách", {}, {"floor_lamp", "table_lamp"}),
    ("chỗ đặt máy tính để làm việc ở nhà", {}, {"desk"}),
    ("trải dưới bàn trà cho ấm chân", {}, {"rug"}),
]
HELD_OUT = 5  # last N entries of EVAL


def show(results):
    for r in results:
        print(f"  {r['score']:.3f}  {r['item_no']}  {r['name']:<14} {str(r['type_name'])[:38]:<38} "
              f"{r['category']:<18} ¥{r['price']:>7,}  {r['w_cm']}x{r['d_cm']}x{r['h_cm']} cm")


def run_eval(index: ProductIndex, k=5, verbose=True):
    groups = {"main": EVAL[:-HELD_OUT], "held_out": EVAL[-HELD_OUT:]}
    for name, items in groups.items():
        hits1 = hitsk = 0
        for q, f, ok in items:
            res = index.search(q, k=k, **f)
            good = [r["category"] in ok for r in res]
            hits1 += bool(good and good[0])
            hitsk += sum(good)
            if verbose:
                print(f"\n{q} {f or ''}  -> top1 {'OK' if good and good[0] else 'MISS'}, {sum(good)}/{len(res)} relevant")
                show(res)
        n = len(items)
        print(f"\n[{name}] {n} queries  Top-1: {hits1}/{n} = {hits1 / n:.0%}   "
              f"Precision@{k}: {hitsk}/{n * k} = {hitsk / (n * k):.0%}")


def main():
    ap = argparse.ArgumentParser(description="Search the IKEA product index")
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--data", default="data")
    ap.add_argument("--category", help="comma-separated crawl categories")
    ap.add_argument("--max-price", type=int)
    ap.add_argument("--min-price", type=int)
    ap.add_argument("--max-width", type=float)
    ap.add_argument("--max-depth", type=float)
    ap.add_argument("--max-height", type=float)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="with --eval: only print the scores")
    args = ap.parse_args()

    index = ProductIndex(args.data)
    if args.eval:
        run_eval(index, args.k, verbose=not args.quiet)
        return
    if not args.query:
        ap.error("give a query or --eval")
    res = index.search(args.query, k=args.k,
                       category=set(args.category.split(",")) if args.category else None,
                       max_price=args.max_price, min_price=args.min_price,
                       max_w=args.max_width, max_d=args.max_depth, max_h=args.max_height)
    print(f"{args.query}  ({index.meta['model']}, {len(index.rows)} products)")
    show(res)


if __name__ == "__main__":
    main()

"""
Catalog statistics: products per category, price range, missing fields, model sizes.

Input : data/catalog.jsonl (+ data/model_meta.jsonl if present)
Output: printed table

Usage: python pipeline/stats.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ["name", "type_name", "summary", "price", "dimensions_cm", "materials", "category_path", "image_file"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)

    rows = [json.loads(l) for l in open(data / "catalog.jsonl", encoding="utf-8")]
    meta = {}
    if (data / "model_meta.jsonl").exists():
        meta = {m["item_no"]: m for m in map(json.loads, open(data / "model_meta.jsonl", encoding="utf-8"))}

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("crawl_category", "?")].append(r)

    print(f"{'category':<18} {'n':>3} {'price min':>10} {'price max':>10} {'avg MB':>7} {'AR ok':>8}")
    for cat, items in sorted(by_cat.items()):
        prices = [i["price"] for i in items if i.get("price")]
        mb = sum(i.get("model_bytes", 0) for i in items) / len(items) / 1e6
        ok = sum(bool(meta.get(i["item_no"], {}).get("ar_usable")) for i in items)
        print(f"{cat:<18} {len(items):>3} {min(prices, default=0):>10,} {max(prices, default=0):>10,} {mb:>7.2f} "
              f"{(str(ok) + '/' + str(len(items))) if meta else '-':>8}")

    total_mb = sum(r.get("model_bytes", 0) for r in rows) / 1e6
    print(f"\nTotal: {len(rows)} products, {len(by_cat)} categories, {total_mb:.0f} MB of models")
    if meta:  # model_meta reads extensionsUsed from the file itself (the URL hint is unreliable)
        print(f"Draco-compressed: {sum(bool(m.get('draco')) for m in meta.values())}/{len(meta)}")
        print(f"AR usable (step 2): {sum(bool(m.get('ar_usable')) for m in meta.values())}/{len(meta)}")
    print("\nMissing fields:")
    for f in FIELDS:
        miss = sum(1 for r in rows if not r.get(f))
        if miss:
            print(f"  {f:<15} {miss}")
    no_h = sum(1 for r in rows if "height" not in (r.get("dimensions_cm") or {}))
    print(f"  {'height (dims)':<15} {no_h}   <- model bbox is used instead (step 2)")


if __name__ == "__main__":
    main()

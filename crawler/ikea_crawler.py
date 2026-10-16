"""
IKEA product + 3D model crawler (Phase 3 - Product Knowledge).

How it works:
  1. Product URLs come from IKEA category listing pages (CATEGORIES below),
     with a per-category quota so the catalog covers every furniture type.
  2. Each product page embeds a schema.org "3DModel" JSON-LD block whose
     contentUrl points to a .glb file (Draco-compressed glTF binary).
  3. Product metadata (name, type, price, dimensions, materials, category...)
     is read from the page's <script type="text/hydrate"> JSON.

Output (default ./data):
  data/models/<itemNo>.glb        3D model
  data/images/<itemNo>.jpg        main product image
  data/products/<itemNo>.json     full metadata per product
  data/catalog.jsonl              one line per product (for embeddings/RAG later)

Usage:
  python crawler/ikea_crawler.py --per-category 25
  python crawler/ikea_crawler.py --per-category 5 --only sofa,armchair
  python crawler/ikea_crawler.py --url https://www.ikea.com/jp/en/p/...   # extra seeds
"""

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from pathlib import Path

import requests

log = logging.getLogger("ikea")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
CATEGORY_URL = "https://www.ikea.com/{country}/{lang}/cat/{slug}/"

# Normalised category -> IKEA category page slugs (verified on ikea.com/jp/en).
# A listing page server-renders ~70 product links; roughly half have a 3D model.
CATEGORIES = {
    "sofa": ["sofas-fu003", "two-seater-fabric-sofas-10668", "three-seater-fabric-sofas-10670"],
    "armchair": ["upholstered-armchairs-10687", "living-room-chairs-footstools-fu006"],
    "coffee_side_table": ["living-room-tables-10705"],
    "tv_bench": ["tv-media-furniture-10475"],
    "ottoman_pouffe": ["ottomans-footstools-pouffes-20926"],
    "dining_table": ["dining-tables-21825"],
    "dining_chair": ["dining-chairs-25219"],
    "desk": ["desks-for-home-20651", "desks-computer-desks-20649"],
    "desk_chair": ["desk-chairs-20652"],
    "bed": ["double-beds-16284", "single-beds-16285"],
    "bedside_table": ["bedside-tables-20656"],
    "wardrobe": ["wardrobes-19053"],
    "chest_of_drawers": ["chest-of-drawers-10451"],
    "bookcase_shelf": ["bookcases-10382", "shelving-units-11465"],
    "sideboard_cabinet": ["sideboards-buffets-10412", "cabinets-10409"],
    "floor_lamp": ["floor-lamps-10731"],
    "table_lamp": ["table-lamps-10732"],
    "rug": ["rugs-10653"],
}

LDJSON_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
HYDRATE_RE = re.compile(r'<script type="text/hydrate"[^>]*>(.*?)</script>', re.S)


class IkeaCrawler:
    def __init__(self, out_dir: Path, country: str = "jp", lang: str = "en", delay: float = 1.5):
        self.out = out_dir
        self.country = country
        self.lang = lang
        self.delay = delay
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": f"{lang},en;q=0.8"})
        for sub in ("models", "images", "products"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ http
    def _get(self, url: str, **kw) -> requests.Response:
        for attempt in range(3):
            try:
                r = self.s.get(url, timeout=30, **kw)
                if r.status_code == 200:
                    return r
                log.warning("HTTP %s for %s", r.status_code, url)
                if r.status_code in (403, 404):
                    r.raise_for_status()
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                log.warning("retry %d %s: %s", attempt + 1, url, e)
            time.sleep(2 ** attempt + random.random())
        r.raise_for_status()
        return r

    def _sleep(self):
        time.sleep(self.delay + random.random() * 0.5)

    # ------------------------------------------------------------- discovery
    def category_product_urls(self, slug: str):
        """Product URLs server-rendered on a category listing page, in page order."""
        url = CATEGORY_URL.format(country=self.country, lang=self.lang, slug=slug)
        html = self._get(url).text
        pat = rf"https://www\.ikea\.com/{self.country}/{self.lang}/p/[a-z0-9-]+-s?\d{{8}}/"
        return list(dict.fromkeys(re.findall(pat, html)))

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _find_key(obj, key):
        """Depth-first search for the first value stored under `key`."""
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            obj = obj.values()
        if isinstance(obj, (list, tuple, type({}.values()))):
            for v in obj:
                found = IkeaCrawler._find_key(v, key)
                if found is not None:
                    return found
        return None

    def parse_product(self, url: str, html: str) -> dict:
        ld_blocks = []
        for raw in LDJSON_RE.findall(html):
            try:
                ld_blocks.append(json.loads(raw))
            except json.JSONDecodeError:
                pass

        model_url = None
        category_path = []
        for b in ld_blocks:
            if b.get("@type") == "3DModel":
                for enc in b.get("encoding", []):
                    if enc.get("encodingFormat") == "model/gltf-binary" or ".glb" in enc.get("contentUrl", ""):
                        model_url = enc["contentUrl"]
                        break
            elif b.get("@type") == "BreadcrumbList":
                category_path = [e.get("name") or e.get("item", {}).get("name")
                                 for e in b.get("itemListElement", [])]

        page_props = {}
        for raw in HYDRATE_RE.findall(html):
            # The wrapper around pageProps changes between IKEA releases
            # ({"pageProps":..} vs {"0":{"loaderData":{"pageProps":..}}}), so search for it.
            if '"pageProps"' not in raw[:200]:
                continue
            try:
                page_props = self._find_key(json.loads(raw), "pageProps") or {}
            except json.JSONDecodeError:
                continue
            if page_props.get("product"):
                break
        if not page_props.get("product"):
            log.warning("  pageProps not found - page layout may have changed")

        product = page_props.get("product", {})
        info = page_props.get("productInformationSectionProps", {})
        meas = info.get("measurementsProps", {})
        details = info.get("productDetailsProps", {}).get("accordionObject", {})
        summary = page_props.get("productSummaryProps", {})

        materials = []
        for group in (details.get("materialsAndCare", {}).get("contentProps", {}).get("materials") or []):
            for m in group.get("materials", []):
                materials.append({"part": m.get("part"), "material": m.get("material")})

        good_to_know = [g.get("text") for g in
                        (details.get("goodToKnow", {}).get("contentProps", {}).get("goodToKnow") or [])]

        prod_desc = info.get("productDetailsProps", {}).get("productDescriptionProps", {})

        images = []
        for m in product.get("mediaList") or []:
            c = m.get("content", {})
            if m.get("type", "").upper() == "IMAGE" or c.get("url", "").endswith((".jpg", ".png", ".avif", ".webp")):
                if c.get("url"):
                    images.append(c["url"])
        main_image = (meas.get("fallbackImage") or {}).get("url") or (images[0] if images else None)

        rating = product.get("ratingProps") or {}

        return {
            "item_no": product.get("itemNo") or re.search(r"(\d{8})/?$", url).group(1),
            "visible_item_no": product.get("visibleItemNo"),
            "name": product.get("name"),
            "type_name": summary.get("typeName") or self._find_key(page_props, "typeName"),
            "description": product.get("description"),
            "summary": summary.get("description"),
            "long_description": prod_desc.get("paragraphs"),
            "designer": prod_desc.get("designerName"),
            "price": product.get("price"),
            "currency": product.get("currencyCode"),
            "rating": rating.get("rating"),
            "review_count": product.get("reviewCount"),
            "category_path": [c for c in category_path if c],
            # assembled product dimensions, e.g. [{"name": "Height", "measure": "55 cm"}]
            "measurements": [{"name": m.get("name"), "measure": m.get("measure")}
                             for m in meas.get("measurements") or []],
            "dimensions_cm": self._dims_cm(meas.get("measurements") or []),
            "materials": materials,
            "good_to_know": good_to_know,
            "url": url,
            "model_url": model_url,
            "image_url": main_image,
            "images": images[:8],
        }

    @staticmethod
    def _dims_cm(measurements):
        """Normalise Width/Depth/Height/Length/Diameter to centimetres (for AR scaling)."""
        out = {}
        for m in measurements:
            name = (m.get("name") or "").lower()
            mm = re.match(r"([\d.,]+)\s*(mm|cm|m)\b", m.get("measure") or "")
            if not mm:
                continue
            val = float(mm.group(1).replace(",", ""))
            val *= {"mm": 0.1, "cm": 1, "m": 100}[mm.group(2)]
            for key in ("width", "depth", "height", "length", "diameter"):
                if name == key and key not in out:
                    out[key] = round(val, 1)
        return out

    # -------------------------------------------------------------- download
    def _download(self, url: str, path: Path) -> int:
        if path.exists() and path.stat().st_size > 0:
            return path.stat().st_size
        r = self._get(url, stream=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(64 * 1024):
                f.write(chunk)
        tmp.replace(path)
        return path.stat().st_size

    def crawl_product(self, url: str):
        # IKEA's CDN occasionally serves a page whose hydrate data belongs to another
        # product/market (seen: a JP sofa URL carrying a Malaysian curtain). Validate
        # itemNo + market and refetch with a cache-buster.
        url_item = re.search(r"(\d{8})/?$", url).group(1)
        for attempt in range(3):
            html = self._get(url + (f"?cb={random.randint(1, 10**9)}" if attempt else "")).text
            data = self.parse_product(url, html)
            market_ok = f'"apiCountry":"{self.country}"' in html
            if data["item_no"] == url_item and market_ok:
                break
            log.warning("  page data mismatch (got item %s), refetching", data["item_no"])
            time.sleep(1 + attempt)
        else:
            raise ValueError("page data does not match URL after 3 attempts")
        if not data["model_url"]:
            log.info("  no 3D model -> skip")
            return None

        item = data["item_no"]
        glb = self.out / "models" / f"{item}.glb"
        size = self._download(data["model_url"], glb)
        with open(glb, "rb") as f:
            if f.read(4) != b"glTF":
                glb.unlink()
                raise ValueError("downloaded file is not a GLB")
        data["model_file"] = glb.relative_to(self.out).as_posix()
        data["model_bytes"] = size
        data["model_draco"] = "draco" in data["model_url"]

        if data["image_url"]:
            img = self.out / "images" / f"{item}.jpg"
            try:
                self._download(data["image_url"], img)
                data["image_file"] = img.relative_to(self.out).as_posix()
            except requests.RequestException as e:
                log.warning("  image failed: %s", e)

        (self.out / "products" / f"{item}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def _load_cached(self, item_no: str):
        p = self.out / "products" / f"{item_no}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if (self.out / d.get("model_file", "-")).exists():
                return d
        return None

    def _save(self, data: dict):
        (self.out / "products" / f"{data['item_no']}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_catalog(self) -> int:
        """Rebuild catalog.jsonl from every saved product that has a category and a model.
        Runs at the end of each crawl, so partial runs (`--only rug`) merge into one catalog
        and an interrupted crawl can be recovered with `--catalog-only`."""
        rows = []
        for p in sorted((self.out / "products").glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("crawl_category") and (self.out / d.get("model_file", "-")).exists():
                rows.append(d)
        order = {c: i for i, c in enumerate(CATEGORIES)}
        rows.sort(key=lambda d: (order.get(d["crawl_category"], 99), d["item_no"]))
        with open(self.out / "catalog.jsonl", "w", encoding="utf-8") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return len(rows)

    def run(self, per_category: int, categories: dict, seeds=(), max_per_name: int = 2):
        """Fill each category up to `per_category` products that have a 3D model.
        `max_per_name` limits colour variants of the same product line (e.g. 5 KIVIK covers)."""
        results, tried = [], Counter()
        # products rejected by pipeline/process_models.py --exclude (size mismatch, too heavy)
        excluded_path = self.out / "excluded.json"
        seen = set(json.loads(excluded_path.read_text(encoding="utf-8"))) if excluded_path.exists() else set()
        jobs = [("seed", None, u) for u in seeds]
        for cat, slugs in categories.items():
            for slug in slugs:
                try:
                    urls = self.category_product_urls(slug)
                except requests.RequestException as e:
                    log.error("category %s failed: %s", slug, e)
                    continue
                log.info("category %-18s %-40s %d links", cat, slug, len(urls))
                jobs += [(cat, slug, u) for u in urls]
                self._sleep()

        got, names = Counter(), Counter()
        for cat, slug, url in jobs:
            if cat != "seed" and got[cat] >= per_category:
                continue
            item_no = re.search(r"(\d{8})/?$", url).group(1)
            if item_no in seen:
                continue
            seen.add(item_no)
            if names[(cat, url.split("/p/")[1].split("-")[0])] >= max_per_name:
                continue

            data = self._load_cached(item_no)
            if data is None:
                tried[cat] += 1
                log.info("[%s %d/%d] %s", cat, got[cat] + 1, per_category, url)
                try:
                    data = self.crawl_product(url)
                except Exception as e:  # keep crawling on single-product failure
                    log.error("  failed: %s", e)
                    data = None
                self._sleep()
                if not data:
                    continue
                log.info("  OK %s %s %s", data["name"], data["type_name"], data["dimensions_cm"])
            data["crawl_category"] = cat
            data["crawl_category_slug"] = slug
            self._save(data)
            results.append(data)
            got[cat] += 1
            names[(cat, url.split("/p/")[1].split("-")[0])] += 1

        total = self.build_catalog()
        log.info("Done: %d products this run, %d in catalog -> %s", len(results), total, self.out.resolve())
        for cat in categories:
            log.info("  %-18s %3d  (new pages fetched: %d)", cat, got[cat], tried[cat])
        return results


def main():
    ap = argparse.ArgumentParser(description="Crawl IKEA products with 3D models")
    ap.add_argument("--per-category", type=int, default=25, help="products with 3D models per category")
    ap.add_argument("--only", default="", help="comma-separated subset of categories, e.g. sofa,rug")
    ap.add_argument("--max-per-name", type=int, default=2, help="max colour variants per product line per category")
    ap.add_argument("--out", default="data")
    ap.add_argument("--country", default="jp")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between page requests")
    ap.add_argument("--url", action="append", default=[], help="extra seed product URL (repeatable)")
    ap.add_argument("--catalog-only", action="store_true", help="only rebuild catalog.jsonl from saved products")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    crawler = IkeaCrawler(Path(args.out), args.country, args.lang, args.delay)
    if args.catalog_only:
        log.info("catalog.jsonl rebuilt: %d products", crawler.build_catalog())
        return
    only = [c.strip() for c in args.only.split(",") if c.strip()]
    cats = {c: CATEGORIES[c] for c in (only or CATEGORIES)}
    crawler.run(args.per_category, cats, args.url, args.max_per_name)


if __name__ == "__main__":
    main()

"""
Visualise the embedding index built by pipeline/index.py (step 4).

Figures (PNG, written to data/index/viz/):
  1_vectors.png              what a product vector looks like (384 numbers as a colour strip)
  2_map.png                  all products projected to 2D with t-SNE, one panel per furniture group
  3_category_similarity.png  cosine similarity between category centres
  4_query.png                one real query: ranking without vs with hard filters
Numbers used in embedding.md are written to data/index/viz/stats.json.

Usage: python pipeline/visualize_index.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from search import ProductIndex  # noqa: E402

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES1, BACKGROUND_DOT = "#2a78d6", "#d6d5cf"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a09f", "#b73332"])

GROUPS = {  # <= 1 highlight colour per panel (small multiples instead of 18 hues)
    "Ghế & sofa": ["sofa", "armchair", "ottoman_pouffe", "dining_chair", "desk_chair"],
    "Bàn": ["coffee_side_table", "dining_table", "desk"],
    "Giường & tủ đầu giường": ["bed", "bedside_table"],
    "Tủ & kệ": ["wardrobe", "chest_of_drawers", "bookcase_shelf", "sideboard_cabinet", "tv_bench"],
    "Đèn": ["floor_lamp", "table_lamp"],
    "Thảm": ["rug"],
}
CAT_VI = {
    "sofa": "sofa", "armchair": "ghế bành", "ottoman_pouffe": "đôn", "dining_chair": "ghế ăn",
    "desk_chair": "ghế văn phòng", "coffee_side_table": "bàn trà", "dining_table": "bàn ăn",
    "desk": "bàn làm việc", "bed": "giường", "bedside_table": "tủ đầu giường", "wardrobe": "tủ quần áo",
    "chest_of_drawers": "tủ ngăn kéo", "bookcase_shelf": "kệ sách", "sideboard_cabinet": "tủ thấp",
    "tv_bench": "kệ TV", "floor_lamp": "đèn cây", "table_lamp": "đèn bàn", "rug": "thảm",
}
QUERY = "ghế đọc sách nhỏ gọn"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def fig_vectors(ix, out, stats):
    """Three products as colour strips: two similar sofas and one lamp."""
    rows = ix.rows
    by_id = {r["item_no"]: i for i, r in enumerate(rows)}
    picks = ["10295025", "70434368", "00110904"]  # HEMLINGBY sofa x2, LERSTA floor lamp
    picks = [p for p in picks if p in by_id]
    vecs = ix.emb[[by_id[p] for p in picks]]
    lim = float(np.abs(vecs).max())
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 0.9 * len(picks) + 1.2), facecolor=SURFACE)
    for ax, p, v in zip(axes, picks, vecs):
        r = rows[by_id[p]]
        ax.imshow(v[None, :], aspect="auto", cmap=DIVERGING, vmin=-lim, vmax=lim)
        ax.set_yticks([])
        ax.set_xticks([0, 127, 255, 383])
        ax.set_xticklabels(["chiều 1", "128", "256", "384"])
        ax.set_ylabel(f"{r['name']}\n{r['type_name']}", rotation=0, ha="right", va="center", color=INK, fontsize=9)
        style(ax)
    sims = vecs @ vecs.T
    stats["vector_example"] = {
        "items": [rows[by_id[p]]["name"] + " " + rows[by_id[p]]["type_name"] for p in picks],
        "cosine_sofa_sofa": round(float(sims[0, 1]), 3),
        "cosine_sofa_lamp": round(float(sims[0, 2]), 3),
        "first_8_values": [round(float(x), 4) for x in vecs[0][:8]],
        "norm": round(float(np.linalg.norm(vecs[0])), 4),
    }
    fig.suptitle(f"Mỗi sản phẩm = 384 con số (xanh = âm, đỏ = dương).  "
                 f"Cosine sofa–sofa = {sims[0, 1]:.3f}, sofa–đèn = {sims[0, 2]:.3f}",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "1_vectors.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def fig_map(ix, out, stats):
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, metric="cosine", perplexity=25, init="pca", random_state=0).fit_transform(ix.emb)
    cats = np.array([r["category"] for r in ix.rows])

    # nearest-neighbour agreement: how often a product's closest other product shares its category
    sim = ix.emb @ ix.emb.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    stats["nn_same_category"] = round(float((cats[nn] == cats).mean()), 3)
    per_cat = {c: round(float((cats[nn] == cats)[cats == c].mean()), 2) for c in sorted(set(cats))}
    stats["nn_same_category_per_cat"] = per_cat

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), facecolor=SURFACE)
    for ax, (gname, gcats) in zip(axes.flat, GROUPS.items()):
        on = np.isin(cats, gcats)
        ax.scatter(xy[~on, 0], xy[~on, 1], s=10, c=BACKGROUND_DOT, linewidths=0)
        ax.scatter(xy[on, 0], xy[on, 1], s=22, c=SERIES1, edgecolors=SURFACE, linewidths=0.8)
        for c in gcats:  # direct label at each category's centre
            m = cats == c
            if m.any():
                cx, cy = np.median(xy[m], axis=0)
                ax.annotate(CAT_VI[c], (cx, cy), color=INK, fontsize=9, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
        ax.set_title(f"{gname}  ({int(on.sum())} sản phẩm)", color=INK, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        style(ax)
    fig.suptitle("Bản đồ 388 sản phẩm: vector 384 chiều được t-SNE ép xuống 2D. "
                 "Hai điểm gần nhau = hai sản phẩm có văn bản mô tả giống nghĩa.\n"
                 f"Điểm xanh = nhóm của khung đó, điểm xám = sản phẩm khác. "
                 f"Sản phẩm gần nhất cùng danh mục: {stats['nn_same_category']:.0%}",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "2_map.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def fig_category_similarity(ix, out, stats):
    order = [c for g in GROUPS.values() for c in g]
    idx = defaultdict(list)
    for i, r in enumerate(ix.rows):
        idx[r["category"]].append(i)
    cent = np.stack([ix.emb[idx[c]].mean(0) for c in order])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    S = cent @ cent.T
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    i, j = np.unravel_index(np.nanargmax(off), off.shape)
    k, l = np.unravel_index(np.nanargmin(off), off.shape)
    stats["closest_categories"] = [order[i], order[j], round(float(S[i, j]), 3)]
    stats["farthest_categories"] = [order[k], order[l], round(float(S[k, l]), 3)]

    fig, ax = plt.subplots(figsize=(8.6, 7.4), facecolor=SURFACE)
    im = ax.imshow(off, cmap=SEQ_BLUE, vmin=np.nanmin(off), vmax=np.nanmax(off))
    labels = [CAT_VI[c] for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=60, ha="right", color=INK2, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, color=INK2, fontsize=8)
    for (a, b) in [(i, j), (k, l)]:  # label only the extremes
        ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center", fontsize=7,
                color=SURFACE if (a, b) == (i, j) else INK)
    style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("cosine giữa tâm 2 danh mục", color=INK2, fontsize=8)
    ax.set_title("Danh mục nào 'gần nghĩa' nhau? (đậm = gần; đường chéo bỏ trống)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "3_category_similarity.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def fig_query(ix, out, stats):
    k = 8
    plain = ix.search(QUERY, k=k)
    filt = ix.search(QUERY, k=k, category={"armchair"}, max_price=20000)
    stats["query"] = {
        "text": QUERY,
        "plain": [(r["name"], r["type_name"], r["category"], r["price"], round(r["score"], 3)) for r in plain],
        "filtered": [(r["name"], r["type_name"], r["category"], r["price"], round(r["score"], 3)) for r in filt],
    }
    lo = min(r["score"] for r in plain + filt) - 0.01
    hi = max(r["score"] for r in plain + filt) + 0.005
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, sharex=True)
    for ax, res, title in [
        (axes[0], plain, "Chỉ tìm theo nghĩa (không lọc)"),
        (axes[1], filt, "Lọc cứng trước (ghế bành, ≤ ¥20.000), rồi xếp hạng"),
    ]:
        style(ax)
        ys = np.arange(len(res))[::-1]
        ax.hlines(ys, lo, [r["score"] for r in res], color=GRID, linewidth=1)
        ax.scatter([r["score"] for r in res], ys, s=60, c=SERIES1, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['name']} – {r['type_name'][:24]}  ({CAT_VI[r['category']]}, ¥{r['price']:,})"
                            for r in res], fontsize=8, color=INK)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("cosine với câu hỏi (cao = giống nghĩa hơn)", color=INK2, fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.suptitle(f"Câu hỏi: \"{QUERY}\"  →  top {k}", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "4_query.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ix = ProductIndex(Path(__file__).resolve().parent.parent / "data")
    out = Path(__file__).resolve().parent.parent / "data" / "index" / "viz"
    out.mkdir(exist_ok=True)
    stats = {"model": ix.meta["model"], "dim": ix.meta["dim"], "count": ix.meta["count"]}

    # one worked example for the doc: text -> tokens -> vector
    ix.embed_query("warm-up")
    r0 = next(r for r in ix.rows if r["item_no"] == "10295025")  # HEMLINGBY sofa
    tok = ix._model.tokenizer
    ids = tok("passage: " + r0["doc"])["input_ids"]
    stats["doc_example"] = {"item_no": r0["item_no"], "doc": r0["doc"], "n_tokens": len(ids),
                            "first_tokens": tok.convert_ids_to_tokens(ids[:16]),
                            "max_seq_length": ix._model.max_seq_length}
    stats["doc_tokens"] = {"mean": int(np.mean([len(tok("passage: " + r["doc"])["input_ids"]) for r in ix.rows])),
                           "max": int(np.max([len(tok("passage: " + r["doc"])["input_ids"]) for r in ix.rows]))}

    fig_vectors(ix, out, stats)
    fig_map(ix, out, stats)
    fig_category_similarity(ix, out, stats)
    fig_query(ix, out, stats)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "doc_example"}, ensure_ascii=False, indent=1)[:3000])
    print("doc tokens:", stats["doc_example"]["n_tokens"], stats["doc_example"]["first_tokens"])


if __name__ == "__main__":
    main()

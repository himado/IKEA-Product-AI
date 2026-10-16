"""
Evaluate step 6 on eval/intent_cases.json: Gemini parser vs. the rule-based fallback.

A case passes when ALL hold:
  items   predicted categories match the expected items one-to-one (no missing, no extra)
  qty     every quantity listed in the case is right
  budget  budget converted to JPY equals the expected one (same fixed rates)
  clarify a clarification question is asked iff the case expects one

Output: eval/results/intent_eval.md (+ per-case JSON) and a console summary.
Usage:  python backend/intent/eval_intent.py [--rules-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse import ROOT, TO_CATALOG, parse_intent, rule_based, to_jpy  # noqa: E402

TO_LLM = {v: k for k, v in TO_CATALOG.items()}


def match(pred_items, expect):
    """Greedy one-to-one match of predicted categories to expected acceptable-sets."""
    pred = list(pred_items)
    for group in sorted(expect, key=len):           # most specific groups first
        hit = next((p for p in pred if p["category"] in group), None)
        if hit is None:
            return False, f"missing {'/'.join(group)}"
        pred.remove(hit)
    if pred:
        return False, "extra " + ", ".join(p["category"] for p in pred)
    return True, ""


def judge(case, items, budget_jpy, clarified):
    errs = []
    want_clarify = case.get("clarify", False)
    if clarified != want_clarify:
        errs.append("should ask a question" if want_clarify else "asked a question instead of answering")
    ok, why = match(items, case["expect"])
    if not ok and not want_clarify:
        errs.append(why)
    for cat, q in (case.get("qty") or {}).items():
        got = next((p["qty"] for p in items if p["category"] == cat), None)
        if got != q:
            errs.append(f"qty {cat}={got}, want {q}")
    want_budget = to_jpy(*case["budget"]) if case.get("budget") else None
    if want_budget != budget_jpy:
        errs.append(f"budget ¥{budget_jpy}, want ¥{want_budget}")
    return errs


def run_llm(case):
    t0 = time.time()
    res = parse_intent(case["text"])
    items = [{"category": TO_LLM.get(i.category, i.category), "qty": i.qty} for i in (res.intent.items if res.intent else [])]
    return items, (res.intent.budget if res.intent else None), res.clarification_question is not None, \
        res.source, round(time.time() - t0, 1), res.model_dump()


def run_rules(case):
    raw = rule_based(case["text"])
    items = [{"category": i.category, "qty": i.qty} for i in raw.items]
    return items, to_jpy(raw.budget_amount, raw.budget_currency), not raw.items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()
    cases = json.loads((ROOT / "eval" / "intent_cases.json").read_text(encoding="utf-8"))["cases"]
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)

    rows, n_llm, n_rules, per_case = [], 0, 0, []
    for k, c in enumerate(cases, 1):
        r_items, r_budget, r_clar = run_rules(c)
        r_errs = judge(c, r_items, r_budget, r_clar)
        n_rules += not r_errs
        if args.rules_only:
            l_errs, l_items, src, secs = ["skipped"], [], "-", 0
        else:
            l_items, l_budget, l_clar, src, secs, dump = run_llm(c)
            l_errs = judge(c, l_items, l_budget, l_clar)
            n_llm += not l_errs
            per_case.append({"text": c["text"], "result": dump, "errors": l_errs})
        fmt = lambda items: ", ".join(f"{i['category']}×{i['qty']}" if i["qty"] > 1 else i["category"] for i in items) or "(hỏi lại)"
        print(f"[{k:02}] {'OK ' if not l_errs else 'ERR'} {src:<17} {secs:>5}s  {c['text'][:52]:<52} -> {fmt(l_items)} {l_errs or ''}"
              f" | rules {'OK' if not r_errs else r_errs}", flush=True)
        rows.append(f"| {k} | {c['text']} | {fmt(l_items)} | {'✅' if not l_errs else '❌ ' + '; '.join(l_errs)} | {src} | "
                    f"{fmt(r_items)} | {'✅' if not r_errs else '❌ ' + '; '.join(r_errs)} |")

    n = len(cases)
    summary = f"Gemini: {n_llm}/{n} = {n_llm / n:.0%}   ·   Luật dự phòng: {n_rules}/{n} = {n_rules / n:.0%}"
    md = ["# Đánh giá bước 6 – hiểu yêu cầu", "", summary, "",
          "| # | Câu | Gemini → | Gemini | nguồn | Luật → | Luật |", "|---|---|---|---|---|---|---|"] + rows
    (out / "intent_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if per_case:
        (out / "intent_eval.json").write_text(json.dumps(per_case, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()

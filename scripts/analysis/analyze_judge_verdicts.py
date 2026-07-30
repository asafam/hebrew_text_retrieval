#!/usr/bin/env python3
"""Compare LLM judge verdicts between Hebrew-only failures and blind controls.

The judge scored both groups without knowing which was which. The question is not
"do the failures have translation problems" — some always will — but whether the
rate is HIGHER than among queries Hebrew answered correctly. Only an elevated rate
implicates translation.

Reports, per metric: rate in each group, the difference, a two-proportion z-test,
and the risk ratio. Also breaks down by dataset and estimates how many of the ~630
failures are plausibly translation-caused.
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, fisher_exact


def two_prop(a_hit, a_n, b_hit, b_n):
    """Two-proportion z-test. Returns (p_a, p_b, diff, z, p_value)."""
    if a_n == 0 or b_n == 0:
        return 0, 0, 0, float("nan"), float("nan")
    p1, p2 = a_hit / a_n, b_hit / b_n
    p = (a_hit + b_hit) / (a_n + b_n)
    se = (p * (1 - p) * (1 / a_n + 1 / b_n)) ** 0.5
    if se == 0:
        return p1, p2, p1 - p2, float("nan"), 1.0
    z = (p1 - p2) / se
    return p1, p2, p1 - p2, z, 2 * (1 - norm.cdf(abs(z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="outputs/analysis/judge/verdicts.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.verdicts)]
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    loss = [r for r in rows if r["group"] == "loss"]
    ctrl = [r for r in rows if r["group"] == "control"]

    print("=" * 96)
    print("LLM JUDGE: Hebrew-only failures vs blind controls")
    print("=" * 96)
    print(f"judged {len(rows)} items ({len(loss)} failures, {len(ctrl)} controls)"
          + (f"; {len(errs)} errors dropped" if errs else ""))
    print("\nThe judge never saw which group an item was in, nor any retrieval outcome.\n")

    METRICS = [
        ("query translation not faithful",
         lambda r: r["query_translation"] != "faithful"),
        ("  ...specifically 'ambiguous'",
         lambda r: r["query_translation"] == "ambiguous"),
        ("  ...specifically 'wrong'",
         lambda r: r["query_translation"] == "wrong"),
        ("doc translation not faithful",
         lambda r: r["doc_translation"] != "faithful"),
        ("key term lost",
         lambda r: bool(r["key_term_lost"])),
        ("retrieval risk high",
         lambda r: r["retrieval_risk"] == "high"),
        ("retrieval risk low or high",
         lambda r: r["retrieval_risk"] in ("low", "high")),
        ("ANY translation fault (query or doc not faithful, or key term lost)",
         lambda r: r["query_translation"] != "faithful"
                   or r["doc_translation"] != "faithful" or bool(r["key_term_lost"])),
        ("--- answer-key quality (judged from English, should NOT differ) ---",
         None),
        ("pair not clearly relevant (loose or unrelated)",
         lambda r: r["pair_relevance"] in ("loose", "unrelated")),
        ("pair outright unrelated",
         lambda r: r["pair_relevance"] == "unrelated"),
    ]

    print(f"{'metric':<62} {'fail':>7} {'ctrl':>7} {'diff':>8} {'RR':>6} {'p':>10}")
    print("-" * 96)
    results = {}
    for name, fn in METRICS:
        if fn is None:
            print(f"\n{name}")
            continue
        lh, ch = sum(map(fn, loss)), sum(map(fn, ctrl))
        p1, p2, diff, z, pv = two_prop(lh, len(loss), ch, len(ctrl))
        rr = (p1 / p2) if p2 > 0 else float("inf")
        star = "  <--" if pv < 0.01 else ("  ~" if pv < 0.05 else "")
        results[name] = (p1, p2, diff, pv)
        print(f"{name:<62} {100*p1:>6.1f}% {100*p2:>6.1f}% {100*diff:>+7.1f} "
              f"{rr:>6.2f} {pv:>10.2e}{star}")

    # Per-dataset: any translation fault
    print("\n" + "=" * 96)
    print("ANY translation fault, by dataset")
    print("-" * 96)
    fault = lambda r: (r["query_translation"] != "faithful"
                       or r["doc_translation"] != "faithful" or bool(r["key_term_lost"]))
    print(f"{'dataset':<18} {'n_fail':>7} {'fault%':>8} {'n_ctrl':>7} {'fault%':>8} {'diff':>8} {'p':>10}")
    for ds in sorted({r["dataset"] for r in rows}):
        L = [r for r in loss if r["dataset"] == ds]
        C = [r for r in ctrl if r["dataset"] == ds]
        p1, p2, diff, z, pv = two_prop(sum(map(fault, L)), len(L), sum(map(fault, C)), len(C))
        print(f"{ds:<18} {len(L):>7} {100*p1:>7.1f}% {len(C):>7} {100*p2:>7.1f}% "
              f"{100*diff:>+7.1f} {pv:>10.2e}")

    # Attributable fraction
    print("\n" + "=" * 96)
    print("HOW MANY FAILURES ARE PLAUSIBLY TRANSLATION-CAUSED?")
    print("-" * 96)
    lf = sum(map(fault, loss)) / len(loss)
    cf = sum(map(fault, ctrl)) / len(ctrl)
    excess = max(0.0, lf - cf)
    print(f"  translation fault rate among failures : {100*lf:.1f}%")
    print(f"  same rate among controls (background) : {100*cf:.1f}%")
    print(f"  excess attributable to translation    : {100*excess:.1f} pts "
          f"-> ~{excess*len(loss):.0f} of {len(loss)} judged failures")
    hi = lambda r: r["retrieval_risk"] == "high"
    lh, ch = sum(map(hi, loss))/len(loss), sum(map(hi, ctrl))/len(ctrl)
    print(f"\n  judge-rated HIGH retrieval risk       : {100*lh:.1f}% vs {100*ch:.1f}% control "
          f"-> excess {100*max(0,lh-ch):.1f} pts (~{max(0,lh-ch)*len(loss):.0f} queries)")
    bad = lambda r: r["pair_relevance"] in ("loose", "unrelated")
    print(f"\n  failures whose answer key is loose/unrelated: {100*sum(map(bad,loss))/len(loss):.1f}% "
          f"(control {100*sum(map(bad,ctrl))/len(ctrl):.1f}%) — these are qrels noise, not translation")

    # Examples the judge flagged hardest
    print("\n" + "=" * 96)
    print("SAMPLE: failures the judge rated high retrieval risk")
    print("-" * 96)
    shown = 0
    for r in loss:
        if r["retrieval_risk"] == "high" and shown < 8:
            print(f"  [{r['dataset'].replace('BeIR_','')}] q={r['query_translation']} "
                  f"doc={r['doc_translation']} key_term_lost={r['key_term_lost']}")
            print(f"     {r['note'][:150]}")
            shown += 1
    if shown == 0:
        print("  (none)")


if __name__ == "__main__":
    main()

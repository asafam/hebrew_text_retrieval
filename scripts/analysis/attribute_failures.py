#!/usr/bin/env python3
"""Attribute Hebrew retrieval failures to translation vs. retrieval.

Inputs (produced by the two sibling scripts):
  outputs/analysis/per_query/<dataset>.jsonl   he_rank / en_rank per query
  outputs/analysis/defects/<dataset>_*.jsonl   model-free translation defect signals

Method
------
Same model, same items, two languages, so each query falls in one cell of:

                     English hit@10
                        yes      no
    Hebrew    yes    concordant  He-only-win
    hit@10     no    He-only-LOSS  both-fail

  both-fail     -> the retriever misses it in English too. Not a translation
                   problem: either an intrinsically hard query or noisy qrels.
  He-only-loss  -> candidate translation-attributable failure.
  He-only-win   -> the reverse discordance. Crucially this is the NOISE FLOOR:
                   rank jitter around the top-10 boundary produces both
                   directions roughly equally. Only the EXCESS of He-only-loss
                   over He-only-win is evidence of systematic Hebrew-side loss.

Two confounds are tested rather than assumed away:

 1. mE5 is simply stronger in English than Hebrew, independent of translation
    quality. If that were the whole story, He-only-loss queries would show no
    more translation defects than concordant ones. So each defect signal is
    compared between those two groups (Mann-Whitney U + rank-biserial effect
    size). Elevated defects => translation is implicated; flat => the gap is
    model language capability, not the translation.

 2. Discordance may be pure boundary noise, addressed by the He-only-win floor
    above.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

HE_ROOT = ("outputs/translation/runs/"
           "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
SIGNALS = ["len_ratio", "latin_residue", "hebrew_frac", "digit_jaccard"]
DATASETS = ["BeIR_arguana", "BeIR_fiqa", "BeIR_nfcorpus", "BeIR_scidocs", "BeIR_scifact"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def load_qrels(ds):
    qrels = defaultdict(dict)
    qdir = Path(HE_ROOT) / ds / "beir" / "qrels"
    for f in sorted(qdir.iterdir()):
        if f.name.startswith(("test", "dev", "validation")):
            for l in open(f):
                r = json.loads(l)
                qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
            break
    return qrels


def rank_biserial(a, b):
    """Effect size for Mann-Whitney: 0 = no difference, +-1 = complete separation."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    try:
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
    except ValueError:
        return float("nan")
    return 2 * u / (len(a) * len(b)) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_query_dir", default="outputs/analysis/per_query")
    ap.add_argument("--defect_dir", default="outputs/analysis/defects")
    ap.add_argument("--k", type=int, default=10, help="hit@k threshold defining success")
    a = ap.parse_args()

    print("=" * 100)
    print("FAILURE ATTRIBUTION: Hebrew vs English, same model (mE5-base), same items")
    print("=" * 100)

    grand = defaultdict(int)
    per_ds_rows = []

    for ds in DATASETS:
        pq_path = Path(a.per_query_dir) / f"{ds}.jsonl"
        if not pq_path.exists():
            print(f"\n{ds}: missing {pq_path} — skipped")
            continue
        rows = load_jsonl(pq_path)
        qd = {r["_id"]: r for r in load_jsonl(Path(a.defect_dir) / f"{ds}_queries.jsonl")}
        cd = {r["_id"]: r for r in load_jsonl(Path(a.defect_dir) / f"{ds}_corpus.jsonl")}
        qrels = load_qrels(ds)

        cells = defaultdict(list)
        for r in rows:
            he = bool(r["he_rank"] and r["he_rank"] <= a.k)
            en = bool(r["en_rank"] and r["en_rank"] <= a.k)
            cells[(he, en)].append(r)

        n = len(rows)
        both_ok = len(cells[(True, True)])
        he_loss = len(cells[(False, True)])
        he_win = len(cells[(True, False)])
        both_fail = len(cells[(False, False)])
        he_fail = he_loss + both_fail

        print(f"\n{'='*100}\n{ds}   ({n:,} queries with judgments)")
        print(f"  Hebrew hit@{a.k}: {100*(both_ok+he_win)/n:5.1f}%     "
              f"English hit@{a.k}: {100*(both_ok+he_loss)/n:5.1f}%     "
              f"gap: {100*(he_loss-he_win)/n:+.1f} pts")
        print(f"  concordant success {both_ok:>6}   both fail {both_fail:>6}   "
              f"He-only-loss {he_loss:>5}   He-only-win {he_win:>5}  (noise floor)")
        if he_fail:
            excess = max(0, he_loss - he_win)
            print(f"  Of {he_fail} Hebrew failures: "
                  f"{100*both_fail/he_fail:.1f}% also fail in English (not translation), "
                  f"{100*he_loss/he_fail:.1f}% English-recoverable, of which "
                  f"{100*excess/he_fail:.1f} pts survive the noise floor.")

        for k_, v in [("both_ok", both_ok), ("he_loss", he_loss),
                      ("he_win", he_win), ("both_fail", both_fail), ("n", n)]:
            grand[k_] += v

        # Do English-recoverable failures have measurably worse translations?
        loss, ok = cells[(False, True)], cells[(True, True)]
        if len(loss) >= 20 and len(ok) >= 20:
            print(f"\n  Translation defects: He-only-loss (n={len(loss)}) vs concordant "
                  f"success (n={len(ok)})")
            print(f"    {'signal':<24} {'loss med':>10} {'ok med':>10} {'effect':>8} {'p':>10}")
            for side, store in (("query", qd), ("gold doc", cd)):
                for sig in SIGNALS:
                    def vals(group):
                        out = []
                        for r in group:
                            if side == "query":
                                d = store.get(r["qid"])
                                if d:
                                    out.append(d[sig])
                            else:
                                gold = [g for g, s in qrels.get(r["qid"], {}).items() if s > 0]
                                v = [store[g][sig] for g in gold if g in store]
                                if v:
                                    # worst gold doc: the defect that could plausibly
                                    # have broken this query's best chance
                                    out.append(min(v) if sig != "latin_residue" else max(v))
                        return np.asarray(out, dtype=float)
                    A, B = vals(loss), vals(ok)
                    if len(A) < 20 or len(B) < 20:
                        continue
                    try:
                        p = mannwhitneyu(A, B, alternative="two-sided").pvalue
                    except ValueError:
                        continue
                    eff = rank_biserial(A, B)
                    star = "  <-- " if (p < 0.01 and abs(eff) > 0.1) else ""
                    print(f"    {side+' '+sig:<24} {np.median(A):>10.3f} {np.median(B):>10.3f} "
                          f"{eff:>+8.3f} {p:>10.2e}{star}")

        per_ds_rows.append((ds, n, both_ok, he_loss, he_win, both_fail))

    if grand["n"]:
        n = grand["n"]; hf = grand["he_loss"] + grand["both_fail"]
        excess = max(0, grand["he_loss"] - grand["he_win"])
        print(f"\n{'='*100}\nOVERALL ({n:,} queries across {len(per_ds_rows)} datasets)")
        print(f"  Hebrew hit@{a.k} {100*(grand['both_ok']+grand['he_win'])/n:.1f}%  vs  "
              f"English {100*(grand['both_ok']+grand['he_loss'])/n:.1f}%")
        print(f"  Hebrew failures: {hf:,}")
        print(f"    {100*grand['both_fail']/hf:5.1f}%  fail in English too      -> retrieval / data, NOT translation")
        print(f"    {100*grand['he_loss']/hf:5.1f}%  recoverable in English    -> upper bound on language+translation")
        print(f"    {100*excess/hf:5.1f}%  net of the reverse-discordance noise floor "
              f"({grand['he_win']:,} He-only-wins)")


if __name__ == "__main__":
    main()

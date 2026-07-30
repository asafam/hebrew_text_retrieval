#!/usr/bin/env python3
"""Recompute BeIR metrics from cached embeddings, without reloading any model.

The NDCG/self-exclusion fixes in eval_beir_retrieval_zeroshot.py changed only the
metric layer — embeddings, FAISS search and ranking are untouched. So every
existing result under outputs/eval/beir_zeroshot/ can be corrected by replaying
the cached query/doc embeddings through the fixed compute_metrics, which avoids
re-encoding ~57K documents per model on a GPU.

All retrieval logic is imported from the eval script rather than reimplemented,
so the two cannot drift.

Verification: with --verify, each directory is first rescored under pre-fix
semantics (exclude_self=never) and checked against the stored metrics. The old
`recall_at_100` was really a hit rate, so it must equal the new
`hit_rate_at_100`, and `mrr` must match exactly. If both reproduce, the
embedding ordering and normalization have been replayed faithfully and any NDCG
delta is attributable to the fix alone.

Usage:
    python scripts/model/eval/rescore_beir_results.py --verify --dry_run
    python scripts/model/eval/rescore_beir_results.py --verify
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO / "src" / "model" / "eval" / "eval_beir_retrieval_zeroshot.py"


def _load_eval_module():
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("_beir_eval", EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_corpus_dirs(runs_root="outputs/translation/runs"):
    """Map dataset name (e.g. BeIR_nfcorpus) -> exported BEIR directory."""
    out = {}
    for path in Path(runs_root).rglob("beir/corpus.jsonl"):
        d = path.parent
        out[d.parent.name] = str(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="outputs/eval/beir_zeroshot")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--exclude_self", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--ndcg_gain", default="linear", choices=["linear", "exponential"])
    ap.add_argument("--verify", action="store_true",
                    help="Reproduce pre-fix metrics first and assert they match the stored ones.")
    ap.add_argument("--dry_run", action="store_true", help="Do not write results.json.")
    args = ap.parse_args()

    ev = _load_eval_module()
    corpus_dirs = find_corpus_dirs()
    print(f"Found {len(corpus_dirs)} exported BEIR corpora: {', '.join(sorted(corpus_dirs))}\n")

    # Cache corpus/qrels per dataset — 24 models share the same 5 corpora.
    loaded = {}

    rows, skipped, verify_fail = [], [], []
    root = Path(args.results_dir)
    for res_path in sorted(root.glob("*/*/results.json")):
        d = res_path.parent
        model_label, dataset = d.parent.name, d.name
        q_cache, d_cache = d / "query_embeddings.pt", d / "doc_embeddings.pt"

        if dataset not in corpus_dirs:
            skipped.append((model_label, dataset, "no matching corpus export"))
            continue
        if not (q_cache.exists() and d_cache.exists()):
            skipped.append((model_label, dataset, "missing embedding cache"))
            continue

        old = json.loads(res_path.read_text())
        # The pre-fix baseline is captured once and never overwritten. Re-running
        # this script must not promote an already-rescored `metrics` block into
        # `metrics_pre_fix` — that would silently destroy the original numbers.
        already_rescored = "metrics_pre_fix" in old
        old_metrics = old["metrics_pre_fix"] if already_rescored else old.get("metrics", {})
        model_path = old.get("config", {}).get("model_path")

        if dataset not in loaded:
            loaded[dataset] = ev.load_beir_local(corpus_dirs[dataset])
        corpus, queries, qrels = loaded[dataset]
        doc_ids, query_ids = list(corpus.keys()), list(queries.keys())

        q_emb = torch.load(q_cache, weights_only=False)
        d_emb = torch.load(d_cache, weights_only=False)
        if isinstance(q_emb, list):
            q_emb = torch.cat([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in q_emb])
        if isinstance(d_emb, list):
            d_emb = torch.cat([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in d_emb])

        if q_emb.shape[0] != len(query_ids) or d_emb.shape[0] != len(doc_ids):
            skipped.append((model_label, dataset,
                            f"shape mismatch: emb {tuple(q_emb.shape)}/{tuple(d_emb.shape)} "
                            f"vs data {len(query_ids)}/{len(doc_ids)}"))
            continue

        # Replicate main()'s normalization branch: InfoNCE caches are stored
        # unnormalized and normalized after load; ST caches are used as-is.
        is_infonce = bool(model_path) and os.path.isdir(model_path) and ev._is_infonce(model_path)
        if is_infonce:
            q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
            d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

        search_k = min(args.top_k + 1, len(doc_ids)) if args.exclude_self != "never" else args.top_k
        scores, indices = ev.faiss_search(q_emb, d_emb, search_k)

        # Verification compares a pre-fix replay against the ORIGINAL stored
        # metrics, so it is meaningful only on a first pass. Once `metrics` holds
        # rescored values there is nothing pre-fix left in it to compare against.
        if args.verify and old_metrics and not already_rescored:
            pre = ev.compute_metrics(scores, indices, doc_ids, qrels, query_ids,
                                     top_k=args.top_k, exclude_self="never", gain="linear")
            # Two tolerances, for two different kinds of check:
            #   hit_rate must reproduce EXACTLY — it depends only on which docs
            #   land in the top-k, so any drift means the embeddings, ordering or
            #   normalization were not replayed faithfully.
            #   mrr depends on the *order* within the top-k, so it is sensitive to
            #   float-level differences in the dot products between the machine /
            #   FAISS build that produced the stored numbers and this one. Observed
            #   drift is ~1e-6..2e-5 (one query moving one rank) on 3 of 122 dirs,
            #   with hit_rate still exact and the replay matching a brute-force
            #   numpy argsort exactly. 1e-4 flags real breakage, not that noise.
            checks = []
            if "mrr" in old_metrics:
                checks.append(("mrr", abs(pre["mrr"] - old_metrics["mrr"]), 1e-4))
            if "recall_at_100" in old_metrics:
                checks.append(("hit_rate==old_recall",
                               abs(pre["hit_rate_at_100"] - old_metrics["recall_at_100"]), 1e-9))
            bad = [(n, v) for n, v, tol in checks if v > tol]
            if bad:
                verify_fail.append((model_label, dataset, bad))

        new = ev.compute_metrics(scores, indices, doc_ids, qrels, query_ids,
                                 top_k=args.top_k, exclude_self=args.exclude_self,
                                 gain=args.ndcg_gain)

        rows.append((model_label, dataset, old_metrics, new))

        if not args.dry_run:
            out = dict(old)
            out["metrics"] = new
            out["metrics_pre_fix"] = old_metrics
            cfg = dict(out.get("config", {}))
            cfg.update({"exclude_self": args.exclude_self, "ndcg_gain": args.ndcg_gain,
                        "rescored_from_cached_embeddings": True})
            out["config"] = cfg
            res_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n{'='*118}")
    print(f"{'model':<40} {'dataset':<16} {'NDCG@10 old→new':>24} {'MRR old→new':>20} {'recall@100':>14}")
    print("-" * 118)
    for model_label, dataset, old_m, new in sorted(rows):
        o10, n10 = old_m.get("ndcg_at_10", float("nan")), new["ndcg_at_10"]
        om, nm = old_m.get("mrr", float("nan")), new["mrr"]
        print(f"{model_label:<40} {dataset:<16} "
              f"{o10:>10.4f} → {n10:<10.4f} {om:>8.4f} → {nm:<8.4f} {new['recall_at_100']:>14.4f}")

    print(f"\nRescored {len(rows)} result files.")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for m, ds, why in skipped:
            print(f"   {m}/{ds}: {why}")
    if args.verify:
        if verify_fail:
            print(f"\n!! VERIFICATION FAILED for {len(verify_fail)} dirs "
                  f"(pre-fix replay did not match stored metrics):")
            for m, ds, bad in verify_fail:
                print(f"   {m}/{ds}: " + ", ".join(f"{n} Δ={v:.2e}" for n, v in bad))
        else:
            print("\nVerification passed: pre-fix replay reproduced stored MRR and "
                  "hit-rate for every directory.")
    return 1 if verify_fail else 0


if __name__ == "__main__":
    sys.exit(main())

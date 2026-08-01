"""
Evaluate retrieval on the welded long-context BeIR benchmark.

One run = one (dataset, condition, rung, position bin, arm). The corpus for a run is
``nongold.jsonl`` plus exactly **one** ``gold_{bin}.jsonl``, which together reproduce the
source corpus document-for-document -- a non-gold is never a query target, so its own
position cannot affect any metric and it needs only one variant.

Document *identity* is unchanged by welding, so ``queries.jsonl`` and ``qrels/test.jsonl``
are the untouched originals and results are directly comparable to the c0 (unpadded) gate.

Reuses the validated pieces of ``eval_beir_retrieval_zeroshot.py`` -- model loading with the
``tokenizer_query/`` fallback, prefix detection, cached encoding, FAISS search, and the
pytrec_eval-verified ``compute_metrics`` -- rather than reimplementing them, so a long-context
number differs from a c0 number only because the documents differ.

Usage::

    python src/model/eval/eval_longctx.py \\
        --benchmark_dir data/retrieval/beir_longctx/v1/BeIR_scifact \\
        --condition random --rung 27000 --position middle \\
        --model_name_or_path <path> --model_label HMB-native \\
        --strategy native --window 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.eval.eval_beir_retrieval_zeroshot import (  # noqa: E402
    compute_metrics,
    encode_all,
    faiss_search,
    get_prefixes,
    load_model,
)
from model.eval.longctx_encoding import (  # noqa: E402
    IncompatibleStrategy,
    aggregate_window_scores,
    model_native_limit,
    plan_windows,
    resolve_window,
)

POSITION_BINS = ("start", "middle", "end")


def load_welded_corpus(benchmark_dir: str, condition: str, rung: int, position: str):
    """Load one evaluable corpus: all non-golds plus one position variant of the golds.

    Returns ``(doc_ids, doc_texts, gold_spans)`` where ``gold_spans`` maps a document id to
    its ``(gold_char_start, gold_char_end)`` -- used to report whether the answering passage
    even falls inside a model's window.
    """
    rung_dir = Path(benchmark_dir) / condition / f"c{rung}"
    shards = [rung_dir / "nongold.jsonl", rung_dir / f"gold_{position}.jsonl"]

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    gold_spans: dict[str, tuple[int, int]] = {}

    for shard in shards:
        if not shard.exists():
            raise FileNotFoundError(f"missing shard: {shard}")
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                doc_ids.append(r["_id"])
                doc_texts.append(r["text"])
                if r.get("is_gold"):
                    gold_spans[r["_id"]] = (r["gold_char_start"], r["gold_char_end"])

    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"duplicate document ids in {rung_dir} -- shards overlap")
    return doc_ids, doc_texts, gold_spans


def load_queries_and_qrels(benchmark_dir: str):
    """Queries and qrels are copied unchanged from the source corpus at build time."""
    qpath = Path(benchmark_dir) / "queries.jsonl"
    queries: dict[str, str] = {}
    with open(qpath, encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]

    qrels: dict[str, dict[str, int]] = {}
    with open(Path(benchmark_dir) / "qrels" / "test.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            score = int(r.get("score", 0))
            if score > 0:
                qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = score
    return queries, qrels


def gold_visibility(
    doc_ids, doc_texts, gold_spans, tokenizer, limit: int, strategy: str,
    sample: int = 400
) -> dict:
    """How often the gold passage falls inside the model's window.

    This is the explanatory variable for the benchmark: a model that cannot reach the
    answering passage scores zero for reasons of capacity, not ranking quality. Sampled, since
    exact offsets need a full tokenization per document.

    Only meaningful for single-pass strategies. A chunked arm encodes *every* window, so "does
    the gold fall in the first ``limit`` tokens" describes nothing it does -- reporting 0.0%
    there reads as a failure when the model may be retrieving perfectly well. Chunked arms
    report gold-in-some-window instead, via the separate containment measurement.
    """
    if strategy not in ("native", "truncate"):
        return {"not_applicable": f"strategy={strategy} reads all windows"}
    idx = [i for i, d in enumerate(doc_ids) if d in gold_spans][:sample]
    if not idx:
        return {}
    visible = 0
    frac_seen = []
    for i in idx:
        text = doc_texts[i]
        a, b = gold_spans[doc_ids[i]]
        enc = tokenizer(text, add_special_tokens=True, truncation=False,
                        return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        tok_end = next(
            (j + 1 for j in range(len(offsets) - 1, -1, -1)
             if offsets[j][0] != offsets[j][1] and offsets[j][0] < b),
            0,
        )
        n_tok = len(offsets)
        visible += tok_end <= limit
        frac_seen.append(min(1.0, limit / max(1, n_tok)))
    return {
        "gold_visible_frac": visible / len(idx),
        "mean_frac_doc_seen": sum(frac_seen) / len(frac_seen),
        "n_sampled": len(idx),
    }


def run(args) -> dict:
    t0 = time.time()
    doc_ids, doc_texts, gold_spans = load_welded_corpus(
        args.benchmark_dir, args.condition, args.rung, args.position
    )
    queries, qrels = load_queries_and_qrels(args.benchmark_dir)
    qids = [q for q in queries if q in qrels]
    qtexts = [queries[q] for q in qids]
    print(f"{len(doc_ids):,} documents, {len(qids):,} queries "
          f"({len(gold_spans):,} golds at position={args.position})", flush=True)

    native = model_native_limit(args.model_name_or_path)
    window = resolve_window(
        args.strategy, args.window, native, model_label=args.model_label
    )
    print(f"strategy={args.strategy} window={window} native_limit={native}", flush=True)

    model = load_model(args.model_name_or_path, args.pooling, max_seq_length=window)
    qprefix, dprefix = get_prefixes(args.model_name_or_path, args)
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else model[0].tokenizer

    plan = plan_windows(
        doc_texts, tokenizer,
        strategy=args.strategy, window=window, stride=args.stride, doc_prefix=dprefix,
    )
    print(f"{plan.n_windows:,} windows ({plan.windows_per_doc:.2f} per document)", flush=True)

    vis = gold_visibility(doc_ids, doc_texts, gold_spans, tokenizer, window, args.strategy)
    if vis.get("gold_visible_frac") is not None:
        print(f"gold visible in {vis['gold_visible_frac']:.1%} of sampled gold documents; "
              f"model sees {vis['mean_frac_doc_seen']:.1%} of a document on average", flush=True)

    # The cache key must include EVERY input that changes the embeddings -- the arm, the
    # strategy, the window and the aggregation, not just the corpus. Keying on the corpus
    # alone made all four arms of a run share one directory, so each overwrote the previous
    # one and later arms scored against another model's vectors: NDB read 100% of the gold
    # and still scored 0.026 against its true 0.501.
    cache = args.embeddings_dir or os.path.join(
        os.path.dirname(args.output_file),
        "emb",
        f"{args.model_label}__{args.strategy}_w{window}_s{args.stride}",
    )
    q_emb, w_emb = encode_all(
        model,
        [qprefix + t for t in qtexts],
        plan.texts,
        args.batch_size,
        cache,
        args.force_reencode,
        model_path=args.model_name_or_path,
    )

    # A cache that belongs to a different corpus or strategy yields the wrong number of
    # vectors. Fail loudly rather than scoring queries against another run's embeddings.
    if w_emb.shape[0] != plan.n_windows:
        raise RuntimeError(
            f"embedding count {w_emb.shape[0]:,} != planned windows {plan.n_windows:,} "
            f"-- stale or shared cache at {cache!r}; delete it or pass --force_reencode"
        )
    if q_emb.shape[0] != len(qids):
        raise RuntimeError(
            f"query embedding count {q_emb.shape[0]:,} != queries {len(qids):,} "
            f"-- stale cache at {cache!r}"
        )

    if plan.n_windows == len(doc_ids):
        scores, indices = faiss_search(q_emb, w_emb, top_k=args.top_k + 1)
    else:
        # Chunked: score against windows, then reduce to documents. Done in query blocks so
        # peak memory is block x n_windows rather than n_queries x n_windows.
        import torch

        agg_scores = []
        agg_indices = []
        w = torch.as_tensor(w_emb)
        q = torch.as_tensor(q_emb)
        w2d = torch.as_tensor(plan.window_to_doc, dtype=torch.long)
        block = max(1, args.query_block)
        for s in range(0, q.shape[0], block):
            sim = q[s : s + block] @ w.T
            doc_scores = aggregate_window_scores(
                sim, w2d, len(doc_ids), how=args.aggregation
            )
            top = doc_scores.topk(min(args.top_k + 1, doc_scores.shape[1]), dim=1)
            agg_scores.append(top.values.cpu().numpy())
            agg_indices.append(top.indices.cpu().numpy())
        import numpy as np

        scores = np.concatenate(agg_scores)
        indices = np.concatenate(agg_indices)

    metrics = compute_metrics(scores, indices, doc_ids, qrels, qids, top_k=args.top_k)

    payload = {
        "arm": args.model_label,
        "dataset": os.path.basename(args.benchmark_dir.rstrip("/")),
        "condition": args.condition,
        "rung_chars": args.rung,
        "position_bin": args.position,
        "strategy": args.strategy,
        "window": window,
        "native_limit": native,
        "aggregation": args.aggregation if plan.n_windows != len(doc_ids) else None,
        "n_docs": len(doc_ids),
        "n_windows": plan.n_windows,
        "windows_per_doc": round(plan.windows_per_doc, 3),
        "n_queries": len(qids),
        "visibility": vis,
        "metrics": metrics,
        "runtime_seconds": round(time.time() - t0, 1),
        "config": {
            "model_path": args.model_name_or_path,
            "pooling": args.pooling,
            "batch_size": args.batch_size,
            "stride": args.stride,
            "query_prefix": qprefix,
            "doc_prefix": dprefix,
        },
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\nNDCG@10={metrics['ndcg_at_10']:.4f} MRR={metrics['mrr']:.4f} "
          f"R@100={metrics['recall_at_100']:.4f}", flush=True)
    print(f"-> {args.output_file}", flush=True)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--benchmark_dir", required=True)
    p.add_argument("--condition", default="random", choices=["random", "bm25"])
    p.add_argument("--rung", type=int, required=True)
    p.add_argument("--position", default="middle", choices=list(POSITION_BINS))
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--model_label", required=True)
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    p.add_argument("--strategy", default="native",
                   choices=["native", "truncate", "chunked", "chunked_para"])
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--aggregation", default="max", choices=["max", "mean"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--query_block", type=int, default=128)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--output_file", required=True)
    p.add_argument("--embeddings_dir", default=None)
    p.add_argument("--force_reencode", action="store_true")
    p.add_argument("--instruction_prefix_query", default=None)
    p.add_argument("--instruction_prefix_doc", default=None)
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except IncompatibleStrategy as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)

"""
BM25 control for the welded long-context benchmark.

This is the task-validity check, not a baseline. BM25 reads the **entire** document -- no
window, no truncation, no pooling -- so its score is a property of the *task*, independent of
any model's architecture.

That makes the c0-to-c27000 trajectory diagnostic:

* BM25 degrades only mildly  -> the query/gold relationship survived welding, and a neural
  model's collapse is a limitation of that model, not of the benchmark.
* BM25 collapses too         -> welding damaged the retrieval task itself, and every neural
  number measured here is an artifact of the construction.

Some degradation is expected and correct: term-frequency normalisation means a gold passage's
terms carry less weight inside a document that is mostly filler, which is the intended
"needle in a haystack" difficulty. What would invalidate the benchmark is a collapse toward
zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from data.long_context.bm25 import BM25Index


def load_corpus(benchmark_dir: str, condition: str, rung: int | None, position: str):
    """Load a welded rung, or the unpadded source corpus when rung is None (c0)."""
    if rung is None:
        src = json.load(open(Path(benchmark_dir) / "manifest.json"))["source_beir_dir"]
        ids, texts = [], []
        with open(Path(src) / "corpus.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                title = (d.get("title") or "").strip()
                text = (d.get("text") or "").strip()
                ids.append(str(d["_id"]))
                texts.append(f"{title} {text}".strip() if title else text)
        return ids, texts

    rung_dir = Path(benchmark_dir) / condition / f"c{rung}"
    ids, texts = [], []
    for shard in ("nongold.jsonl", f"gold_{position}.jsonl"):
        with open(rung_dir / shard, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                ids.append(d["_id"])
                texts.append(d["text"])
    return ids, texts


def load_queries_qrels(benchmark_dir: str):
    queries = {}
    with open(Path(benchmark_dir) / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(benchmark_dir) / "qrels" / "test.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if int(r.get("score", 0)) > 0:
                qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return queries, qrels


def ndcg_at_k(ranked_rel, all_rel, k=10):
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ranked_rel[:k]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(all_rel, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def run(args):
    t0 = time.time()
    ids, texts = load_corpus(args.benchmark_dir, args.condition, args.rung, args.position)
    queries, qrels = load_queries_qrels(args.benchmark_dir)
    qids = [q for q in queries if q in qrels]
    print(f"{len(ids):,} docs, {len(qids):,} queries", flush=True)

    index = BM25Index.build(ids, texts)
    hits = index.top_k([queries[q] for q in qids], k=args.top_k, block=200)

    ndcgs, mrrs, recalls = [], [], []
    for qid, hit in zip(qids, hits):
        rel = qrels[qid]
        ranked = [float(rel.get(ids[i], 0)) for i, _ in hit]
        ndcgs.append(ndcg_at_k(ranked, list(rel.values()), 10))
        first = next((r + 1 for r, g in enumerate(ranked) if g > 0), None)
        mrrs.append(1.0 / first if first else 0.0)
        n_rel = len(rel)
        recalls.append(sum(1 for g in ranked if g > 0) / n_rel if n_rel else 0.0)

    metrics = {
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "mrr": sum(mrrs) / len(mrrs),
        f"recall_at_{args.top_k}": sum(recalls) / len(recalls),
        "num_queries_evaluated": len(qids),
    }
    payload = {
        "arm": "BM25",
        "dataset": os.path.basename(args.benchmark_dir.rstrip("/")),
        "condition": args.condition,
        "rung_chars": args.rung or 0,
        "position_bin": args.position,
        "n_docs": len(ids),
        "metrics": metrics,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"NDCG@10={metrics['ndcg_at_10']:.4f} MRR={metrics['mrr']:.4f} "
          f"-> {args.output_file}", flush=True)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--benchmark_dir", required=True)
    p.add_argument("--condition", default="random", choices=["random", "bm25"])
    p.add_argument("--rung", type=int, default=None, help="omit for c0 (unpadded source)")
    p.add_argument("--position", default="middle", choices=["start", "middle", "end"])
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--output_file", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

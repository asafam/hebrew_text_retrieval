"""Mine hard negatives for BEIR training using BM25.

For each training (query, positive) pair, retrieves top-K BM25 results from the
corpus, excludes known positives, and saves the top hard_negatives_per_query
results as explicit negatives.

Output: {corpus_dir}/hard_negatives_train.jsonl
  Each line: {"query": str, "positive": str, "hard_negs": [str, str, ...]}

Usage:
    python src/data/mining/mine_hard_negatives.py \
        --corpora_root outputs/translation/runs/.../corpus \
        --num_hard_negatives 2 \
        --bm25_top_k 100
"""

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _tokenize(text: str) -> list:
    """Simple whitespace+punctuation tokenizer — sufficient for BM25 negative mining."""
    return re.findall(r'\w+', text.lower())


def load_beir_corpus(corpus_dir: str):
    corpus, queries, qrels = {}, {}, {}

    with open(os.path.join(corpus_dir, "corpus.jsonl")) as f:
        for line in f:
            doc = json.loads(line)
            title = (doc.get("title") or "").strip()
            text = (doc.get("text") or "").strip()
            corpus[doc["_id"]] = (title + " " + text).strip() if title else text

    with open(os.path.join(corpus_dir, "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels_path = os.path.join(corpus_dir, "qrels", "train.tsv")
    if not os.path.exists(qrels_path):
        return corpus, queries, qrels

    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                qid, docid, score = parts[0], parts[1], int(float(parts[2]))
            except ValueError:
                continue
            if score > 0:
                qrels.setdefault(qid, set()).add(docid)

    return corpus, queries, qrels


def mine_corpus(corpus_dir: str, num_hard_negatives: int, bm25_top_k: int) -> str:
    out_path = os.path.join(corpus_dir, "hard_negatives_train.jsonl")
    dataset_name = os.path.basename(corpus_dir)
    print(f"\n[{dataset_name}] Loading data...")

    corpus, queries, qrels = load_beir_corpus(corpus_dir)
    if not qrels:
        print(f"  No train qrels — skipping.")
        return None

    # Build BM25 index
    print(f"  Building BM25 index on {len(corpus):,} documents...")
    doc_ids = list(corpus.keys())
    tokenized_corpus = [_tokenize(corpus[did]) for did in tqdm(doc_ids, desc="Tokenizing corpus")]
    bm25 = BM25Okapi(tokenized_corpus)

    # Mine hard negatives for each training pair
    pairs_written = 0
    skipped_no_neg = 0

    with open(out_path, "w") as out_f:
        for qid, pos_ids in tqdm(qrels.items(), desc="Mining negatives"):
            if qid not in queries:
                continue
            query_text = queries[qid]
            tokenized_query = _tokenize(query_text)

            scores = bm25.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)

            # Collect hard negatives (top BM25 hits excluding known positives)
            hard_negs = []
            for idx in ranked_indices[:bm25_top_k]:
                doc_id = doc_ids[idx]
                if doc_id not in pos_ids:
                    hard_negs.append(corpus[doc_id])
                if len(hard_negs) >= num_hard_negatives:
                    break

            if not hard_negs:
                skipped_no_neg += 1
                continue

            # Write one record per positive
            for pos_id in pos_ids:
                if pos_id not in corpus:
                    continue
                record = {
                    "query": query_text,
                    "positive": corpus[pos_id],
                    "hard_negs": hard_negs,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pairs_written += 1

    print(f"  Written {pairs_written:,} pairs with hard negatives → {out_path}")
    if skipped_no_neg:
        print(f"  Skipped {skipped_no_neg} queries with no hard negatives found")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Mine BM25 hard negatives for BeIR training.")
    parser.add_argument("--corpora_root", required=True,
                        help="Root dir containing BeIR corpus subdirs.")
    parser.add_argument("--num_hard_negatives", type=int, default=2,
                        help="Hard negatives per query (default: 2).")
    parser.add_argument("--bm25_top_k", type=int, default=100,
                        help="Number of BM25 results to consider before picking negatives.")
    args = parser.parse_args()

    train_tsvs = glob(os.path.join(args.corpora_root, "**/qrels/train.tsv"), recursive=True)
    corpus_dirs = sorted(os.path.dirname(os.path.dirname(f)) for f in train_tsvs)

    if not corpus_dirs:
        print(f"No BEIR corpora with train splits found under {args.corpora_root}")
        return

    print(f"Found {len(corpus_dirs)} corpus/corpora: {[os.path.basename(d) for d in corpus_dirs]}")
    for corpus_dir in corpus_dirs:
        mine_corpus(corpus_dir, args.num_hard_negatives, args.bm25_top_k)

    print("\nDone. Hard negatives written to hard_negatives_train.jsonl in each corpus dir.")


if __name__ == "__main__":
    main()

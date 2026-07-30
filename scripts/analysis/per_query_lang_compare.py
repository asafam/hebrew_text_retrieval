#!/usr/bin/env python3
"""Per-query Hebrew vs English retrieval, same model, same items.

Runs one multilingual model (default mE5-base) over each dataset twice — once on
the Hebrew translation, once on the English source — and records, for every
query, where the first relevant document landed. Because ids and qrels are
shared, the two runs are the same retrieval problem in two languages.

The resulting 2x2 (succeeds in He? x succeeds in En?) is what separates
translation-attributable failures from failures the retriever has in both
languages. Note the English side is an upper bound that also benefits from mE5
being stronger in English than Hebrew — attribute_failures.py tests that
confound rather than assuming it away.

Output: outputs/analysis/per_query/<dataset>.jsonl, one row per query:
    {qid, n_relevant, he_rank, en_rank, he_hit10, en_hit10, he_rr, en_rr}
`*_rank` is the 1-based rank of the first relevant document, or null if none in
the retrieved top-k.
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO / "src" / "model" / "eval" / "eval_beir_retrieval_zeroshot.py"

HE_ROOT = ("outputs/translation/runs/"
           "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
EN_ROOT = "outputs/analysis/english_mirror"


def load_eval_module():
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("_beir_eval", EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rank_of_first_relevant(indices_row, doc_ids, judged, qid, exclude_self, top_k):
    """1-based rank of the first judged-relevant doc, or None. Mirrors the
    self-exclusion rule used by compute_metrics so the two agree."""
    rank = 0
    for idx in indices_row:
        did = doc_ids[idx]
        if did == qid and exclude_self and judged.get(did, 0) <= 0:
            continue
        rank += 1
        if rank > top_k:
            break
        if judged.get(did, 0) > 0:
            return rank
    return None


def run_lang(ev, model, corpus_dir, cache_dir, batch_size, top_k, tag):
    corpus, queries, qrels = ev.load_beir_local(corpus_dir)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())

    overlap = len(set(query_ids) & set(doc_ids)) / max(1, len(query_ids))
    exclude_self = overlap >= ev.SELF_OVERLAP_THRESHOLD
    print(f"  [{tag}] {len(doc_ids):,} docs, {len(query_ids):,} queries, "
          f"self-overlap {overlap:.1%} -> exclude_self={exclude_self}")

    q_texts = ["query: " + queries[q] for q in query_ids]
    d_texts = ["passage: " + corpus[d] for d in doc_ids]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    q_emb, d_emb = ev.encode_all(model, q_texts, d_texts, batch_size, cache_dir,
                                 force_reencode=False, model_path=None)

    search_k = min(top_k + 1, len(doc_ids)) if exclude_self else top_k
    _, indices = ev.faiss_search(q_emb, d_emb, search_k)

    out = {}
    for i, qid in enumerate(query_ids):
        judged = qrels.get(qid)
        if not judged or not any(s > 0 for s in judged.values()):
            continue
        r = rank_of_first_relevant(indices[i], doc_ids, judged, qid, exclude_self, top_k)
        out[qid] = {"rank": r,
                    "n_relevant": sum(1 for s in judged.values() if s > 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model_tag", default="mE5-base")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--out_dir", default="outputs/analysis/per_query")
    a = ap.parse_args()

    ev = load_eval_module()
    print(f"Loading {a.model} ...")
    model = ev.load_model(a.model, pooling_mode="mean", max_seq_length=512)

    datasets = a.datasets or sorted(os.listdir(HE_ROOT))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        he_dir, en_dir = f"{HE_ROOT}/{ds}/beir", f"{EN_ROOT}/{ds}/beir"
        if not os.path.exists(en_dir):
            print(f"skip {ds}: no English mirror")
            continue
        print(f"\n=== {ds} ===")
        cache = f"outputs/analysis/embeddings/{a.model_tag}/{ds}"
        he = run_lang(ev, model, he_dir, f"{cache}/he", a.batch_size, a.top_k, "he")
        en = run_lang(ev, model, en_dir, f"{cache}/en", a.batch_size, a.top_k, "en")

        path = Path(a.out_dir) / f"{ds}.jsonl"
        n = 0
        with open(path, "w") as f:
            for qid in he:
                if qid not in en:
                    continue
                hr, er = he[qid]["rank"], en[qid]["rank"]
                f.write(json.dumps({
                    "qid": qid,
                    "n_relevant": he[qid]["n_relevant"],
                    "he_rank": hr, "en_rank": er,
                    "he_hit10": bool(hr and hr <= 10), "en_hit10": bool(er and er <= 10),
                    "he_rr": (1.0 / hr) if hr else 0.0, "en_rr": (1.0 / er) if er else 0.0,
                }) + "\n")
                n += 1
        print(f"  wrote {n} queries -> {path}")


if __name__ == "__main__":
    main()

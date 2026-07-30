#!/usr/bin/env python3
"""Zero-shot retrieval benchmark for Hebrew IR datasets.

Evaluates any HuggingFace model against MIRACL Hebrew or local BEIR-format data.
Reports NDCG@10, NDCG@100, R@100, MRR.

Supports three model types, auto-detected from the checkpoint:
  - SentenceTransformer with saved pooling config (e.g. SBERT fine-tuned models)
  - InfoNCEDualEncoder (local dual-encoder checkpoints from dual_encoder/ training)
  - Any HF model built as Transformer+Pooling (HF hub names or raw base LMs)

Usage examples:
    # mE5 baseline
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path intfloat/multilingual-e5-base \
        --corpus_dir outputs/translation/runs/.../corpus/BeIR_nfcorpus/beir

    # SBERT fine-tuned checkpoint (pooling from saved config)
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path outputs/archive/models/sbert/sbert-hebmodernbert-hebnli/ckpt_... \
        --corpus_dir ...

    # InfoNCE dual-encoder checkpoint
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path outputs/archive/models/dual_encoder/.../model \
        --corpus_dir ...
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# InfoNCE dual-encoder wrapper
# ---------------------------------------------------------------------------

class InfoNCEWrapper:
    """Wraps InfoNCEDualEncoder to expose batched text encoding for queries and docs."""

    def __init__(self, model_path, max_length=512):
        from model.dual_encoder.models import InfoNCEDualEncoderConfig, InfoNCEDualEncoder

        config = InfoNCEDualEncoderConfig.from_pretrained(model_path)
        self.dual = InfoNCEDualEncoder.from_pretrained(model_path, config=config)
        self.dual.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dual.to(self.device)
        self.max_length = max_length

        tok_path = os.path.join(model_path, "tokenizer_query")
        if not os.path.exists(tok_path):
            tok_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

    def _encode(self, inner_encoder, texts, batch_size, show_progress_bar):
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Encoding", disable=not show_progress_bar):
            batch = texts[i:i + batch_size]
            tok = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
            with torch.no_grad():
                emb = self.dual.encode(inner_encoder, input_ids, attention_mask)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

    def encode_queries(self, texts, batch_size=256, show_progress_bar=True):
        return self._encode(self.dual.query_encoder, texts, batch_size, show_progress_bar)

    def encode_docs(self, texts, batch_size=256, show_progress_bar=True):
        return self._encode(self.dual.doc_encoder, texts, batch_size, show_progress_bar)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _is_infonce(path):
    cfg = os.path.join(path, "config.json")
    if not os.path.exists(cfg):
        return False
    with open(cfg) as f:
        c = json.load(f)
    return (c.get("model_type") == "info_nce_dual_encoder" or
            "InfoNCEDualEncoder" in c.get("architectures", []))


def _has_st_config(path):
    return (os.path.isdir(os.path.join(path, "1_Pooling")) or
            os.path.exists(os.path.join(path, "sentence_bert_config.json")))


def load_model(model_name_or_path, pooling_mode="mean", max_seq_length=512):
    """Load model — auto-detects InfoNCEDualEncoder, SentenceTransformer, or plain HF model."""
    if os.path.isdir(model_name_or_path):
        if _is_infonce(model_name_or_path):
            print("Detected InfoNCEDualEncoder checkpoint.")
            return InfoNCEWrapper(model_name_or_path, max_length=max_seq_length)
        if _has_st_config(model_name_or_path):
            print("Detected SentenceTransformer checkpoint — loading with saved pooling config.")
            return SentenceTransformer(model_name_or_path)

    # Build ST explicitly from any HF model (hub name or raw local checkpoint)
    transformer = models.Transformer(
        model_name_or_path,
        max_seq_length=max_seq_length,
        tokenizer_args={"trust_remote_code": True},
        model_args={"trust_remote_code": True},
    )
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=(pooling_mode == "mean"),
        pooling_mode_cls_token=(pooling_mode == "cls"),
    )
    return SentenceTransformer(modules=[transformer, pooling])


def get_prefixes(model_name_or_path, args):
    """Return (query_prefix, doc_prefix). Auto-detect E5 by model name; allow CLI override."""
    if args.instruction_prefix_query is not None:
        return args.instruction_prefix_query, (args.instruction_prefix_doc or "")
    if "multilingual-e5" in model_name_or_path.lower():
        return "query: ", "passage: "
    return "", ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _join_title_text(title, text):
    parts = [p.strip() for p in [title or "", text or ""] if p and p.strip()]
    return " ".join(parts)


def load_miracl(lang="he"):
    """Load MIRACL corpus, queries, and qrels from HuggingFace."""
    from datasets import load_dataset

    print(f"Loading MIRACL corpus (lang={lang})...")
    corpus_ds = load_dataset("miracl/miracl-corpus", lang, split="train", trust_remote_code=True)
    corpus = {}
    for doc in tqdm(corpus_ds, desc="Building corpus"):
        corpus[doc["docid"]] = _join_title_text(doc.get("title", ""), doc.get("text", ""))

    print(f"Loading MIRACL queries (lang={lang}, dev split)...")
    queries_ds = load_dataset("miracl/miracl", lang, split="dev", trust_remote_code=True)
    queries = {}
    qrels = {}
    for item in queries_ds:
        qid = item["query_id"]
        queries[qid] = item["query"]
        qrels[qid] = {pos["docid"]: 1 for pos in item.get("positive_passages", [])}

    return corpus, queries, qrels


def load_beir_local(corpus_dir):
    """Load corpus, queries, and qrels from a local BEIR-format directory."""
    print(f"Loading BEIR data from {corpus_dir}...")
    corpus = {}
    queries = {}
    qrels = {}

    with open(os.path.join(corpus_dir, "corpus.jsonl")) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = _join_title_text(doc.get("title", ""), doc.get("text", ""))

    with open(os.path.join(corpus_dir, "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels_dir = os.path.join(corpus_dir, "qrels")
    qrels_loaded = False
    for fname in ["test.jsonl", "dev.jsonl", "validation.jsonl", "test.tsv", "dev.tsv", "validation.tsv"]:
        qrels_path = os.path.join(qrels_dir, fname)
        if not os.path.exists(qrels_path):
            continue
        with open(qrels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if fname.endswith(".jsonl"):
                    rec = json.loads(line)
                    qid, docid, score = str(rec["query-id"]), str(rec["corpus-id"]), int(rec["score"])
                else:
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        qid, docid, score = parts[0], parts[1], int(float(parts[2]))
                    except ValueError:
                        continue
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = score
        qrels_loaded = True
        print(f"Loaded qrels from {qrels_path}")
        break

    if not qrels_loaded:
        raise FileNotFoundError(f"No qrels file found in {qrels_dir}")

    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Embedding with caching and partial-resume
# ---------------------------------------------------------------------------

def batched_encode_st(encode_fn, texts, batch_size, cache_file=None, chunk_size=50000):
    """Encode texts via encode_fn, with chunk-level partial-resume support.

    encode_fn(chunk_texts, batch_size, show_progress_bar) -> Tensor
    """
    all_embeddings = []
    start_chunk = 0
    partial_file = cache_file.replace(".pt", "_partial.pt") if cache_file else None

    if partial_file and os.path.exists(partial_file):
        print(f"[Resuming] Loading partial embeddings from {partial_file}")
        all_embeddings = list(torch.load(partial_file, weights_only=False))
        start_chunk = len(all_embeddings)
        print(f"[Resuming] {start_chunk} chunks already encoded ({start_chunk * chunk_size:,} texts).")

    total_chunks = (len(texts) + chunk_size - 1) // chunk_size

    for chunk_idx in range(start_chunk, total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]
        print(f"Encoding chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_texts):,} texts)...")

        chunk_emb = encode_fn(chunk_texts, batch_size=batch_size, show_progress_bar=True)
        if not isinstance(chunk_emb, torch.Tensor):
            chunk_emb = torch.tensor(chunk_emb)
        all_embeddings.append(chunk_emb.cpu())

        if partial_file:
            Path(os.path.dirname(partial_file)).mkdir(parents=True, exist_ok=True)
            torch.save(all_embeddings, partial_file)

    final_embeddings = torch.cat(all_embeddings, dim=0)

    if cache_file:
        Path(os.path.dirname(cache_file)).mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings to {cache_file}")
        torch.save(final_embeddings.cpu(), cache_file)

    if partial_file and os.path.exists(partial_file):
        os.remove(partial_file)

    return final_embeddings


def _st_encode_fn(model):
    """Return an encode function for a SentenceTransformer model."""
    def fn(texts, batch_size, show_progress_bar):
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
        )
    return fn


def _model_mtime(model_path):
    """Newest mtime among a model's weight/config files (None if not found).

    Used to auto-invalidate cached embeddings: if the model is newer than the
    cache, the cache was produced by a different/older model and is stale.
    """
    if not model_path or not os.path.exists(model_path):
        return None
    candidates = []
    for root, _dirs, files in os.walk(model_path):
        for fn in files:
            if fn.endswith((".safetensors", ".bin", ".pt", ".json")):
                try:
                    candidates.append(os.path.getmtime(os.path.join(root, fn)))
                except OSError:
                    pass
    return max(candidates) if candidates else None


def encode_all(model, query_texts, doc_texts, batch_size, embeddings_dir, force_reencode, model_path=None):
    q_cache = os.path.join(embeddings_dir, "query_embeddings.pt") if embeddings_dir else None
    d_cache = os.path.join(embeddings_dir, "doc_embeddings.pt") if embeddings_dir else None

    # Auto-invalidate stale caches: if the model is newer than a cached embedding
    # file, that cache came from a different model version -> must re-encode.
    model_mtime = _model_mtime(model_path)

    def _stale(cache_file):
        if model_mtime is None or not cache_file or not os.path.exists(cache_file):
            return False
        if os.path.getmtime(cache_file) < model_mtime:
            print(f"[stale-cache] {cache_file} is older than the model — re-encoding.")
            return True
        return False

    # Build encode functions — dual encoder uses separate query/doc encoders
    if isinstance(model, InfoNCEWrapper):
        encode_q = lambda texts, batch_size, show_progress_bar: model.encode_queries(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        encode_d = lambda texts, batch_size, show_progress_bar: model.encode_docs(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
    else:
        encode_q = encode_d = _st_encode_fn(model)

    print(f"Encoding {len(query_texts):,} queries...")
    if not force_reencode and not _stale(q_cache) and q_cache and os.path.exists(q_cache):
        q_emb = torch.load(q_cache, weights_only=False)
        print(f"Loaded query embeddings from {q_cache}")
    else:
        q_emb = batched_encode_st(encode_q, query_texts, batch_size=batch_size, cache_file=q_cache)

    print(f"Encoding {len(doc_texts):,} documents...")
    if not force_reencode and not _stale(d_cache) and d_cache and os.path.exists(d_cache):
        d_emb = torch.load(d_cache, weights_only=False)
        print(f"Loaded doc embeddings from {d_cache}")
    else:
        d_emb = batched_encode_st(encode_d, doc_texts, batch_size=batch_size, cache_file=d_cache)

    return q_emb, d_emb


# ---------------------------------------------------------------------------
# FAISS search
# ---------------------------------------------------------------------------

def faiss_search(q_emb, d_emb, top_k):
    """L2-normalized inner-product search (= cosine similarity)."""
    dim = d_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(d_emb.numpy().astype("float32"))
    scores, indices = index.search(q_emb.numpy().astype("float32"), top_k)
    return scores, indices  # both (n_queries, top_k)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _gains(relevance, gain_mode):
    """Map raw relevance grades to DCG gains.

    'linear' (gain = rel) is the trec_eval `ndcg_cut` convention, which is what
    pytrec_eval and therefore published BEIR numbers use — verified to match
    pytrec_eval to <1e-12 on graded nfcorpus qrels. 'exponential' (2^rel - 1) is
    the Burges/Kaggle variant; it is NOT what BEIR reports. The two only differ
    where grades exceed 1 (nfcorpus, trec-covid, dbpedia-entity, webis-touche2020).
    """
    rel = np.asarray(relevance, dtype=float)
    return np.power(2.0, rel) - 1.0 if gain_mode == "exponential" else rel


# Fraction of query ids that must also be document ids before `--exclude_self auto`
# treats the overlap as structural (ArguAna: 92%) rather than coincidental (fiqa: 9%).
SELF_OVERLAP_THRESHOLD = 0.5


def _dcg(gains):
    if len(gains) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(np.asarray(gains, dtype=float) * discounts))


def _ndcg_at_k(ranked_relevance, ideal_relevance, k, gain_mode):
    """NDCG@k with the ideal ranking taken from the FULL qrels, not the retrieved slice.

    Normalizing against only the retrieved documents inflates the score whenever a
    query has more relevant documents than fit in the retrieved top-k (nfcorpus
    averages 38 positives/query), so `ideal_relevance` must be every judged grade
    for the query, sorted descending.
    """
    dcg = _dcg(_gains(ranked_relevance[:k], gain_mode))
    idcg = _dcg(_gains(ideal_relevance[:k], gain_mode))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(retrieved_scores, retrieved_indices, doc_ids, qrels, query_ids,
                    top_k=100, exclude_self="auto", gain="linear"):
    """Compute NDCG@10, NDCG@100, R@100, MRR from FAISS results.

    retrieved_* may contain more than `top_k` columns so that self-exclusion can
    drop a document without shortening the evaluated ranking.

    exclude_self controls removal of the document whose id equals the query id
    (ArguAna ships its query arguments inside the corpus, so a model retrieves the
    query's own near-duplicate at rank 1 and demotes the true counterargument):
      auto   — apply id-based exclusion only when the dataset shows *structural*
               query-in-corpus overlap (>= SELF_OVERLAP_THRESHOLD of query ids
               present as documents, as in ArguAna's 92%). Datasets where query
               and corpus ids merely share an integer namespace by coincidence —
               fiqa overlaps on 9% of ids, with entirely different text — are left
               untouched, so genuine documents are not dropped from the ranking.
      always — drop the same-id document unconditionally (strict BEIR protocol).
      never  — keep it.
    A document that qrels judge relevant for the query is never dropped.
    """
    ndcg10, ndcg100, recall, hit, mrr_scores = [], [], [], [], []
    num_self_excluded = 0

    if exclude_self == "auto":
        overlap = len(set(query_ids) & set(doc_ids)) / max(1, len(query_ids))
        structural = overlap >= SELF_OVERLAP_THRESHOLD
        print(f"[self-exclusion] {overlap:.1%} of query ids appear as document ids -> "
              f"{'structural leakage, excluding' if structural else 'coincidental, keeping'}")
        if not structural:
            exclude_self = "never"

    for i, qid in enumerate(query_ids):
        judged = qrels.get(qid)
        if not judged:
            continue

        # Full-qrels ideal ranking (score<=0 rows, e.g. scidocs' negative pool,
        # contribute zero gain and are dropped from the denominator).
        ideal = sorted((g for g in judged.values() if g > 0), reverse=True)
        num_relevant = len(ideal)

        ranked_ids, ranked_rel = [], []
        for rank_pos, idx in enumerate(retrieved_indices[i]):
            did = doc_ids[idx]
            if did == qid and exclude_self != "never":
                if exclude_self == "always" or judged.get(did, 0) <= 0:
                    num_self_excluded += 1
                    continue
            ranked_ids.append(did)
            ranked_rel.append(judged.get(did, 0))
            if len(ranked_ids) >= top_k:
                break

        ndcg10.append(_ndcg_at_k(ranked_rel, ideal, 10, gain))
        ndcg100.append(_ndcg_at_k(ranked_rel, ideal, 100, gain))

        rel_arr = np.asarray(ranked_rel, dtype=float)
        num_found = int(np.sum(rel_arr > 0))
        recall.append(num_found / num_relevant if num_relevant else 0.0)
        hit.append(float(num_found > 0))

        positions = np.where(rel_arr > 0)[0]
        mrr_scores.append(1.0 / (positions[0] + 1) if len(positions) > 0 else 0.0)

    if not ndcg10:
        warnings.warn("No queries with valid qrels found — cannot compute metrics.")
        return {}

    return {
        "ndcg_at_10": float(np.mean(ndcg10)),
        "ndcg_at_100": float(np.mean(ndcg100)),
        # True recall: fraction of a query's judged-relevant docs found in top-k.
        # `hit_rate_at_100` is the previously-reported "any relevant retrieved"
        # figure, kept so older results.json files stay comparable.
        "recall_at_100": float(np.mean(recall)),
        "hit_rate_at_100": float(np.mean(hit)),
        "mrr": float(np.mean(mrr_scores)),
        "num_queries_evaluated": len(ndcg10),
        "num_self_excluded": num_self_excluded,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot BEIR retrieval evaluation.")
    parser.add_argument("--model_name_or_path", required=True,
                        help="HuggingFace model name or local checkpoint path.")
    parser.add_argument("--model_label", default=None,
                        help="Short display name for results (default: derived from path).")
    parser.add_argument("--dataset_hf", default=None,
                        help="HuggingFace dataset name (e.g. miracl/miracl). Use with --lang.")
    parser.add_argument("--lang", default="he",
                        help="Language code for HF dataset (default: he).")
    parser.add_argument("--corpus_dir", default=None,
                        help="Local BEIR corpus dir with corpus.jsonl, queries.jsonl, qrels/.")
    parser.add_argument("--dataset_name", default=None,
                        help="Short name for output paths. Auto-inferred if not set.")
    parser.add_argument("--output_dir", default=None,
                        help="Base output dir. Default: outputs/eval/beir_zeroshot/{model}/{dataset}/")
    parser.add_argument("--output_file", default=None,
                        help="Full path to JSON results file (overrides --output_dir).")
    parser.add_argument("--embeddings_dir", default=None,
                        help="Directory for .pt embedding cache files. Default: same as output_dir.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"],
                        help="Pooling for plain HF models. Ignored for ST/dual-encoder checkpoints.")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of documents to retrieve per query (default: 100).")
    parser.add_argument("--exclude_self", default="auto", choices=["auto", "always", "never"],
                        help="Drop the corpus doc whose id equals the query id before scoring. "
                             "Required for correct ArguAna numbers (its queries are in the corpus). "
                             "auto (default) keeps it when qrels judge it relevant; never = pre-fix behavior.")
    parser.add_argument("--ndcg_gain", default="linear", choices=["linear", "exponential"],
                        help="DCG gain function. linear (default, gain=rel) matches trec_eval/"
                             "pytrec_eval and published BEIR numbers. exponential (2^rel-1) is the "
                             "Burges variant and is NOT BEIR-comparable. Only differs on graded "
                             "qrels (nfcorpus, trec-covid, dbpedia-entity, webis-touche2020).")
    parser.add_argument("--force_reencode", action="store_true",
                        help="Re-encode even if cached .pt files exist.")
    parser.add_argument("--instruction_prefix_query", default=None,
                        help="Override query instruction prefix (e.g. 'query: ').")
    parser.add_argument("--instruction_prefix_doc", default=None,
                        help="Override document instruction prefix (e.g. 'passage: ').")
    args = parser.parse_args()

    if not args.dataset_hf and not args.corpus_dir:
        parser.error("Provide --dataset_hf or --corpus_dir")

    # Infer dataset name for output path
    if args.dataset_name:
        dataset_name = args.dataset_name
    elif args.dataset_hf:
        dataset_name = f"{args.dataset_hf.replace('/', '_')}_{args.lang}"
    else:
        p = Path(args.corpus_dir)
        dataset_name = p.parent.name if p.name == "beir" else p.name

    # Model slug for output paths (use --model_label if provided)
    if args.model_label:
        model_slug = args.model_label
    else:
        model_slug = args.model_name_or_path.rstrip("/\\").replace("/", "_").replace("\\", "_")

    # Resolve output paths
    if args.output_file:
        output_file = args.output_file
        out_dir = str(Path(output_file).parent)
    else:
        out_dir = args.output_dir or f"outputs/eval/beir_zeroshot/{model_slug}/{dataset_name}"
        output_file = os.path.join(out_dir, "results.json")

    embeddings_dir = args.embeddings_dir or out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    if args.dataset_hf:
        corpus, queries, qrels = load_miracl(lang=args.lang)
    else:
        corpus, queries, qrels = load_beir_local(args.corpus_dir)

    print(f"Corpus: {len(corpus):,} docs | Queries: {len(queries):,} | "
          f"Qrels: {len(qrels):,} queries with judgments")

    # Load model
    print(f"\nLoading model: {args.model_name_or_path}")
    model = load_model(args.model_name_or_path, pooling_mode=args.pooling,
                       max_seq_length=args.max_length)

    query_prefix, doc_prefix = get_prefixes(args.model_name_or_path, args)
    if not isinstance(model, InfoNCEWrapper):
        print(f"Instruction prefixes — query: {repr(query_prefix)}  doc: {repr(doc_prefix)}")

    # Build ordered lists
    doc_ids = list(corpus.keys())
    doc_texts = [doc_prefix + corpus[did] for did in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [query_prefix + queries[qid] for qid in query_ids]

    # Encode
    print()
    q_emb, d_emb = encode_all(model, query_texts, doc_texts, args.batch_size,
                               embeddings_dir, args.force_reencode, model_path=args.model_name_or_path)
    print(f"Query embeddings: {tuple(q_emb.shape)} | Doc embeddings: {tuple(d_emb.shape)}")

    # Normalize InfoNCEWrapper embeddings (already normalized for ST models)
    if isinstance(model, InfoNCEWrapper):
        q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
        d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

    # Search one extra document so self-exclusion cannot shorten the top-k ranking.
    search_k = min(args.top_k + 1, len(doc_ids)) if args.exclude_self != "never" else args.top_k
    print(f"\nSearching top-{search_k} with FAISS IndexFlatIP...")
    scores, indices = faiss_search(q_emb, d_emb, search_k)

    # Metrics
    metrics = compute_metrics(scores, indices, doc_ids, qrels, query_ids,
                              top_k=args.top_k, exclude_self=args.exclude_self,
                              gain=args.ndcg_gain)
    if metrics.get("num_self_excluded"):
        print(f"Self-exclusion: removed the query's own document for "
              f"{metrics['num_self_excluded']:,} queries.")

    result = {
        "model": model_slug,
        "dataset": dataset_name,
        "metrics": metrics,
        "config": {
            "model_path": args.model_name_or_path,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "exclude_self": args.exclude_self,
            "ndcg_gain": args.ndcg_gain,
            "query_prefix": query_prefix,
            "doc_prefix": doc_prefix,
        },
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

    # One-line summary
    m = metrics
    header = f"{'Model':<45} {'Dataset':<25} {'NDCG@10':>8} {'NDCG@100':>9} {'R@100':>8} {'MRR':>8}"
    row = (
        f"{model_slug:<45} {dataset_name:<25} "
        f"{m.get('ndcg_at_10', 0):>8.4f} {m.get('ndcg_at_100', 0):>9.4f} "
        f"{m.get('recall_at_100', 0):>8.4f} {m.get('mrr', 0):>8.4f}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    print(row)


if __name__ == "__main__":
    main()

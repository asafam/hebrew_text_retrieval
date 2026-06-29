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
from sklearn.metrics import ndcg_score
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


def encode_all(model, query_texts, doc_texts, batch_size, embeddings_dir, force_reencode):
    q_cache = os.path.join(embeddings_dir, "query_embeddings.pt") if embeddings_dir else None
    d_cache = os.path.join(embeddings_dir, "doc_embeddings.pt") if embeddings_dir else None

    # Build encode functions — dual encoder uses separate query/doc encoders
    if isinstance(model, InfoNCEWrapper):
        encode_q = lambda texts, batch_size, show_progress_bar: model.encode_queries(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        encode_d = lambda texts, batch_size, show_progress_bar: model.encode_docs(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
    else:
        encode_q = encode_d = _st_encode_fn(model)

    print(f"Encoding {len(query_texts):,} queries...")
    if not force_reencode and q_cache and os.path.exists(q_cache):
        q_emb = torch.load(q_cache, weights_only=False)
        print(f"Loaded query embeddings from {q_cache}")
    else:
        q_emb = batched_encode_st(encode_q, query_texts, batch_size=batch_size, cache_file=q_cache)

    print(f"Encoding {len(doc_texts):,} documents...")
    if not force_reencode and d_cache and os.path.exists(d_cache):
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

def compute_metrics(retrieved_scores, retrieved_indices, doc_ids, qrels, query_ids):
    """Compute NDCG@10, NDCG@100, R@100, MRR from FAISS top-K results."""
    ndcg_y_true = []
    ndcg_y_score = []
    r_at_100 = []
    mrr_scores = []

    for i, qid in enumerate(query_ids):
        if qid not in qrels or not qrels[qid]:
            continue

        top_doc_ids = [doc_ids[idx] for idx in retrieved_indices[i]]
        top_scores = retrieved_scores[i].astype(float)
        relevance = np.array([qrels[qid].get(did, 0) for did in top_doc_ids], dtype=float)

        ndcg_y_true.append(relevance)
        ndcg_y_score.append(top_scores)

        r_at_100.append(float(any(relevance > 0)))

        relevant_positions = np.where(relevance > 0)[0]
        mrr_scores.append(1.0 / (relevant_positions[0] + 1) if len(relevant_positions) > 0 else 0.0)

    if not ndcg_y_true:
        warnings.warn("No queries with valid qrels found — cannot compute metrics.")
        return {}

    Y_true = np.vstack(ndcg_y_true)
    Y_score = np.vstack(ndcg_y_score)
    k_max = Y_true.shape[1]

    return {
        "ndcg_at_10": float(ndcg_score(Y_true, Y_score, k=min(10, k_max))),
        "ndcg_at_100": float(ndcg_score(Y_true, Y_score, k=min(100, k_max))),
        "recall_at_100": float(np.mean(r_at_100)),
        "mrr": float(np.mean(mrr_scores)),
        "num_queries_evaluated": len(ndcg_y_true),
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
                               embeddings_dir, args.force_reencode)
    print(f"Query embeddings: {tuple(q_emb.shape)} | Doc embeddings: {tuple(d_emb.shape)}")

    # Normalize InfoNCEWrapper embeddings (already normalized for ST models)
    if isinstance(model, InfoNCEWrapper):
        q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
        d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

    # Search
    print(f"\nSearching top-{args.top_k} with FAISS IndexFlatIP...")
    scores, indices = faiss_search(q_emb, d_emb, args.top_k)

    # Metrics
    metrics = compute_metrics(scores, indices, doc_ids, qrels, query_ids)

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

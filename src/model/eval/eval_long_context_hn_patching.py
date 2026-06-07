"""
Evaluate retrieval quality degradation under hard-negative document patching.

Encodes a corpus with a dual-encoder model, finds K nearest-neighbor hard-negative
documents per passage via FAISS, concatenates them into longer "patched" documents,
re-encodes, and measures retrieval metrics at each K level.

Supports BeIR datasets (loaded directly from HuggingFace) and any dataset name
recognised by the project's data factory.

Usage:
    python src/model/eval/eval_long_context_hn_patching.py \
        --model_name_or_path <path_or_hub_name> \
        --dataset_name BeIR/scifact \
        --k_values 0,1,3,5 \
        --output_dir outputs/eval/long_context_hn_patching/alephbert
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from data.long_context.patch_documents import build_patched_corpus
from model.dual_encoder.models import InfoNCEDualEncoder, InfoNCEDualEncoderConfig
from model.eval.eval_retrieval import batched_encode


# ── BeIR loader ───────────────────────────────────────────────────────────────

def load_beir_eval_data(dataset_name: str, split: str = 'test'):
    """
    Load (query_texts, passages, gold_indices) directly from a BeIR HuggingFace dataset.

    For each query we keep the single highest-scored relevant document as gold.
    Queries without a valid qrel entry are silently dropped.
    """
    print(f"Loading BeIR corpus from {dataset_name} ...")
    corpus_ds  = load_dataset(dataset_name, 'corpus')['corpus']
    queries_ds = load_dataset(dataset_name, 'queries')['queries']
    qrels_ds   = load_dataset(f'{dataset_name}-qrels')

    # Merge all qrel splits (train/validation/test) so we don't miss anything
    all_qrels = []
    for s in qrels_ds.keys():
        all_qrels.extend(qrels_ds[s])

    # Build a map: query_id → best corpus_id (highest score)
    qrel_best: dict[str, tuple[str, int]] = {}  # qid → (cid, score)
    for qrel in all_qrels:
        qid   = str(qrel['query-id'])
        cid   = str(qrel['corpus-id'])
        score = int(qrel['score'])
        if score > 0:
            if qid not in qrel_best or score > qrel_best[qid][1]:
                qrel_best[qid] = (cid, score)

    # Index corpus
    passages: list[str] = [str(doc['text']) for doc in corpus_ds]
    doc_id_to_idx: dict[str, int] = {
        str(doc['_id']): i for i, doc in enumerate(corpus_ds)
    }

    query_texts: list[str] = []
    gold_indices: list[int] = []

    for query in queries_ds:
        qid = str(query['_id'])
        if qid not in qrel_best:
            continue
        cid = qrel_best[qid][0]
        if cid not in doc_id_to_idx:
            continue
        query_texts.append(str(query['text']))
        gold_indices.append(doc_id_to_idx[cid])

    print(f"  {len(passages):,} passages,  {len(query_texts):,} queries with valid qrels")
    return query_texts, passages, gold_indices


# ── Generic loader (non-BeIR) ─────────────────────────────────────────────────

def load_factory_eval_data(dataset_name: str, task_name: str | None, split: str):
    """Load via the project's data factory (heq, heq_fact_passage_syn, etc.)."""
    from data import build_eval_dataset

    raw = build_eval_dataset(dataset_name, split=split)

    if isinstance(raw, dict):
        if task_name is None:
            task_name = next(iter(raw))
            print(f"[Info] No --task_name given; defaulting to '{task_name}'")
        if task_name not in raw:
            raise ValueError(
                f"Task '{task_name}' not in '{dataset_name}'. "
                f"Available: {list(raw.keys())}"
            )
        dataset = raw[task_name]
    else:
        dataset = raw

    passages: list[str] = []
    passage_to_idx: dict[str, int] = {}
    query_texts: list[str] = []
    gold_indices: list[int] = []

    for item in dataset:
        q = item['anchor_text']
        p = item['positive_text']
        if p not in passage_to_idx:
            passage_to_idx[p] = len(passages)
            passages.append(p)
        query_texts.append(q)
        gold_indices.append(passage_to_idx[p])

    return query_texts, passages, gold_indices


# ── Metrics ───────────────────────────────────────────────────────────────────

def retrieval_metrics(q_emb: torch.Tensor, d_emb: torch.Tensor,
                      gold_indices: list, ks=(1, 5, 10)) -> dict:
    sim   = torch.matmul(q_emb, d_emb.t()).cpu().numpy()  # (Q, D)
    ranks = []
    for i, gi in enumerate(gold_indices):
        order = np.argsort(sim[i])[::-1]
        rank  = int(np.where(order == gi)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    out = {
        'acc@1': float(np.mean(ranks == 1)),
        'mrr':   float(np.mean(1.0 / ranks)),
    }
    for k in ks:
        out[f'recall@{k}'] = float(np.mean(ranks <= k))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"\nLoading model from {args.model_name_or_path} ...")
    if args.pretrain and os.path.isdir(args.model_name_or_path):
        config = InfoNCEDualEncoderConfig.from_pretrained(args.model_name_or_path)
        model  = InfoNCEDualEncoder.from_pretrained(args.model_name_or_path, config=config)
    else:
        config = InfoNCEDualEncoderConfig(
            query_model_name=args.model_name_or_path,
            doc_model_name=args.model_name_or_path,
            pooling='cls', temperature=0.05,
        )
        model = InfoNCEDualEncoder(config)
    model = model.to(device).eval()

    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer_q = AutoTokenizer.from_pretrained(tok_path)
    tokenizer_d = AutoTokenizer.from_pretrained(tok_path)

    # Load data
    print(f"\nLoading dataset: {args.dataset_name} ...")
    if args.dataset_name.startswith('BeIR/'):
        queries, passages, gold_indices = load_beir_eval_data(
            args.dataset_name, split=args.split)
    else:
        queries, passages, gold_indices = load_factory_eval_data(
            args.dataset_name, args.task_name, args.split)

    # Encode queries (fixed across all K)
    print(f"\nEncoding {len(queries):,} queries ...")
    q_emb = batched_encode(model=model, encoder=model.query_encoder,
                           tokenizer=tokenizer_q, texts=queries,
                           device=device, batch_size=args.batch_size,
                           max_length=args.max_length)
    q_emb = F.normalize(q_emb.float(), p=2, dim=-1)

    # Encode original passages (used for K=0 baseline and for FAISS NN search)
    print(f"Encoding {len(passages):,} passages ...")
    d_emb_orig = batched_encode(model=model, encoder=model.doc_encoder,
                                tokenizer=tokenizer_d, texts=passages,
                                device=device, batch_size=args.batch_size,
                                max_length=args.max_length)
    d_emb_orig = F.normalize(d_emb_orig.float(), p=2, dim=-1)

    k_values = [int(k) for k in args.k_values.split(',')]
    all_results: dict[int, dict] = {}

    for k in k_values:
        print(f"\n── K={k} ({'baseline' if k == 0 else f'{k} hard neg(s) per doc'}) ──")
        patched_texts, _ = build_patched_corpus(
            documents=passages,
            embeddings=d_emb_orig,
            k=k,
            positive_position=args.positive_position,
            seed=args.seed,
        )
        orig_avg  = np.mean([len(p) for p in passages])
        patch_avg = np.mean([len(t) for t in patched_texts])
        print(f"  Avg doc length: {patch_avg:,.0f} chars  (orig: {orig_avg:,.0f})")

        if k == 0:
            d_emb = d_emb_orig
        else:
            print(f"  Re-encoding patched corpus ...")
            d_emb = batched_encode(model=model, encoder=model.doc_encoder,
                                   tokenizer=tokenizer_d, texts=patched_texts,
                                   device=device, batch_size=args.batch_size,
                                   max_length=args.max_length)
            d_emb = F.normalize(d_emb.float(), p=2, dim=-1)

        metrics = retrieval_metrics(q_emb, d_emb, gold_indices)
        all_results[k] = metrics
        for key, val in metrics.items():
            print(f"  {key:12s}: {val:.4f}")

    # Save results
    dataset_slug = args.dataset_name.replace('/', '_').replace('-', '_')
    task_slug    = (args.task_name or '').lower()
    stem         = f"{dataset_slug}_{task_slug}".rstrip('_')
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file     = output_dir / f"{stem}.json"

    payload = {
        'model':             args.model_name_or_path,
        'dataset':           args.dataset_name,
        'task':              args.task_name,
        'split':             args.split,
        'num_queries':       len(queries),
        'num_passages':      len(passages),
        'k_values':          k_values,
        'positive_position': args.positive_position,
        'results':           {str(k): v for k, v in all_results.items()},
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {out_file}")

    # Summary table
    metric_keys = list(next(iter(all_results.values())).keys())
    header = f"{'K':>4}  " + "  ".join(f"{m:>10}" for m in metric_keys)
    print(f"\n── Summary: {args.dataset_name} ──")
    print(header)
    print('-' * len(header))
    for k, m in all_results.items():
        print(f"{k:>4}  " + "  ".join(f"{m[key]:>10.4f}" for key in metric_keys))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Retrieval eval under hard-negative document patching.")
    p.add_argument('--model_name_or_path', required=True)
    p.add_argument('--tokenizer_name_or_path', default=None)
    p.add_argument('--pretrain', action='store_true',
                   help="Load from a trained checkpoint directory")
    p.add_argument('--dataset_name', required=True,
                   help="BeIR/scifact, BeIR/nfcorpus, etc., or a factory dataset name")
    p.add_argument('--task_name', default=None,
                   help="Required for factory datasets that return multiple tasks")
    p.add_argument('--split', default='test')
    p.add_argument('--k_values', default='0,1,3,5',
                   help="Comma-separated hard-negative counts (default: 0,1,3,5)")
    p.add_argument('--positive_position', default='random',
                   choices=['first', 'last', 'random'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--output_dir', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())

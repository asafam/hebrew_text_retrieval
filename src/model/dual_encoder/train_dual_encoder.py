import argparse
import json
import os
from glob import glob
import shutil
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, AutoConfig, PreTrainedModel
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from data.heq import HeQDatasetBuilder, HeQTranslatedDatasetBuilder
from data.squad_v2 import SquadV2DatasetBuilder
from model.dual_encoder.models import InfoNCEDualEncoder, InfoNCEDualEncoderConfig


def load_beir_hard_negatives(corpora_root=None, corpus_dirs=None, val_size=0.05, seed=42):
    """Load pre-mined hard negatives from hard_negatives_train.jsonl files.

    Returns DatasetDict with 'train' and 'validation' splits.
    Fields: 'query', 'positive', 'hard_neg_0', 'hard_neg_1', ... (one per hard negative).
    Falls back to positive-only if no hard_negatives_train.jsonl found.
    """
    if corpus_dirs is None:
        if corpora_root is None:
            raise ValueError("Provide corpora_root or corpus_dirs")
        train_tsvs = glob(os.path.join(corpora_root, "**/qrels/train.tsv"), recursive=True)
        corpus_dirs = [os.path.dirname(os.path.dirname(f)) for f in sorted(train_tsvs)]

    queries_all, positives_all, hard_negs_all = [], [], []
    num_hard_negs = 0

    for corpus_dir in corpus_dirs:
        hn_path = os.path.join(corpus_dir, "hard_negatives_train.jsonl")
        if not os.path.exists(hn_path):
            print(f"  WARNING: {hn_path} not found — skipping {os.path.basename(corpus_dir)}")
            continue
        loaded = 0
        with open(hn_path) as f:
            for line in f:
                record = json.loads(line)
                queries_all.append(record["query"])
                positives_all.append(record["positive"])
                hard_negs_all.append(record["hard_negs"])
                num_hard_negs = max(num_hard_negs, len(record["hard_negs"]))
                loaded += 1
        print(f"  {os.path.basename(corpus_dir)}: {loaded:,} pairs with hard negatives")

    if not queries_all:
        print("No hard negatives found — falling back to positive-only training")
        return load_beir_dataset(corpora_root=corpora_root, corpus_dirs=corpus_dirs,
                                  val_size=val_size, seed=seed)

    print(f"Hard negative total: {len(queries_all):,} pairs, {num_hard_negs} negs/query")

    data = {"query": queries_all, "document": positives_all}
    for i in range(num_hard_negs):
        data[f"hard_neg_{i}"] = [hn[i] if i < len(hn) else "" for hn in hard_negs_all]

    dataset = Dataset.from_dict(data)
    splits = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def load_beir_dataset(corpora_root=None, corpus_dirs=None, val_size=0.05, seed=42):
    """Load Hebrew translated BeIR corpora as (query, document) training pairs.

    Discovers all subdirectories with qrels/train.tsv under corpora_root, or
    uses an explicit list via corpus_dirs. Returns a DatasetDict with
    'train' and 'validation' splits using field names 'query' and 'document'.
    """
    if corpus_dirs is None:
        if corpora_root is None:
            raise ValueError("Provide --beir_corpora_root or corpus_dirs")
        train_tsvs = glob(os.path.join(corpora_root, "**/qrels/train.tsv"), recursive=True)
        corpus_dirs = [os.path.dirname(os.path.dirname(f)) for f in sorted(train_tsvs)]

    queries_all, docs_all = [], []

    for corpus_dir in corpus_dirs:
        corpus_path = os.path.join(corpus_dir, "corpus.jsonl")
        queries_path = os.path.join(corpus_dir, "queries.jsonl")
        qrels_path = os.path.join(corpus_dir, "qrels", "train.tsv")
        if not all(os.path.exists(p) for p in [corpus_path, queries_path, qrels_path]):
            continue

        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                title = (doc.get("title") or "").strip()
                text = (doc.get("text") or "").strip()
                corpus[doc["_id"]] = (title + " " + text).strip() if title else text

        queries = {}
        with open(queries_path) as f:
            for line in f:
                q = json.loads(line)
                queries[q["_id"]] = q["text"]

        loaded = 0
        with open(qrels_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    qid, docid, score = parts[0], parts[1], int(float(parts[2]))
                except ValueError:
                    continue
                if score > 0 and qid in queries and docid in corpus:
                    queries_all.append(queries[qid])
                    docs_all.append(corpus[docid])
                    loaded += 1

        print(f"  {os.path.basename(corpus_dir)}: {loaded:,} training pairs loaded")

    print(f"BeIR total: {len(queries_all):,} training pairs from {len(corpus_dirs)} dataset(s)")
    dataset = Dataset.from_dict({"query": queries_all, "document": docs_all})
    splits = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def get_dataset(dataset_name: str, **kwargs):
    if dataset_name.lower() == "beir_hebrew_hn":
        return load_beir_hard_negatives(
            corpora_root=kwargs.get("beir_corpora_root"),
            corpus_dirs=kwargs.get("corpus_dirs"),
        )
    elif dataset_name.lower() == "beir_hebrew":
        return load_beir_dataset(
            corpora_root=kwargs.get("beir_corpora_root"),
            corpus_dirs=kwargs.get("corpus_dirs"),
        )
    elif dataset_name.lower() == "heq":
        dataset_builder = HeQDatasetBuilder(query_field=kwargs.get("query_field"),
                                            document_field=kwargs.get("document_field"))
        dataset = dataset_builder.build_dataset(filter_empty_answers=True)
        return dataset
    elif dataset_name.lower() == "heq_translated":
        dataset_builder = HeQTranslatedDatasetBuilder(queries_base_path='data/heq_translated',
                                                      query_field=kwargs.get("query_field"),
                                                      document_field=kwargs.get("document_field"))
        dataset = dataset_builder.build_dataset(filter_empty_answers=True)
        return dataset
    elif dataset_name.lower() == "squad_v2":
        dataset_builder = SquadV2DatasetBuilder(queries_base_path='data/squad_v2',
                                                query_field=kwargs.get("query_field"),
                                                document_field=kwargs.get("document_field"))
        dataset = dataset_builder.build_dataset()
        return dataset


def preprocess(
        example,
        tokenizer_q,
        tokenizer_d,
        query_field='query',
        document_field='context',
        truncation=True,
        padding="max_length",
        max_length=1024
    ):
    q = tokenizer_q(example[query_field], truncation=truncation, padding=padding, max_length=max_length)
    d = tokenizer_d(example[document_field], truncation=truncation, padding=padding, max_length=max_length)
    result = {
        "q_input_ids": q['input_ids'],
        "q_attention_mask": q['attention_mask'],
        "d_input_ids": d['input_ids'],
        "d_attention_mask": d['attention_mask'],
    }
    # Handle hard negatives: hard_neg_0, hard_neg_1, ...
    i = 0
    while f"hard_neg_{i}" in example and example[f"hard_neg_{i}"]:
        hn = tokenizer_d(example[f"hard_neg_{i}"], truncation=truncation, padding=padding, max_length=max_length)
        result[f"hn_{i}_input_ids"] = hn['input_ids']
        result[f"hn_{i}_attention_mask"] = hn['attention_mask']
        i += 1
    return result


def collate_fn(batch):
    result = {
        "query_input_ids": torch.tensor([item["q_input_ids"] for item in batch]),
        "query_attention_mask": torch.tensor([item["q_attention_mask"] for item in batch]),
        "doc_input_ids": torch.tensor([item["d_input_ids"] for item in batch]),
        "doc_attention_mask": torch.tensor([item["d_attention_mask"] for item in batch]),
    }
    # Stack hard negatives into (B, K, seq_len) tensors if present
    i = 0
    while f"hn_{i}_input_ids" in batch[0]:
        result[f"neg_input_ids"] = torch.stack([
            torch.tensor(item[f"hn_{i}_input_ids"]) for item in batch
        ], dim=0).unsqueeze(1) if i == 0 else torch.cat([
            result["neg_input_ids"],
            torch.stack([torch.tensor(item[f"hn_{i}_input_ids"]) for item in batch], dim=0).unsqueeze(1)
        ], dim=1)
        result[f"neg_attention_mask"] = torch.stack([
            torch.tensor(item[f"hn_{i}_attention_mask"]) for item in batch
        ], dim=0).unsqueeze(1) if i == 0 else torch.cat([
            result["neg_attention_mask"],
            torch.stack([torch.tensor(item[f"hn_{i}_attention_mask"]) for item in batch], dim=0).unsqueeze(1)
        ], dim=1)
        i += 1
    return result


def get_latest_checkpoint(output_dir):
    # Find all checkpoint folders
    checkpoint_dirs = sorted(
        glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1
    )
    # Keep only those with trainer_state.json present
    valid_checkpoints = [
        c for c in checkpoint_dirs if os.path.isfile(os.path.join(c, "trainer_state.json"))
    ]
    return valid_checkpoints[-1] if valid_checkpoints else None


def main(
    dataset_name: str,
    query_model_name: str,
    doc_model_name: str,
    output_dir: str,
    query_field: str = 'question',
    document_field: str = 'context',
    beir_corpora_root: str = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate=2e-5,
    bf16: bool = True,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    max_length=1024,
    pooling: str = "mean",
    remove_to_overwrite: bool = False,
    force: bool = True
):
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Do you wish to overwrite it? (Y/n)")
        response = input().strip().lower() if not force else 'y'

        if response == 'y':
            print(f"Overwriting output directory {output_dir}.")
            if remove_to_overwrite:
                shutil.rmtree(output_dir)
        else:
            print("Exiting without training.")
                
    # Load .env file
    load_dotenv()

    # Access the token
    hf_token = os.getenv("HF_TOKEN")
    print(f"Using Hugging Face token: {hf_token}")

    # Authenticate with Hugging Face
    # login(hf_token)

    tokenizer_q = AutoTokenizer.from_pretrained(query_model_name, trust_remote_code=True)
    tokenizer_d = AutoTokenizer.from_pretrained(doc_model_name, trust_remote_code=True)
    
    dataset = get_dataset(dataset_name, query_field=query_field, document_field=document_field,
                          beir_corpora_root=beir_corpora_root)
    processed = dataset.map(lambda sample: preprocess(sample,
                                                      tokenizer_q,
                                                      tokenizer_d,
                                                      query_field=query_field,
                                                      document_field=document_field,
                                                      max_length=max_length))

    config = InfoNCEDualEncoderConfig(query_model_name=query_model_name, 
                                      doc_model_name=doc_model_name, 
                                      query_tokenizer_path=tokenizer_q.name_or_path,
                                      doc_tokenizer_path=tokenizer_d.name_or_path,
                                      pooling=pooling, 
                                      temperature=0.05)
    model = InfoNCEDualEncoder(config)

    eval_split = processed.get('validation') or processed.get('test')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        remove_unused_columns=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy if eval_split is not None else "no",
        eval_steps=eval_steps,
        report_to="wandb",
        run_name=f"dual_encoder_infonce_{dataset_name}_{query_model_name.replace('/', '-')}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed['train'],
        eval_dataset=eval_split,
        data_collator=collate_fn,
    )

    # Find latest checkpoint in output_dir (if any)
    latest_checkpoint = get_latest_checkpoint(output_dir)

    if latest_checkpoint is not None:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("No checkpoint found, starting fresh training.")
        trainer.train()

    # Save the model, tokenizer, and config
    model_dir = os.path.join(output_dir, "model")
    tokenizer_q_dir = os.path.join(model_dir, "tokenizer_query")
    tokenizer_d_dir = os.path.join(model_dir, "tokenizer_doc")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(tokenizer_q_dir).mkdir(parents=True, exist_ok=True)
    Path(tokenizer_d_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer_q.save_pretrained(tokenizer_q_dir)
    tokenizer_d.save_pretrained(tokenizer_d_dir)
    config.save_pretrained(model_dir)

if __name__ == "__main__":
    argparse.ArgumentParser(description="Train a dual encoder model with InfoNCE loss on HeQ dataset.")
    parser = argparse.ArgumentParser(description="Train a dual encoder model with InfoNCE loss on HeQ dataset.")
    parser.add_argument("--dataset_name", type=str, default="heq", help="Dataset name: heq, heq_translated, squad_v2, beir_hebrew")
    parser.add_argument("--beir_corpora_root", type=str, default=None, help="Root dir containing BeIR corpus subdirs (used with --dataset_name beir_hebrew)")
    parser.add_argument("--query_model_name", type=str, default="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250522_1841", help="Query model name or path")
    parser.add_argument("--doc_model_name", type=str, default="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250522_1841", help="Document model name or path (optional, defaults to query model if not provided)"),
    parser.add_argument("--output_dir", type=str, default="./outputs/models/dual_encoder/dual_encoder_infonce_heq", help="Output directory for model and logs")    
    parser.add_argument("--query_field", type=str, default="query", help="Field name for query in the dataset")
    parser.add_argument("--document_field", type=str, default="context", help="Field name for document in the dataset")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * accum)")
    parser.add_argument("--bf16", action='store_true', default=True, help="Use bf16 mixed precision (recommended for A100)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Fraction of training steps used for LR warmup")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler: linear, cosine, cosine_with_restarts, etc.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every X steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization")
    parser.add_argument("--pooling", type=str, default="mean", help="Pooling strategy: 'cls' or 'mean'.")
    parser.add_argument("--force", action='store_true', help="Remove output directory if it exists before training")
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset_name,
        query_model_name=args.query_model_name,
        doc_model_name=args.doc_model_name,
        output_dir=args.output_dir,
        query_field=args.query_field,
        document_field=args.document_field,
        beir_corpora_root=args.beir_corpora_root,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        max_length=args.max_length,
        pooling=args.pooling,
        force=args.force,
    )
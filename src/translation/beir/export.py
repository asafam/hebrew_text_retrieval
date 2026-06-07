"""
Export translated BeIR CSVs to HuggingFace-ready JSONL format.

Output per dataset:
    beir/
        corpus.jsonl   — {"_id", "title" (HE), "title_en" (EN), "text" (HE), "text_en" (EN)}
        queries.jsonl  — {"_id", "text" (HE), "text_en" (EN)}
        qrels/
            test.tsv   — query-id\tcorpus-id\tscore  (no header, one file per split)
        metadata.json
"""

import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from datasets import load_dataset


def export_to_beir_jsonl(
    translated_queries_csv: str,
    translated_documents_csv: str,
    dataset_name: str,
    output_dir: str,
    run_metadata: dict,
    segment_separator: str = " ",
    force: bool = False,
) -> None:
    """
    Convert translated CSV files into BeIR-compatible JSONL files.

    Args:
        translated_queries_csv:   Path to translated queries CSV.
        translated_documents_csv: Path to translated documents CSV.
        dataset_name:             HuggingFace BeIR dataset name (e.g. "BeIR/msmarco").
        output_dir:               Directory to write beir/ outputs into.
        run_metadata:             Dict with run_id, model_name, prompt_file, etc.
        segment_separator:        String used to join translated document segments.
        force:                    Overwrite existing outputs if True.
    """
    beir_dir = os.path.join(output_dir, "beir")
    corpus_path = os.path.join(beir_dir, "corpus.jsonl")
    queries_path = os.path.join(beir_dir, "queries.jsonl")
    qrels_dir = os.path.join(beir_dir, "qrels")
    metadata_path = os.path.join(beir_dir, "metadata.json")

    if not force and os.path.exists(corpus_path) and os.path.exists(queries_path):
        print(f"  [export] Skipping {dataset_name} — BeIR outputs already exist at {beir_dir}")
        return

    os.makedirs(beir_dir, exist_ok=True)
    os.makedirs(qrels_dir, exist_ok=True)

    corpus_stats = _export_corpus(translated_documents_csv, corpus_path, segment_separator)
    queries_stats = _export_queries(translated_queries_csv, queries_path)
    _export_qrels(dataset_name, qrels_dir)

    full_metadata = {
        **run_metadata,
        "dataset_name": dataset_name,
        "exported_at": datetime.now().isoformat(),
        "corpus_stats": corpus_stats,
        "queries_stats": queries_stats,
    }
    _write_run_metadata(beir_dir, full_metadata)

    print(
        f"  [export] {dataset_name}: "
        f"{corpus_stats['num_documents']} docs, {queries_stats['num_queries']} queries → {beir_dir}"
    )


def _export_corpus(
    translated_documents_csv: str,
    output_path: str,
    segment_separator: str,
) -> dict:
    """
    Read translated documents CSV, rejoin segments per document, write corpus.jsonl.

    Each record:  {"_id", "title" (HE), "title_en" (EN), "text" (HE), "text_en" (EN)}
    """
    df = pd.read_csv(translated_documents_csv, encoding="utf-8")

    missing = int(df["translation"].isnull().sum())
    if missing:
        print(f"  [export] Warning: {missing} document segments missing translation — skipping.")
    df = df.dropna(subset=["translation"])

    num_segments = len(df)
    records = []

    for doc_id, group in df.groupby("_id", sort=False):
        if "segment_id" in group.columns:
            group = group.sort_values("segment_id")

        # Original English fields
        title_en = str(group["title"].iloc[0]) if "title" in group.columns and pd.notna(group["title"].iloc[0]) else ""
        text_en = segment_separator.join(str(t) if pd.notna(t) else "" for t in group["segment_text"]) if "segment_text" in group.columns else ""

        # Hebrew translations
        text_he = segment_separator.join(str(t) if pd.notna(t) else "" for t in group["translation"])

        # Title translation: use title_translation column if available, else fall back to English
        has_title_translation = "title_translation" in group.columns
        title_he_val = group["title_translation"].iloc[0] if has_title_translation else None
        title_he = str(title_he_val) if has_title_translation and pd.notna(title_he_val) and str(title_he_val).strip() else title_en

        records.append({
            "_id": str(doc_id),
            "title": title_he,
            "title_en": title_en,
            "text": text_he,
            "text_en": text_en,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"num_documents": len(records), "num_segments": num_segments, "missing_translations": missing}


def _export_queries(
    translated_queries_csv: str,
    output_path: str,
) -> dict:
    """
    Read translated queries CSV, write queries.jsonl.

    Each record:  {"_id", "text" (HE), "text_en" (EN)}
    """
    df = pd.read_csv(translated_queries_csv, encoding="utf-8")

    missing = int(df["translation"].isnull().sum())
    if missing:
        print(f"  [export] Warning: {missing} queries missing translation — skipping.")
    df = df.dropna(subset=["translation"])

    records = []
    for _, row in df.iterrows():
        text_en = str(row["text"]) if "text" in df.columns and pd.notna(row.get("text")) else ""
        records.append({
            "_id": str(row["_id"]),
            "text": str(row["translation"]),
            "text_en": text_en,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"num_queries": len(records), "missing_translations": missing}


def _export_qrels(dataset_name: str, qrels_dir: str) -> None:
    """
    Download qrels from HuggingFace and write one JSONL per split.
    Format:  {"query-id": "...", "corpus-id": "...", "score": N}
    JSONL is used (not TSV) because HF datasets-server reliably parses JSONL
    regardless of whether IDs are strings, integers, or hashes.
    """
    try:
        qrels_dataset = load_dataset(f"{dataset_name}-qrels")
    except Exception as e:
        print(f"  [export] Warning: could not load qrels for {dataset_name}: {e}")
        return

    for split_name, split_data in qrels_dataset.items():
        jsonl_path = os.path.join(qrels_dir, f"{split_name}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in split_data:
                query_id = row.get("query-id", row.get("query_id", ""))
                corpus_id = row.get("corpus-id", row.get("corpus_id", ""))
                score = row.get("score", 1)
                f.write(json.dumps({"query-id": str(query_id),
                                    "corpus-id": str(corpus_id),
                                    "score": int(score)},
                                   ensure_ascii=False) + "\n")
        print(f"  [export] Wrote qrels/{split_name}.jsonl ({len(split_data)} rows)")


def _write_run_metadata(output_dir: str, metadata: dict) -> None:
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

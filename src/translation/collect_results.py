"""
Walks the experiment output tree and assembles evaluated translation CSVs
into a master CSV with one row per evaluated translation.

Two path structures are supported:

  Main experiment:
    <results_dir>/<dataset_slug>/<translation_model>/<prompt_slug>/evaluations/<judge_model>/
        queries_translated_evaluated.csv
        documents_translated_evaluated.csv

  Eval prompt calibration (--calibration flag):
    <results_dir>/<dataset_slug>/<translation_model>/<prompt_slug>/eval_calibration/<eval_prompt_slug>/<judge_model>/
        queries_translated_evaluated.csv
        documents_translated_evaluated.csv

All experimental factors are extracted from the directory path.

Usage:
  # Main experiment
  python src/translation/collect_results.py \
      --results_dir outputs/translation/BeIR/candidates \
      --output_path outputs/translation/BeIR/results_master.csv

  # Eval prompt calibration
  python src/translation/collect_results.py \
      --results_dir outputs/translation/BeIR/candidates \
      --output_path outputs/translation/BeIR/results_eval_calibration.csv \
      --calibration
"""

import argparse
import os
import re
import pandas as pd
from pathlib import Path


DATASET_CATEGORY = {
    "BeIR_msmarco":        "Misc",
    "BeIR_fever":          "Fact checking",
    "BeIR_climate-fever":  "Fact checking",
    "BeIR_scifact":        "Fact checking",
    "BeIR_scidocs":        "Citation-Prediction",
    "BeIR_quora":          "Duplicate question retrieval",
    "BeIR_arguana":        "Argument retrieval",
    "BeIR_nq":             "Question answering",
    "BeIR_hotpotqa":       "Question answering",
    "BeIR_trec-covid":     "Bio-medical IR",
    "BeIR_nfcorpus":       "Bio-medical IR",
    "BeIR_dbpedia-entity": "Entity retrieval",
}

# Eval prompt slug → the category it was designed for
EVAL_PROMPT_DESIGN_TARGET = {
    "general":   "Misc / Fact checking / Entity retrieval",
    "technical": "Bio-medical IR / Citation-Prediction / scifact",
    "qa":        "Question answering / Argument retrieval / Duplicate questions",
}

def _parse_main_path(rel: str) -> dict | None:
    """Parse dataset/translation_model.../prompt_slug/evaluations/judge_model.../filename.csv
    translation_model and judge_model may contain slashes (e.g. moonshotai/kimi-k2.6)."""
    anchor = "/evaluations/"
    idx = rel.find(anchor)
    if idx == -1:
        return None
    left = rel[:idx].split("/")     # [dataset, *translation_model_parts, prompt_slug]
    right = rel[idx + len(anchor):].split("/")  # [*judge_model_parts, filename]
    if len(left) < 3 or len(right) < 2:
        return None
    return {
        "dataset":           left[0],
        "translation_model": "/".join(left[1:-1]),
        "prompt_slug":       left[-1],
        "judge_model":       "/".join(right[:-1]),
        "filename":          right[-1],
    }


def _parse_calibration_path(rel: str) -> dict | None:
    """Parse dataset/translation_model.../prompt_slug/eval_calibration/eval_prompt_slug/judge_model.../filename.csv"""
    anchor = "/eval_calibration/"
    idx = rel.find(anchor)
    if idx == -1:
        return None
    left = rel[:idx].split("/")
    right = rel[idx + len(anchor):].split("/")
    # right: [eval_prompt_slug, *judge_model_parts, filename]
    if len(left) < 3 or len(right) < 3:
        return None
    return {
        "dataset":           left[0],
        "translation_model": "/".join(left[1:-1]),
        "prompt_slug":       left[-1],
        "eval_prompt_slug":  right[0],
        "judge_model":       "/".join(right[1:-1]),
        "filename":          right[-1],
    }


def _add_length_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "segment_id" in df.columns and "_id" in df.columns:
        max_seg = df.groupby("_id")["segment_id"].max().rename("max_segment_id")
        df = df.merge(max_seg, on="_id", how="left")
        df["text_length_bucket"] = df["max_segment_id"].apply(
            lambda x: "long (>1 segment)" if x >= 1 else "short (1 segment)"
        )
        df.drop(columns=["max_segment_id"], inplace=True)
    else:
        df["text_length_bucket"] = "unknown"
    return df


def collect(results_dir: str, output_path: str, calibration: bool = False) -> pd.DataFrame:
    results_dir = Path(results_dir)
    frames = []

    csv_files = list(results_dir.rglob("*_evaluated.csv"))
    # Exclude calibration files from main collection and vice versa
    if calibration:
        csv_files = [p for p in csv_files if "eval_calibration" in p.as_posix()]
    else:
        csv_files = [p for p in csv_files if "eval_calibration" not in p.as_posix()]

    print(f"Found {len(csv_files)} evaluated CSV files ({'calibration' if calibration else 'main'} mode)")

    parse = _parse_calibration_path if calibration else _parse_main_path

    for csv_path in csv_files:
        rel = csv_path.relative_to(results_dir).as_posix()
        m = parse(rel)
        if not m:
            print(f"  Skipping (path does not match expected structure): {rel}")
            continue

        dataset_slug      = m["dataset"]
        translation_model = m["translation_model"]
        prompt_slug       = m["prompt_slug"]
        judge_model       = m["judge_model"]
        filename          = m["filename"]
        text_type         = "query" if "queries" in filename else "document"

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}")
            continue

        if df.empty or "score" not in df.columns:
            print(f"  Skipping (empty or missing score column): {rel}")
            continue

        df["dataset_slug"]      = dataset_slug
        df["category"]          = DATASET_CATEGORY.get(dataset_slug, "Unknown")
        df["translation_model"] = translation_model
        df["prompt_slug"]       = prompt_slug
        df["judge_model"]       = judge_model
        df["text_type"]         = text_type
        df = _add_length_bucket(df)

        if calibration:
            eval_prompt_slug = m.group("eval_prompt_slug")
            df["eval_prompt_slug"]          = eval_prompt_slug
            df["eval_prompt_design_target"] = EVAL_PROMPT_DESIGN_TARGET.get(eval_prompt_slug, "Unknown")

        frames.append(df)

    if not frames:
        print("No results found.")
        return pd.DataFrame()

    master = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    master.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Master CSV written to {output_path}  ({len(master):,} rows)")
    return master


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--calibration", action="store_true",
                        help="Collect eval prompt calibration results instead of main experiment results.")
    args = parser.parse_args()
    collect(args.results_dir, args.output_path, calibration=args.calibration)


if __name__ == "__main__":
    main()

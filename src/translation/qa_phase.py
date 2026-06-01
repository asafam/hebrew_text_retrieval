"""
Mid-phase translation QA script.

Checks translation coverage and scores a random sample via LLM-as-a-judge,
then compares against the baseline from the 25-sample experiment.

Exit codes:
  0 — all datasets pass (coverage ≥ 95%, scores within threshold)
  1 — coverage warnings only (no score degradation)
  2 — score degradation detected → halt the pipeline

Usage:
    python src/translation/qa_phase.py \\
        --phase-run-dir outputs/beir_translation/full_corpus/full_corpus_p1_small \\
        --baseline-csv  outputs/translation/BeIR/results_translation_eval.csv \\
        --baseline-model gpt-5.4-mini \\
        --baseline-prompt zeroshot_nocontext \\
        --judge-model claude-sonnet-4-6 \\
        --sample-size 25 \\
        --output-report outputs/beir_translation/full_corpus/qa_p1.json
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Dataset category → evaluation prompt
DATASET_EVAL_PROMPTS = {
    "BeIR_msmarco":        "prompts/translation/api/evaluation/translation_evaluation_nogold_v20260531.yaml",
    "BeIR_fever":          "prompts/translation/api/evaluation/translation_evaluation_nogold_v20260531.yaml",
    "BeIR_climate-fever":  "prompts/translation/api/evaluation/translation_evaluation_nogold_v20260531.yaml",
    "BeIR_scifact":        "prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20260531.yaml",
    "BeIR_scidocs":        "prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20260531.yaml",
    "BeIR_quora":          "prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20260531.yaml",
    "BeIR_arguana":        "prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20260531.yaml",
    "BeIR_nq":             "prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20260531.yaml",
    "BeIR_hotpotqa":       "prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20260531.yaml",
    "BeIR_trec-covid":     "prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20260531.yaml",
    "BeIR_nfcorpus":       "prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20260531.yaml",
    "BeIR_dbpedia-entity": "prompts/translation/api/evaluation/translation_evaluation_nogold_v20260531.yaml",
}

# Degradation thresholds
DEGRADATION_Z_THRESHOLD = 1.5   # sample_mean < baseline_mean - Z * baseline_std
DEGRADATION_ABS_THRESHOLD = 0.5  # absolute point drop on 1–5 scale
COVERAGE_WARN_THRESHOLD = 0.95   # warn if < 95% of rows are translated


def _dataset_slug_from_path(path: str) -> str:
    """Extract dataset slug from a translated CSV path like .../BeIR_nfcorpus/queries_translated.csv"""
    parts = Path(path).parts
    for part in reversed(parts):
        if part.startswith("BeIR_"):
            return part
    return Path(path).parent.name


def check_coverage(phase_run_dir: str) -> dict:
    """Return coverage stats per (dataset_slug, text_type). Warns if < COVERAGE_WARN_THRESHOLD."""
    results = {}
    for csv_path in Path(phase_run_dir).rglob("*_translated.csv"):
        slug = _dataset_slug_from_path(str(csv_path))
        text_type = "query" if "queries" in csv_path.name else "document"
        df = pd.read_csv(csv_path, encoding="utf-8")
        total = len(df)
        translated = df["translation"].notna().sum() if "translation" in df.columns else 0
        coverage = translated / total if total > 0 else 0.0
        key = (slug, text_type)
        results[key] = {"total": total, "translated": int(translated), "coverage": coverage, "path": str(csv_path)}
        status = "OK" if coverage >= COVERAGE_WARN_THRESHOLD else "WARN"
        print(f"  [{status}] {slug} {text_type}: {translated:,}/{total:,} ({100*coverage:.1f}%)")
    return results


def _evaluate_sample(translated_csv: str, dataset_slug: str, text_type: str,
                     judge_model: str, sample_size: int, workers: int = 4) -> pd.DataFrame:
    """Sample rows from translated CSV and run LLM-as-a-judge. Returns evaluated DataFrame."""
    from translation.api.evaluate_translations import run_evaluate_translations

    df = pd.read_csv(translated_csv, encoding="utf-8")
    df = df[df["translation"].notna()]
    if df.empty:
        return pd.DataFrame()

    sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Write sample to a temp file so run_evaluate_translations can read it
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as tmp:
        sample.to_csv(tmp, index=False)
        tmp_path = tmp.name

    prompt_file = DATASET_EVAL_PROMPTS.get(dataset_slug,
        "prompts/translation/api/evaluation/translation_evaluation_nogold_v20260531.yaml")

    # Output goes to a temp dir
    tmp_out_dir = tempfile.mkdtemp()

    try:
        evaluated_df = run_evaluate_translations(
            source_file_path=tmp_path,
            output_dir=tmp_out_dir,
            gold_file_path=None,
            prompt_file_name=prompt_file,
            model_name=judge_model,
            limit=0,
            sample=0.0,
            force=True,
            workers=workers,
            english_key="text" if text_type == "query" else "segment_text",
            hebrew_key="translation",
        )
    finally:
        os.unlink(tmp_path)

    return evaluated_df if evaluated_df is not None else pd.DataFrame()


def load_baseline(baseline_csv: str, baseline_model: str, baseline_prompt: str,
                  judge_model: str) -> pd.DataFrame:
    """Load and filter the baseline master CSV to the target model/prompt/judge combo."""
    df = pd.read_csv(baseline_csv, encoding="utf-8")
    mask = (
        (df["translation_model"] == baseline_model) &
        (df["prompt_slug"] == baseline_prompt) &
        (df["judge_model"] == judge_model)
    )
    filtered = df[mask]
    if filtered.empty:
        print(f"WARNING: No baseline rows found for model={baseline_model}, "
              f"prompt={baseline_prompt}, judge={judge_model}")
    return filtered


def compare_scores(sample_mean: float, baseline_mean: float, baseline_std: float) -> tuple[bool, str]:
    """Return (is_degraded, reason). Degraded if z-score or absolute drop exceeds thresholds."""
    if baseline_std == 0:
        abs_drop = baseline_mean - sample_mean
        if abs_drop > DEGRADATION_ABS_THRESHOLD:
            return True, f"absolute drop {abs_drop:.3f} > {DEGRADATION_ABS_THRESHOLD}"
        return False, ""

    z = (baseline_mean - sample_mean) / baseline_std
    abs_drop = baseline_mean - sample_mean
    reasons = []
    if z > DEGRADATION_Z_THRESHOLD:
        reasons.append(f"z={z:.2f} > {DEGRADATION_Z_THRESHOLD}")
    if abs_drop > DEGRADATION_ABS_THRESHOLD:
        reasons.append(f"drop={abs_drop:.3f} > {DEGRADATION_ABS_THRESHOLD}")
    degraded = bool(reasons)
    return degraded, ", ".join(reasons)


def print_spot_check(translated_csv: str, n: int = 5) -> None:
    """Print N random source→translation pairs for human review."""
    df = pd.read_csv(translated_csv, encoding="utf-8")
    df = df[df["translation"].notna()]
    if df.empty:
        return
    sample = df.sample(n=min(n, len(df)), random_state=99)
    text_col = "text" if "text" in df.columns else "segment_text"
    print(f"\n  --- Spot-check ({n} random translations) ---")
    for _, row in sample.iterrows():
        src = str(row.get(text_col, ""))[:120].replace("\n", " ")
        tgt = str(row.get("translation", ""))[:120].replace("\n", " ")
        print(f"  EN: {src}")
        print(f"  HE: {tgt}")
        print()


def run_qa(phase_run_dir: str, baseline_csv: str, baseline_model: str, baseline_prompt: str,
           judge_model: str, sample_size: int, output_report: str, workers: int) -> int:
    """Main QA logic. Returns exit code (0/1/2)."""
    print("\n=== Translation QA — Coverage Check ===")
    coverage = check_coverage(phase_run_dir)

    coverage_warn = any(v["coverage"] < COVERAGE_WARN_THRESHOLD for v in coverage.values())

    print("\n=== Loading Baseline ===")
    baseline_df = load_baseline(baseline_csv, baseline_model, baseline_prompt, judge_model)
    if baseline_df.empty:
        print("Cannot compare without baseline — skipping score comparison.")
        return 1 if coverage_warn else 0

    baseline_stats = (
        baseline_df
        .groupby(["dataset_slug", "text_type"])["score"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "baseline_mean", "std": "baseline_std", "count": "baseline_n"})
    )

    print("\n=== Evaluating QA Samples ===")
    report = {"phase_run_dir": phase_run_dir, "datasets": {}}
    any_degraded = False

    for csv_path in sorted(Path(phase_run_dir).rglob("*_translated.csv")):
        slug = _dataset_slug_from_path(str(csv_path))
        text_type = "query" if "queries" in csv_path.name else "document"
        key = f"{slug}/{text_type}"
        print(f"\n  Evaluating {key}...")

        evaluated = _evaluate_sample(str(csv_path), slug, text_type,
                                     judge_model, sample_size, workers)
        if evaluated is None or evaluated.empty or "score" not in evaluated.columns:
            print(f"  SKIP: no scores returned for {key}")
            report["datasets"][key] = {"status": "no_scores"}
            continue

        valid_scores = evaluated["score"].dropna()
        if valid_scores.empty:
            print(f"  SKIP: all scores null for {key}")
            report["datasets"][key] = {"status": "no_scores"}
            continue

        sample_mean = float(valid_scores.mean())
        sample_std = float(valid_scores.std())
        sample_n = len(valid_scores)

        # Look up baseline
        idx = (slug, text_type)
        if idx not in baseline_stats.index:
            print(f"  INFO: no baseline for {key} — reporting score only")
            report["datasets"][key] = {
                "status": "no_baseline",
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "sample_n": sample_n,
            }
            print(f"  Score: {sample_mean:.3f} ± {sample_std:.3f} (n={sample_n})")
            continue

        b = baseline_stats.loc[idx]
        baseline_mean = float(b["baseline_mean"])
        baseline_std = float(b["baseline_std"])
        baseline_n = int(b["baseline_n"])

        degraded, reason = compare_scores(sample_mean, baseline_mean, baseline_std)

        status = "DEGRADED" if degraded else "OK"
        if degraded:
            any_degraded = True

        print(f"  [{status}] {key}: sample={sample_mean:.3f}±{sample_std:.3f} (n={sample_n})  "
              f"baseline={baseline_mean:.3f}±{baseline_std:.3f} (n={baseline_n})"
              + (f"  ← {reason}" if reason else ""))

        report["datasets"][key] = {
            "status": status,
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "sample_n": sample_n,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "baseline_n": baseline_n,
            "degradation_reason": reason,
        }

        if degraded:
            print_spot_check(str(csv_path))

    # Determine exit code and summary
    if any_degraded:
        exit_code = 2
        summary = "FAILED — score degradation detected. Review qa_report and spot-checks before proceeding."
    elif coverage_warn:
        exit_code = 1
        summary = "WARN — coverage below 95% in some datasets. Scores OK."
    else:
        exit_code = 0
        summary = "PASSED — all datasets within expected score range."

    report["summary"] = summary
    report["exit_code"] = exit_code

    os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== QA Result: {summary} ===")
    print(f"Report written to: {output_report}")
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Mid-phase translation QA.")
    parser.add_argument("--phase-run-dir", required=True,
                        help="Directory containing *_translated.csv files from the pipeline run.")
    parser.add_argument("--baseline-csv", required=True,
                        help="Master results CSV from the 25-sample evaluation experiment.")
    parser.add_argument("--baseline-model", default="gpt-5.4-mini",
                        help="Translation model to filter in the baseline CSV.")
    parser.add_argument("--baseline-prompt", default="zeroshot_nocontext",
                        help="Prompt slug to filter in the baseline CSV.")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6",
                        help="LLM-as-a-judge model for QA evaluation.")
    parser.add_argument("--sample-size", type=int, default=25,
                        help="Rows to sample per dataset/text_type for evaluation.")
    parser.add_argument("--output-report", required=True,
                        help="Path to write the JSON QA report.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for LLM-as-a-judge calls.")
    args = parser.parse_args()

    exit_code = run_qa(
        phase_run_dir=args.phase_run_dir,
        baseline_csv=args.baseline_csv,
        baseline_model=args.baseline_model,
        baseline_prompt=args.baseline_prompt,
        judge_model=args.judge_model,
        sample_size=args.sample_size,
        output_report=args.output_report,
        workers=args.workers,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

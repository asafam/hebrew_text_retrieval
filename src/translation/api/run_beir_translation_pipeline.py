"""
BeIR Translation Pipeline Orchestrator.

Reads a YAML config and runs the full translation pipeline for each BeIR dataset:
  Phase 1: Build candidate CSVs (queries.csv, documents.csv) from HuggingFace
  Phase 2: Translate queries and documents via LLM API (parallel workers or OpenAI Batch API)
  Phase 3: Export translated CSVs to BeIR JSONL format (HuggingFace-ready)

Supports continuation: progress is checkpointed per dataset so a crashed run resumes
from where it left off by re-running the same command.

Usage:
    python src/translation/api/run_beir_translation_pipeline.py --config config/translation/beir_translation_zeroshot_gpt4o_mini.yaml
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from translation.api.translate import run_translation_pipeline as run_parallel_pipeline
from translation.beir.export import export_to_beir_jsonl
from translation.beir.cache import TranslationCache


# ── Logging helpers ───────────────────────────────────────────────────────────
# All output is routed through the active Progress console (when the sticky
# progress bar is running) so log lines scroll above the bar instead of
# breaking it.

_console = Console()
_active_progress: Optional[Progress] = None


def _log(msg: str) -> None:
    if _active_progress is not None:
        _active_progress.console.print(msg)
    else:
        _console.print(msg)


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=38),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[phase]}"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    )


# ── Defaults applied when config keys are absent ────────────────────────────

DEFAULTS = {
    "run_id": "",
    "model": {
        "name": "gpt-4o-mini-2024-07-18",
        "temperature": 0.7,
    },
    "prompt": {
        "file": "",
        "response_format": "Translation",
        "english_key": "English",
        "hebrew_key": "Hebrew",
        "context_key": "Context",
        "hebrew_key_query": "Hebrew Query",
        "hebrew_key_document": "Hebrew Document",
    },
    "datasets": {
        "names": [],
        "num_samples": 0,
        "max_document_segment_tokens": 512,
        "tokenizer_model": "gpt-4o-mini-2024-07-18",
        "translate_titles": True,
        "random_seed": 42,
    },
    "execution": {
        "mode": "auto",
        "num_workers": 8,
        "sleep_time": 0,
        "limit": 0,
        "force_candidates": False,
        "force_translation": False,
        "force_export": False,
    },
    "paths": {
        "candidates_base": "outputs/translation/BeIR",
        "runs_base": "outputs/beir_translation",
    },
    "batch": {
        "poll_interval_seconds": 3600,
        "max_wait_hours": 24,
        "job_tracking_dir": "jobs",
    },
    "export": {
        "segment_separator": " ",
        "include_context": False,
    },
    # ── New sections ──────────────────────────────────────────────────────────
    "guardrails": {
        # Hard stop before submitting if estimated cost exceeds this
        "max_cost_usd": 99999.0,
        # Batch API pricing (50% discount vs synchronous)
        "cost_per_1m_input_tokens": 0.075,
        "cost_per_1m_output_tokens": 0.300,
    },
    "progression": {
        # Translate this many rows first, run QA, then continue with the rest.
        # Set to 0 to skip pilot and go straight to full translation.
        "pilot_n": 0,
        "pilot_qa": False,
    },
    "qa": {
        # Per-dataset QA after each dataset finishes translating.
        "enabled": False,
        "baseline_csv": "",
        "baseline_model": "gpt-5.4-mini",
        "baseline_prompt": "zeroshot_nocontext",
        "judge_model": "claude-sonnet-4-6",
        "sample_size": 25,
        "workers": 4,
    },
}


# ── Config loading ───────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursively for nested dicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    """Load YAML config and apply defaults for missing fields."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    config = _deep_merge(DEFAULTS, raw)
    config["_config_path"] = config_path
    return config


# ── Run ID ───────────────────────────────────────────────────────────────────

def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")


def make_run_id(config: dict) -> str:
    if config.get("run_id"):
        return config["run_id"]
    model_slug = _slugify(config["model"]["name"])
    prompt_slug = _slugify(Path(config["prompt"]["file"]).stem)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_slug}__{prompt_slug}__{ts}"


# ── Progress tracking ────────────────────────────────────────────────────────

def _progress_path(run_dir: str, filename: str = "progress.json") -> str:
    return os.path.join(run_dir, filename)


def load_or_init_progress(run_dir: str, config: dict, run_id: str, filename: str = "progress.json") -> dict:
    path = _progress_path(run_dir, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            progress = json.load(f)
        _log(f"[progress] Resuming run '{run_id}' from {path}")
        return progress

    dataset_names = config["datasets"]["names"]
    # Per-type configs (queries/documents) don't have a top-level model/prompt;
    # fall back gracefully so this works for both schemas.
    model_name = config.get("model", {}).get("name") or config.get("queries", {}).get("model", "")
    prompt_file = config.get("prompt", {}).get("file") or config.get("queries", {}).get("prompt", {}).get("file", "")
    progress = {
        "run_id": run_id,
        "config_file": config.get("_config_path", ""),
        "model_name": model_name,
        "prompt_file": prompt_file,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "datasets": {
            _dataset_slug(name): _empty_dataset_entry(name)
            for name in dataset_names
        },
    }
    os.makedirs(run_dir, exist_ok=True)
    save_progress(run_dir, progress, filename)
    return progress


def save_progress(run_dir: str, progress: dict, filename: str = "progress.json") -> None:
    """Atomic write: write to .tmp then rename to avoid corruption on crash."""
    progress["updated_at"] = datetime.now().isoformat()
    path = _progress_path(run_dir, filename)
    tmp_path = path + ".tmp"
    os.makedirs(run_dir, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    os.replace(tmp_path, path)


# ── Execution mode ────────────────────────────────────────────────────────────

def resolve_execution_mode(config: dict) -> str:
    mode = config["execution"]["mode"]
    if mode == "auto":
        model = config["model"]["name"]
        if re.match(r".*gpt.*", model):
            return "batch"
        if re.match(r"gemini.*", model):
            return "gemini_batch"
        return "parallel"
    return mode  # "batch" | "gemini_batch" | "parallel" | "serial"


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def _empty_dataset_entry(dataset_name: str) -> dict:
    return {
        "dataset_name": dataset_name,
        "candidates_built": False,
        "queries_pilot_done": False,
        "queries_pilot_qa_passed": None,
        "queries_translated": False,
        "queries_qa_passed": False,
        "titles_translated": False,
        "documents_pilot_done": False,
        "documents_pilot_qa_passed": None,
        "documents_translated": False,
        "documents_qa_passed": False,
        "exported_to_beir": False,
        "queries_batch_job_id": None,
        "titles_batch_job_id": None,
        "docs_batch_job_id": None,
        "error": None,
    }


def _estimate_and_confirm(csv_path: str, text_col: str, config: dict, yes: bool = False) -> None:
    """Estimate cost, check budget cap, and optionally prompt for confirmation."""
    from translation.cost_estimator import estimate_batch_cost
    est = estimate_batch_cost(csv_path, text_col, config)
    if est["n_pending"] == 0:
        return

    max_cost = config.get("guardrails", {}).get("max_cost_usd", float("inf"))
    _log(f"  [cost] {est['n_pending']:,} pending rows × ~{est['avg_text_tokens']} tokens "
         f"→ estimated ${est['estimated_cost_usd']:.2f} "
         f"({est['estimated_input_tokens']/1e6:.2f}M in + {est['estimated_output_tokens']/1e6:.2f}M out tokens)")

    if est["estimated_cost_usd"] > max_cost:
        raise RuntimeError(
            f"Estimated cost ${est['estimated_cost_usd']:.2f} exceeds budget cap ${max_cost:.2f}. "
            f"Raise guardrails.max_cost_usd in the config to proceed."
        )

    if not yes:
        try:
            answer = input("  Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes"):
            raise RuntimeError("Translation cancelled by user at cost confirmation.")


def _save_qa_history(
    run_dir: str,
    config: dict,
    dataset_slug: str,
    text_type: str,
    sample_size: int,
    score_mean: float,
    score_std: float,
    passed: bool,
) -> None:
    import csv
    run_id = config.get("run_id", "")
    cfg_key = "queries" if text_type == "query" else "documents"
    translation_model = config.get(cfg_key, {}).get("model", "")
    judge_model = config.get("qa", {}).get("judge_model", "")
    history_path = str(Path(run_dir).parent.parent / "qa_history.csv")
    row = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "dataset_slug": dataset_slug,
        "text_type": text_type,
        "translation_model": translation_model,
        "judge_model": judge_model,
        "sample_size": sample_size,
        "score_mean": round(score_mean, 4),
        "score_std": round(score_std, 4),
        "passed": passed,
    }
    write_header = not os.path.exists(history_path)
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    try:
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        _log(f"  [qa] WARNING: could not save QA history: {e}")


def _run_dataset_qa(
    translated_csv: str,
    dataset_slug: str,
    text_type: str,
    config: dict,
    run_dir: str,
) -> bool:
    """
    Sample translated rows and run LLM-as-a-judge. Compare against baseline.
    Returns True if QA passes, False if degradation detected.
    """
    qa_cfg = config.get("qa", {})
    if not qa_cfg.get("enabled", False):
        return True

    # The baseline is only used for degradation comparison. Its absence must NOT
    # skip the judge — without a baseline we still score and gate on min_score.
    baseline_csv = qa_cfg.get("baseline_csv", "")
    have_baseline = bool(baseline_csv) and os.path.exists(baseline_csv)

    import tempfile
    import pandas as pd
    from translation.api.evaluate_translations import run_evaluate_translations
    from translation.qa_phase import (
        compare_scores, load_baseline, DATASET_EVAL_PROMPTS,
    )

    sample_size = qa_cfg.get("sample_size", 25)
    judge_model = qa_cfg.get("judge_model", "gemini-2.5-pro")
    judge_location = qa_cfg.get("judge_location")
    baseline_model = qa_cfg.get("baseline_model", "gpt-5.4-mini")
    baseline_prompt_slug = qa_cfg.get("baseline_prompt", "zeroshot_nocontext")
    workers = qa_cfg.get("workers", 4)
    sleep_time = qa_cfg.get("sleep_time", 0)

    df = pd.read_csv(translated_csv, encoding="utf-8")
    df = df[df["translation"].notna()]
    if df.empty:
        _log(f"  [qa] No translated rows to evaluate in {translated_csv}.")
        return True

    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    suffix = "_queries.csv" if text_type == "query" else "_documents.csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as tmp:
        sample.to_csv(tmp, index=False)
        tmp_path = tmp.name

    prompt_file = DATASET_EVAL_PROMPTS.get(
        dataset_slug,
        "prompts/translation/api/evaluation/translation_evaluation_nogold_v20250406.yaml",
    )
    tmp_out_dir = tempfile.mkdtemp()
    text_col = "text" if text_type == "query" else "segment_text"

    # Override GEMINI_LOCATION for the judge if it lives in a different region
    _prev_location = os.environ.get("GEMINI_LOCATION")
    if judge_location:
        os.environ["GEMINI_LOCATION"] = judge_location
    try:
        evaluated = run_evaluate_translations(
            source_file_path=tmp_path,
            output_dir=tmp_out_dir,
            gold_file_path=None,
            prompt_file_name=prompt_file,
            model_name=judge_model,
            limit=0,
            sample=0.0,
            force=True,
            parallel=True,
            num_workers=workers,
            sleep_time=sleep_time,
            english_key=text_col,
            hebrew_key="translation",
            prompt_type=text_type,
        )
    finally:
        os.unlink(tmp_path)
        if judge_location:
            if _prev_location is not None:
                os.environ["GEMINI_LOCATION"] = _prev_location
            else:
                os.environ.pop("GEMINI_LOCATION", None)

    if evaluated is None or "score" not in evaluated.columns:
        _log(f"  [qa] Evaluation returned no scores — skipping comparison.")
        return True

    valid = evaluated["score"].dropna()
    if valid.empty:
        _log(f"  [qa] All scores null — skipping comparison.")
        return True

    sample_mean = float(valid.mean())
    sample_std = float(valid.std())

    # No baseline → gate on absolute min_score (same rule as the ladder pipeline).
    if not have_baseline:
        min_score = qa_cfg.get("min_score", 3.5)
        passed = sample_mean >= min_score
        status = "PASS" if passed else "FAIL"
        _log(f"  [qa] {dataset_slug}/{text_type}: [{status}] "
             f"score={sample_mean:.3f}±{sample_std:.3f}  (min_score={min_score}, no baseline)")
        _save_qa_history(run_dir, config, dataset_slug, text_type, len(valid), sample_mean, sample_std, passed)
        return passed

    baseline_df = load_baseline(baseline_csv, baseline_model, baseline_prompt_slug, judge_model)
    baseline_stats = (
        baseline_df.groupby(["dataset_slug", "text_type"])["score"]
        .agg(["mean", "std"])
    )
    idx = (dataset_slug, text_type)
    if idx not in baseline_stats.index:
        _log(f"  [qa] No baseline for {dataset_slug}/{text_type} — reporting score only: {sample_mean:.3f}±{sample_std:.3f}")
        _save_qa_history(run_dir, config, dataset_slug, text_type, len(valid), sample_mean, sample_std, True)
        return True

    baseline_mean = float(baseline_stats.loc[idx, "mean"])
    baseline_std_val = float(baseline_stats.loc[idx, "std"])
    degraded, reason = compare_scores(sample_mean, baseline_mean, baseline_std_val)

    status = "DEGRADED" if degraded else "OK"
    _log(f"  [qa] {dataset_slug}/{text_type}: [{status}] "
         f"sample={sample_mean:.3f}±{sample_std:.3f}  baseline={baseline_mean:.3f}±{baseline_std_val:.3f}"
         + (f"  ← {reason}" if reason else ""))
    _save_qa_history(run_dir, config, dataset_slug, text_type, len(valid), sample_mean, sample_std, not degraded)
    return not degraded


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _apply_cache_and_dedup(
    source_csv: str,
    output_csv: str,
    cache: TranslationCache,
    model_name: str,
    prompt_file: str,
    text_col: str,
    context_col: Optional[str],
) -> dict:
    """
    Pre-populate the output CSV before the translation pipeline runs.

    Two dedup layers:
      1. Cache lookup: rows whose (text, context) already exist in the cache
         get their `translation` pre-filled — the pipeline will skip them.
      2. Within-file dedup: if multiple rows share the same (text, context),
         any already-translated row's value is propagated to the untranslated
         duplicates in the same file.

    Reads from `output_csv` if it exists (resuming a previous run), otherwise
    from `source_csv`.  Only writes the output CSV when there are actual fills,
    so a clean first run without cache hits produces no file until the pipeline
    itself writes it.

    Returns stats dict: {total, already_done, cache_hits, dedup_fills, remaining}
    """
    input_csv = output_csv if os.path.exists(output_csv) else source_csv
    df = pd.read_csv(input_csv, encoding="utf-8")

    if "translation" not in df.columns:
        df["translation"] = None

    already_done = int(df["translation"].notna().sum())

    # ── Layer 1: cache lookup ────────────────────────────────────────────────
    cache_hits = 0
    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, "translation"]):
            continue
        text = str(row[text_col]) if text_col in df.columns and pd.notna(row.get(text_col)) else ""
        context = (
            str(row[context_col])
            if context_col and context_col in df.columns and pd.notna(row.get(context_col))
            else ""
        )
        cached = cache.lookup(model_name, prompt_file, text, context)
        if cached is not None:
            df.at[idx, "translation"] = cached
            cache_hits += 1

    # ── Layer 2: within-file dedup ───────────────────────────────────────────
    # For each group of rows sharing identical (text, context), propagate any
    # existing translation to the unfilled rows in that group.
    group_keys = [text_col] + ([context_col] if context_col and context_col in df.columns else [])
    dedup_fills = 0
    for _, group in df.groupby(group_keys, sort=False, dropna=False):
        translated_mask = group["translation"].notna()
        unfilled_mask = group["translation"].isna()
        if translated_mask.any() and unfilled_mask.any():
            first_translation = group.loc[translated_mask, "translation"].iloc[0]
            df.loc[group.index[unfilled_mask], "translation"] = first_translation
            dedup_fills += int(unfilled_mask.sum())

    remaining = int(df["translation"].isna().sum())

    if cache_hits > 0 or dedup_fills > 0:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")

    _log(
        f"  [dedup] {already_done} already done, "
        f"{cache_hits} cache hits, {dedup_fills} within-file fills, "
        f"{remaining} remaining to translate"
    )
    return {
        "total": len(df),
        "already_done": already_done,
        "cache_hits": cache_hits,
        "dedup_fills": dedup_fills,
        "remaining": remaining,
    }


def _expand_dedup_and_update_cache(
    translated_csv: str,
    cache: TranslationCache,
    model_name: str,
    prompt_file: str,
    text_col: str,
    context_col: Optional[str],
) -> None:
    """
    After the pipeline completes, run a second pass to:
      1. Expand within-file duplicates: rows that are still unfilled but share
         (text, context) with a newly translated row get that translation.
         This handles the case where duplicates were not pre-filled because
         neither was in the cache before this run.
      2. Store all new translations into the cache for future runs/datasets.
    """
    if not os.path.exists(translated_csv):
        return

    df = pd.read_csv(translated_csv, encoding="utf-8")

    # ── Expand within-file duplicates ────────────────────────────────────────
    group_keys = [text_col] + ([context_col] if context_col and context_col in df.columns else [])
    dedup_fills = 0
    for _, group in df.groupby(group_keys, sort=False, dropna=False):
        translated_mask = group["translation"].notna()
        unfilled_mask = group["translation"].isna()
        if translated_mask.any() and unfilled_mask.any():
            first_translation = group.loc[translated_mask, "translation"].iloc[0]
            df.loc[group.index[unfilled_mask], "translation"] = first_translation
            dedup_fills += int(unfilled_mask.sum())

    if dedup_fills > 0:
        df.to_csv(translated_csv, index=False, encoding="utf-8")
        _log(f"  [dedup] Expanded {dedup_fills} within-file duplicates after pipeline.")

    # ── Update cache ─────────────────────────────────────────────────────────
    new_entries = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("translation")):
            continue
        text = str(row[text_col]) if text_col in df.columns and pd.notna(row.get(text_col)) else ""
        context = (
            str(row[context_col])
            if context_col and context_col in df.columns and pd.notna(row.get(context_col))
            else ""
        )
        if cache.lookup(model_name, prompt_file, text, context) is None:
            cache.store(model_name, prompt_file, text, context, str(row["translation"]))
            new_entries += 1

    _log(f"  [cache] Stored {new_entries} new translations (cache size: {len(cache)})")


# ── Phase 1: Build candidates ─────────────────────────────────────────────────

def _phase_build_candidates(dataset_name: str, dataset_slug: str, config: dict) -> None:
    import shutil
    from translation.build_translation_candidates import build_dataset_candidates
    candidates_base = config["paths"]["candidates_base"]
    ds_cfg = config["datasets"]

    _log(f"  [candidates] Building candidate CSVs for {dataset_name}...")
    build_dataset_candidates(
        dataset_names=[dataset_name],
        num_samples=ds_cfg["num_samples"],
        max_document_segment_tokens=ds_cfg["max_document_segment_tokens"],
        model_name_or_path=ds_cfg["tokenizer_model"],
        output_path=candidates_base,
        force=config["execution"]["force_candidates"],
        random_state=ds_cfg["random_seed"],
    )

    # build_dataset_candidates saves to {slug}/test/ — promote to {slug}/ if needed
    slug_dir = os.path.join(candidates_base, dataset_slug)
    for fname in ("queries.csv", "documents.csv"):
        src = os.path.join(slug_dir, "test", fname)
        dst = os.path.join(slug_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


# ── Phase 2: Translate files ──────────────────────────────────────────────────

def _phase_translate_file(
    source_csv: str,
    output_dir: str,
    config: dict,
    execution_mode: str,
    job_key: str,
    progress_entry: dict,
    run_dir: str,
    progress: dict,
    override_limit: int = -1,
) -> None:
    """
    Translate a single CSV file (queries or documents).

    Parallel/serial: delegates to run_translation_pipeline() which has built-in
    continuation (skips rows where `translation` is already populated).

    Batch: submits to OpenAI Batch API, polls hourly until complete, then retrieves results.
    """
    prompt_cfg = config["prompt"]
    exec_cfg = config["execution"]

    effective_limit = override_limit if override_limit >= 0 else exec_cfg["limit"]

    common_kwargs = dict(
        source_file_path=source_csv,
        output_dir=output_dir,
        prompt_file_name=prompt_cfg["file"],
        model_name=config["model"]["name"],
        limit=effective_limit,
        force=exec_cfg["force_translation"],
        english_key=prompt_cfg["english_key"],
        hebrew_key=prompt_cfg["hebrew_key"],
        context_key=prompt_cfg["context_key"],
        hebrew_key_query=prompt_cfg["hebrew_key_query"],
        hebrew_key_document=prompt_cfg["hebrew_key_document"],
        response_format=prompt_cfg["response_format"],
        sleep_time=exec_cfg["sleep_time"],
    )

    if execution_mode == "batch":
        _translate_batch_mode(common_kwargs, job_key, progress_entry, run_dir, progress, config)
    elif execution_mode == "gemini_batch":
        _translate_gemini_batch_mode(common_kwargs, job_key, progress_entry, run_dir, progress, config)
    elif execution_mode == "parallel":
        run_parallel_pipeline(parallel=True, num_workers=exec_cfg["num_workers"], **common_kwargs)
    else:  # serial
        run_parallel_pipeline(parallel=False, **common_kwargs)


def _translate_batch_mode(
    pipeline_kwargs: dict,
    job_key: str,
    progress_entry: dict,
    run_dir: str,
    progress: dict,
    config: dict,
) -> None:
    """Submit to OpenAI Batch API, poll until all chunks done, retrieve results."""
    from translation.api.translate_batch import (
        run_translation_pipeline as run_batch_pipeline,
        check_batch_status,
        retrieve_batch_results,
    )

    batch_cfg = config["batch"]
    poll_secs = batch_cfg["poll_interval_seconds"]
    max_secs = batch_cfg["max_wait_hours"] * 3600
    job_tracking_dir = batch_cfg["job_tracking_dir"]
    source_file = pipeline_kwargs["source_file_path"]

    # progress_entry[job_key] stores a JSON list of job IDs (one per chunk)
    existing = progress_entry.get(job_key)
    if existing:
        job_ids = json.loads(existing) if isinstance(existing, str) and existing.startswith('[') else [existing]
        _log(f"  [batch] Resuming {len(job_ids)} chunk job(s) for {source_file}...")
    else:
        _log(f"  [batch] Submitting batch jobs for {source_file}...")
        pipeline_kwargs["job_tracking_dir"] = job_tracking_dir
        job_ids = run_batch_pipeline(**pipeline_kwargs)
        if not job_ids:
            _log(f"  [batch] No pending rows — skipping.")
            return
        progress_entry[job_key] = json.dumps(job_ids)
        save_progress(run_dir, progress)
        _log(f"  [batch] Submitted {len(job_ids)} chunk(s): {job_ids}")

    # Poll until all chunks complete
    elapsed = 0
    while elapsed < max_secs:
        all_jobs = check_batch_status(job_tracking_dir)
        chunk_jobs = [j for j in all_jobs if j.get("job_id") in job_ids]

        statuses = {j["job_id"]: j.get("status", "unknown") for j in chunk_jobs}
        failed = [jid for jid, s in statuses.items() if s in ("failed", "expired", "cancelled")]
        if failed:
            raise RuntimeError(f"Batch chunk(s) failed: {failed}")

        done_count = sum(1 for s in statuses.values() if s == "completed")
        _log(f"  [batch] {done_count}/{len(job_ids)} chunks completed (elapsed {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m)")

        if done_count == len(job_ids):
            retrieve_batch_results(source_file_path=source_file, job_tracking_dir=job_tracking_dir)
            _log(f"  [batch] All chunks done and results retrieved for {source_file}.")
            return

        _log(f"  [batch] Sleeping {poll_secs // 3600:.0f}h before next poll...")
        time.sleep(poll_secs)
        elapsed += poll_secs

    raise TimeoutError(f"Batch jobs for {source_file} did not complete within {batch_cfg['max_wait_hours']}h.")


def _translate_gemini_batch_mode(
    pipeline_kwargs: dict,
    job_key: str,
    progress_entry: dict,
    run_dir: str,
    progress: dict,
    config: dict,
) -> None:
    """Submit to Gemini Batch API, poll until done, retrieve results."""
    from translation.api.translate_batch_gemini import (
        run_translation_pipeline as run_gemini_pipeline,
        check_batch_status,
        retrieve_batch_results,
        FAILED_STATES,
    )

    batch_cfg = config["batch"]
    poll_secs = batch_cfg["poll_interval_seconds"]
    max_secs = batch_cfg["max_wait_hours"] * 3600
    tracking_dir = batch_cfg["job_tracking_dir"]
    source_file = pipeline_kwargs["source_file_path"]

    existing_job_name = progress_entry.get(job_key)
    if existing_job_name:
        _log(f"  [gemini_batch] Resuming job {existing_job_name} for {source_file}...")
        job_name = existing_job_name
    else:
        _log(f"  [gemini_batch] Submitting batch job for {source_file}...")
        job_name = run_gemini_pipeline(**pipeline_kwargs, tracking_dir=tracking_dir)
        if not job_name:
            _log(f"  [gemini_batch] No pending rows — skipping.")
            return
        progress_entry[job_key] = job_name
        save_progress(run_dir, progress)
        _log(f"  [gemini_batch] Submitted: {job_name}")

    elapsed = 0
    while elapsed < max_secs:
        jobs = check_batch_status(job_names=[job_name], tracking_dir=tracking_dir)
        if not jobs:
            raise RuntimeError(f"Gemini job {job_name} not found in tracking file.")

        status = jobs[0].get("status", "unknown")

        if status in FAILED_STATES:
            raise RuntimeError(f"Gemini batch job {job_name} reached failed state: {status}")

        if status == "JOB_STATE_SUCCEEDED":
            retrieve_batch_results(job_names=[job_name], tracking_dir=tracking_dir)
            _log(f"  [gemini_batch] Job {job_name} complete and results retrieved.")
            return

        _log(f"  [gemini_batch] Status: {status}. Sleeping {poll_secs // 3600:.0f}h "
             f"{(poll_secs % 3600) // 60:.0f}m before next poll...")
        time.sleep(poll_secs)
        elapsed += poll_secs

    raise TimeoutError(
        f"Gemini batch job {job_name} did not complete within {batch_cfg['max_wait_hours']}h."
    )


# ── Phase 2b: Translate titles ────────────────────────────────────────────────

def _phase_translate_titles(
    dataset_slug: str,
    candidates_base: str,
    run_dir: str,
    config: dict,
    execution_mode: str,
    progress_entry: dict,
    progress: dict,
) -> None:
    """
    Translate unique document titles. Runs a separate lightweight translation pass
    on a deduplicated titles CSV, then merges title_translation back into the
    translated documents CSV.
    """
    import pandas as pd

    documents_csv = os.path.join(candidates_base, dataset_slug, "documents.csv")
    dataset_run_dir = os.path.join(run_dir, dataset_slug)
    titles_csv = os.path.join(dataset_run_dir, "titles.csv")
    titles_translated_csv = os.path.join(dataset_run_dir, "titles_translated.csv")
    documents_translated_csv = os.path.join(dataset_run_dir, "documents_translated.csv")

    # Build a deduplicated titles CSV (one row per unique _id with a non-empty title)
    if not os.path.exists(titles_csv) or config["execution"]["force_translation"]:
        docs_df = pd.read_csv(documents_csv, encoding="utf-8")
        titles_df = (
            docs_df[["_id", "title"]]
            .drop_duplicates(subset=["_id"])
            .query("title.notna() and title.str.strip() != ''", engine="python")
            .rename(columns={"title": "segment_text"})  # reuse segment_text field in prompt
        )
        if titles_df.empty:
            _log(f"  [titles] No non-empty titles found for {dataset_slug}, skipping title translation.")
            return
        os.makedirs(dataset_run_dir, exist_ok=True)
        titles_df.to_csv(titles_csv, index=False, encoding="utf-8")

    # Translate titles (reuses the document prompt type)
    _phase_translate_file(
        source_csv=titles_csv,
        output_dir=dataset_run_dir,
        config=config,
        execution_mode=execution_mode,
        job_key="titles_batch_job_id",
        progress_entry=progress_entry,
        run_dir=run_dir,
        progress=progress,
    )

    # The translated file will be titles_translated.csv (output_dir + basename + _translated)
    if not os.path.exists(titles_translated_csv):
        _log(f"  [titles] titles_translated.csv not found, skipping title merge.")
        return

    # Merge title_translation into documents_translated.csv
    if os.path.exists(documents_translated_csv):
        docs_translated = pd.read_csv(documents_translated_csv, encoding="utf-8")
        titles_translated = pd.read_csv(titles_translated_csv, encoding="utf-8")[["_id", "translation"]]
        titles_translated = titles_translated.rename(columns={"translation": "title_translation"})
        # Drop existing title_translation column if present
        if "title_translation" in docs_translated.columns:
            docs_translated = docs_translated.drop(columns=["title_translation"])
        docs_translated = docs_translated.merge(titles_translated, on="_id", how="left")
        docs_translated.to_csv(documents_translated_csv, index=False, encoding="utf-8")
        _log(f"  [titles] Merged title_translation into {documents_translated_csv}")


# ── Per-dataset orchestration ─────────────────────────────────────────────────

def _process_dataset(
    dataset_name: str,
    config: dict,
    run_id: str,
    run_dir: str,
    progress: dict,
    execution_mode: str,
    cache: TranslationCache,
    text_type: str = "both",
    yes: bool = False,
    rich_progress: Optional[Progress] = None,
    overall_task_id: Optional[TaskID] = None,
) -> None:
    dataset_slug = _dataset_slug(dataset_name)
    entry = progress["datasets"].setdefault(dataset_slug, _empty_dataset_entry(dataset_name))
    # Ensure new progress keys exist for datasets created before this version
    for key, default in _empty_dataset_entry(dataset_name).items():
        entry.setdefault(key, default)

    candidates_base = config["paths"]["candidates_base"]
    dataset_run_dir = os.path.join(run_dir, dataset_slug)

    queries_csv = os.path.join(candidates_base, dataset_slug, "queries.csv")
    documents_csv = os.path.join(candidates_base, dataset_slug, "documents.csv")

    do_queries = text_type in ("query", "both")
    do_documents = text_type in ("document", "both")
    do_titles = config["datasets"]["translate_titles"] and do_documents

    pilot_n = config.get("progression", {}).get("pilot_n", 0)
    pilot_qa = config.get("progression", {}).get("pilot_qa", False)

    # Count phases for progress bar
    with_titles = do_titles
    num_phases = 1  # candidates
    if do_queries:
        num_phases += (2 if pilot_n > 0 else 1)  # pilot + full, or just full
    if with_titles:
        num_phases += 1
    if do_documents:
        num_phases += (2 if pilot_n > 0 else 1)
    num_phases += 1  # export

    phases_done = sum([
        entry["candidates_built"],
        entry.get("queries_pilot_done", False) if do_queries and pilot_n > 0 else (0 if do_queries and pilot_n > 0 else 0),
        entry["queries_translated"] if do_queries else True,
        entry["titles_translated"] if with_titles else True,
        entry.get("documents_pilot_done", False) if do_documents and pilot_n > 0 else 0,
        entry["documents_translated"] if do_documents else True,
        entry["exported_to_beir"],
    ])

    ds_task_id: Optional[TaskID] = None
    if rich_progress is not None:
        ds_task_id = rich_progress.add_task(
            f"[cyan]{dataset_name}",
            total=num_phases,
            completed=phases_done,
            phase="",
        )

    def _advance(phase_desc: str = "") -> None:
        if rich_progress is not None and ds_task_id is not None:
            rich_progress.update(ds_task_id, advance=1, phase=phase_desc)

    def _set_phase(phase_desc: str) -> None:
        if rich_progress is not None and ds_task_id is not None:
            rich_progress.update(ds_task_id, phase=phase_desc)

    _log(f"  ── {dataset_name} (text_type={text_type}) ──")

    def _fail(phase: str, exc: Exception) -> None:
        entry["error"] = f"{phase}: {exc}"
        save_progress(run_dir, progress)
        _log(f"  [ERROR] {phase} failed for {dataset_name}: {exc}")
        if rich_progress and ds_task_id is not None:
            rich_progress.remove_task(ds_task_id)

    # ── Phase 1: Build candidates ─────────────────────────────────
    if not entry["candidates_built"] or config["execution"]["force_candidates"]:
        _set_phase("building candidates")
        try:
            _phase_build_candidates(dataset_name, dataset_slug, config)
            entry["candidates_built"] = True
            save_progress(run_dir, progress)
            _advance("candidates done")
        except Exception as e:
            _fail("candidates", e); return
    else:
        _log(f"  [skip] Candidates already built.")

    os.makedirs(dataset_run_dir, exist_ok=True)
    queries_translated_csv = os.path.join(dataset_run_dir, "queries_translated.csv")
    documents_translated_csv = os.path.join(dataset_run_dir, "documents_translated.csv")

    # ── Helper: translate one file with optional pilot + QA ──────
    def _translate_with_progression(
        source_csv: str,
        translated_csv: str,
        text_col: str,
        context_col: Optional[str],
        job_key: str,
        pilot_done_key: str,
        pilot_qa_key: str,
        translated_key: str,
        qa_key: str,
        phase_label: str,
    ) -> bool:
        """Returns False if QA fails and we should abort this dataset."""
        force = config["execution"]["force_translation"]

        # Apply pre-translation cache / dedup
        _apply_cache_and_dedup(
            source_csv=source_csv,
            output_csv=translated_csv,
            cache=cache,
            model_name=config["model"]["name"],
            prompt_file=config["prompt"]["file"],
            text_col=text_col,
            context_col=context_col,
        )

        # ── Pilot phase ───────────────────────────────────────────
        if pilot_n > 0 and not entry.get(pilot_done_key, False):
            _set_phase(f"pilot {phase_label} ({pilot_n} rows)")
            _log(f"  [pilot] Translating first {pilot_n} rows of {phase_label}...")
            try:
                _estimate_and_confirm(translated_csv, text_col, config, yes)
                _phase_translate_file(
                    source_csv=source_csv,
                    output_dir=dataset_run_dir,
                    config=config,
                    execution_mode=execution_mode,
                    job_key=job_key + "_pilot",
                    progress_entry=entry,
                    run_dir=run_dir,
                    progress=progress,
                    override_limit=pilot_n,
                )
                entry[pilot_done_key] = True
                save_progress(run_dir, progress)
                _advance(f"pilot {phase_label} done")
            except Exception as e:
                _fail(f"pilot_{phase_label}", e); return False

        # ── Pilot QA ─────────────────────────────────────────────
        if pilot_n > 0 and pilot_qa and not entry.get(pilot_qa_key, False):
            _set_phase(f"QA pilot {phase_label}")
            tt = "query" if "quer" in phase_label else "document"
            passed = _run_dataset_qa(translated_csv, dataset_slug, tt, config, run_dir)
            entry[pilot_qa_key] = passed
            save_progress(run_dir, progress)
            if not passed:
                entry["error"] = f"pilot_qa_failed_{phase_label}"
                save_progress(run_dir, progress)
                _log(f"  [QA FAIL] Pilot QA failed for {dataset_name}/{phase_label}. "
                     f"Fix the issue and re-run to continue from the pilot QA checkpoint.")
                return False

        # ── Full translation ──────────────────────────────────────
        if not entry.get(translated_key, False) or force:
            _set_phase(f"translating {phase_label}")
            try:
                _estimate_and_confirm(translated_csv, text_col, config, yes)
                _phase_translate_file(
                    source_csv=source_csv,
                    output_dir=dataset_run_dir,
                    config=config,
                    execution_mode=execution_mode,
                    job_key=job_key,
                    progress_entry=entry,
                    run_dir=run_dir,
                    progress=progress,
                    override_limit=0,
                )
                _expand_dedup_and_update_cache(
                    translated_csv=translated_csv,
                    cache=cache,
                    model_name=config["model"]["name"],
                    prompt_file=config["prompt"]["file"],
                    text_col=text_col,
                    context_col=context_col,
                )
                entry[translated_key] = True
                entry["error"] = None
                save_progress(run_dir, progress)
                _advance(f"{phase_label} done")
            except Exception as e:
                _fail(f"{phase_label}_translation", e); return False
        else:
            _log(f"  [skip] {phase_label} already translated.")

        # ── Post-translation QA ───────────────────────────────────
        qa_enabled = config.get("qa", {}).get("enabled", False)
        if qa_enabled and not entry.get(qa_key, False):
            _set_phase(f"QA {phase_label}")
            tt = "query" if "quer" in phase_label else "document"
            passed = _run_dataset_qa(translated_csv, dataset_slug, tt, config, run_dir)
            entry[qa_key] = passed
            save_progress(run_dir, progress)
            if not passed:
                entry["error"] = f"qa_failed_{phase_label}"
                save_progress(run_dir, progress)
                _log(f"  [QA FAIL] Post-translation QA failed for {dataset_name}/{phase_label}.")
                return False

        return True

    # ── Phase 2a: Queries ─────────────────────────────────────────
    if do_queries:
        ok = _translate_with_progression(
            source_csv=queries_csv,
            translated_csv=queries_translated_csv,
            text_col="text",
            context_col="context_text",
            job_key="queries_batch_job_id",
            pilot_done_key="queries_pilot_done",
            pilot_qa_key="queries_pilot_qa_passed",
            translated_key="queries_translated",
            qa_key="queries_qa_passed",
            phase_label="queries",
        )
        if not ok:
            if rich_progress and ds_task_id is not None:
                rich_progress.remove_task(ds_task_id)
            return

    # ── Phase 2b: Document titles ─────────────────────────────────
    if do_titles:
        if not entry["titles_translated"] or config["execution"]["force_translation"]:
            _set_phase("translating titles")
            try:
                _phase_translate_titles(
                    dataset_slug=dataset_slug,
                    candidates_base=candidates_base,
                    run_dir=run_dir,
                    config=config,
                    execution_mode=execution_mode,
                    progress_entry=entry,
                    progress=progress,
                )
                entry["titles_translated"] = True
                entry["error"] = None
                save_progress(run_dir, progress)
                _advance("titles done")
            except Exception as e:
                _fail("titles_translation", e); return
        else:
            _log(f"  [skip] Titles already translated.")

    # ── Phase 2c: Documents ───────────────────────────────────────
    if do_documents:
        ok = _translate_with_progression(
            source_csv=documents_csv,
            translated_csv=documents_translated_csv,
            text_col="segment_text",
            context_col=None,
            job_key="docs_batch_job_id",
            pilot_done_key="documents_pilot_done",
            pilot_qa_key="documents_pilot_qa_passed",
            translated_key="documents_translated",
            qa_key="documents_qa_passed",
            phase_label="documents",
        )
        if not ok:
            if rich_progress and ds_task_id is not None:
                rich_progress.remove_task(ds_task_id)
            return

    # ── Phase 3: Export to BeIR JSONL ─────────────────────────────
    # Export only when both the files this run produced exist
    queries_ready = os.path.exists(queries_translated_csv)
    documents_ready = os.path.exists(documents_translated_csv)
    can_export = queries_ready and documents_ready

    if can_export and (not entry["exported_to_beir"] or config["execution"]["force_export"]):
        _set_phase("exporting")
        try:
            run_metadata = {
                "run_id": run_id,
                "model_name": config["model"]["name"],
                "prompt_file": config["prompt"]["file"],
                "response_format": config["prompt"]["response_format"],
                "temperature": config["model"]["temperature"],
            }
            export_to_beir_jsonl(
                translated_queries_csv=queries_translated_csv,
                translated_documents_csv=documents_translated_csv,
                dataset_name=dataset_name,
                output_dir=dataset_run_dir,
                run_metadata=run_metadata,
                segment_separator=config["export"]["segment_separator"],
                force=config["execution"]["force_export"],
            )
            entry["exported_to_beir"] = True
            entry["error"] = None
            save_progress(run_dir, progress)
            _advance("done")
        except Exception as e:
            _fail("export", e); return
    elif not can_export:
        _log(f"  [skip] Export deferred — both queries and documents must be translated first.")
    else:
        _log(f"  [skip] Already exported to BeIR format.")

    _set_phase("[green]complete")
    if rich_progress is not None and overall_task_id is not None:
        rich_progress.update(overall_task_id, advance=1)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_beir_translation_pipeline(
    config_path: str,
    text_type: str = "both",
    yes: bool = False,
) -> None:
    config = load_config(config_path)
    run_id = make_run_id(config)
    # Write the resolved run_id back so it's consistent across resumptions
    config["run_id"] = run_id

    runs_base = config["paths"]["runs_base"]
    run_dir = os.path.join(runs_base, run_id)

    progress = load_or_init_progress(run_dir, config, run_id)
    execution_mode = resolve_execution_mode(config)

    # Shared translation cache — keyed by (model, prompt) so it's safe to reuse
    # across all datasets in this run and across future runs with the same settings.
    model_slug = _slugify(config["model"]["name"])
    prompt_slug = _slugify(Path(config["prompt"]["file"]).stem)
    cache_path = os.path.join(runs_base, "cache", f"{model_slug}__{prompt_slug}.jsonl")
    cache = TranslationCache(cache_path)

    dataset_names = config["datasets"]["names"]
    pilot_n = config.get("progression", {}).get("pilot_n", 0)
    _console.print(f"\n[bold]BeIR Translation Pipeline[/bold]")
    _console.print(f"  Run ID:     {run_id}")
    _console.print(f"  Mode:       {execution_mode}")
    _console.print(f"  Text type:  {text_type}")
    _console.print(f"  Model:      {config['model']['name']}")
    _console.print(f"  Prompt:     {config['prompt']['file']}")
    _console.print(f"  Datasets:   {len(dataset_names)}")
    _console.print(f"  Pilot rows: {pilot_n if pilot_n > 0 else 'disabled'}")
    _console.print(f"  Output dir: {run_dir}")
    _console.print(f"  Cache:      {cache_path} ({len(cache)} entries)\n")

    errors = []

    global _active_progress
    with _make_progress() as rich_progress:
        _active_progress = rich_progress
        overall_task_id = rich_progress.add_task(
            "[bold]Overall[/bold]",
            total=len(dataset_names),
            phase="",
        )

        for dataset_name in dataset_names:
            dataset_slug = _dataset_slug(dataset_name)
            if dataset_slug not in progress["datasets"]:
                # Dataset added to config after run started — add it to progress
                progress["datasets"][dataset_slug] = {
                    "dataset_name": dataset_name,
                    "candidates_built": False,
                    "queries_translated": False,
                    "titles_translated": False,
                    "documents_translated": False,
                    "exported_to_beir": False,
                    "queries_batch_job_id": None,
                    "titles_batch_job_id": None,
                    "docs_batch_job_id": None,
                    "error": None,
                }
                save_progress(run_dir, progress)

            _process_dataset(
                dataset_name, config, run_id, run_dir, progress, execution_mode, cache,
                text_type=text_type, yes=yes,
                rich_progress=rich_progress, overall_task_id=overall_task_id,
            )

            if progress["datasets"][dataset_slug]["error"]:
                errors.append(dataset_name)

        _active_progress = None

    _console.print(f"\n[bold]Pipeline complete.[/bold] "
                   f"{len(dataset_names) - len(errors)}/{len(dataset_names)} datasets succeeded.")
    if errors:
        _console.print(f"[red]Failed:[/red] {errors}")
        _console.print("Re-run the same command to retry failed datasets.")
    _console.print(f"Progress file: {os.path.join(run_dir, 'progress.json')}")


def main():
    parser = argparse.ArgumentParser(description="Run the BeIR translation pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config file.")
    parser.add_argument(
        "--text-type", default="both", choices=["query", "document", "both"],
        help="Which text type to translate: 'query', 'document', or 'both' (default).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip cost confirmation prompts (non-interactive / CI mode).",
    )
    args = parser.parse_args()
    run_beir_translation_pipeline(
        config_path=args.config,
        text_type=args.text_type,
        yes=args.yes,
    )


if __name__ == "__main__":
    main()

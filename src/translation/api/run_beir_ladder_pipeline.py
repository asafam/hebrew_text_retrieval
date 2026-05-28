"""
Fixed-shard ladder translation pipeline for BeIR datasets.

For each dataset, reads a shard_manifest.json and processes shards in order:
  1. Translate queries shard  → {run_dir}/{slug}/shards/
  2. Translate documents shard
  3. Append both to accumulated CSVs
  4. Judge a sample from the accumulated CSV
  5. Persist scores to qa_scores.csv + progress.json
  6. Update plots
  7. Gate: stop this dataset on QA fail; continue on pass

Candidate shards are built by:
  python -m translation.build_translation_candidates \\
      --shard-size <N> --output-path outputs/translation/candidates ...

Safety:
  If an in-progress run is detected, the pipeline exits with instructions.
  Use --resume to explicitly opt in to continuing that run.

Usage:
  python -m translation.api.run_beir_ladder_pipeline \\
      --config config/translation/full_corpus.yaml \\
      [--dataset BeIR/nfcorpus] \\
      [--resume] \\
      [--dry-run]
"""

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from translation.api.run_beir_translation_pipeline import (
    load_config,
    _dataset_slug,
    save_progress,
)
from translation.api.run_beir_batch_gcs import (
    _upload_kwargs,
    _id_columns_for,
    _get_gcs_config,
    _validate_gcs_auth,
    _make_gemini_client,
)
from translation.api.translate_batch_gemini import TERMINAL_STATES, FAILED_STATES
from translation.api.translate_batch_gemini_gcs import (
    get_gcs_client,
    build_and_upload_input,
    submit_gcs_batch_job,
    check_job_status,
    write_translated_csv,
    _strip_gs_uri,
)
from translation.api.plot_ladder_scores import render_plots

logger = logging.getLogger(__name__)


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(run_dir: str) -> None:
    log_path = os.path.join(run_dir, "run.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
        root.addHandler(logging.StreamHandler(sys.stdout))


# ── Config helpers ─────────────────────────────────────────────────────────────

def _ladder_candidates_base(config: dict) -> str:
    return config.get("paths", {}).get("ladder_candidates_base", "outputs/translation/candidates")


def _ladder_runs_base(config: dict) -> str:
    return config.get("paths", {}).get("ladder_runs_base", "outputs/translation/runs")


# ── Run directory management ──────────────────────────────────────────────────

def _find_existing_run_dir(runs_base: str, run_id: str) -> Optional[str]:
    """Return the most recent run_dir whose name contains run_id, or None."""
    if not os.path.isdir(runs_base):
        return None
    candidates = [
        d for d in os.listdir(runs_base)
        if run_id in d and os.path.isfile(os.path.join(runs_base, d, "progress.json"))
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)  # ISO timestamp prefix — latest first
    return os.path.join(runs_base, candidates[0])


# ── Progress schema ────────────────────────────────────────────────────────────

def _empty_ladder_entry(dataset_name: str) -> dict:
    return {
        "dataset_name": dataset_name,
        "ladder_current_stage": 0,
        "ladder_stopped": False,
        "ladder_all_done": False,
        "ladder_stop_reason": None,
        "ladder_stage_scores": {},  # str(stage_idx) → score dict
    }


def _load_or_init_progress(run_dir: str, config: dict, run_id: str) -> dict:
    path = os.path.join(run_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        logger.info(f"Resuming ladder run '{run_id}' from {path}")
        return p
    dataset_names = config["datasets"]["names"]
    p = {
        "run_id": run_id,
        "config_file": config.get("_config_path", ""),
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "datasets": {
            _dataset_slug(n): _empty_ladder_entry(n) for n in dataset_names
        },
    }
    os.makedirs(run_dir, exist_ok=True)
    save_progress(run_dir, p)
    return p


# ── Manifest ───────────────────────────────────────────────────────────────────

def _load_manifest(manifest_path: str) -> Optional[dict]:
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f)


# ── Translation (GCS batch) ────────────────────────────────────────────────────

def _gcs_shard_prefix(run_id: str, slug: str, shard_idx: int, text_type: str) -> str:
    return f"beir_ladder/{run_id}/{slug}/shard_{shard_idx:03d}/{text_type}"


def _download_shard_results(gcs_client, bucket: str, gcs_output_prefix: str) -> tuple:
    """
    Download prediction shards from GCS.
    Returns (translations, input_tokens, output_tokens).
    Reads actual token counts from usageMetadata in each response line.
    """
    bucket_obj = gcs_client.bucket(bucket)
    blobs = sorted(
        [b for b in bucket_obj.list_blobs(prefix=gcs_output_prefix)
         if "prediction" in os.path.basename(b.name) and b.name.endswith(".jsonl")],
        key=lambda b: b.name,
    )
    if not blobs:
        raise RuntimeError(
            f"No prediction shards found at gs://{bucket}/{gcs_output_prefix}"
        )
    results, input_tokens, output_tokens = [], 0, 0
    for blob in blobs:
        for raw_line in blob.download_as_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
                text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                translation = json.loads(text).get("translation", text)
                usage = obj["response"].get("usageMetadata", {})
                input_tokens  += usage.get("promptTokenCount",     0)
                output_tokens += usage.get("candidatesTokenCount", 0)
            except Exception:
                translation = ""
            results.append({"translation": translation})
    return results, input_tokens, output_tokens


def _compute_cost(input_tokens: int, output_tokens: int, config: dict) -> float:
    """Compute USD cost from actual token counts using guardrails pricing."""
    g = config.get("guardrails", {})
    cost_in  = g.get("cost_per_1m_input_tokens",  0.0)
    cost_out = g.get("cost_per_1m_output_tokens", 0.0)
    return (input_tokens * cost_in + output_tokens * cost_out) / 1_000_000


def _translate_shard_batch(
    shard_csv: str,
    output_path: str,
    text_type: str,
    type_cfg: dict,
    run_id: str,
    slug: str,
    shard_idx: int,
    gemini_client,
    gcs_client,
    bucket: str,
    poll_interval: int,
    max_wait_seconds: int,
    config: dict,
) -> tuple:
    """
    Translate one shard via GCS batch.
    Returns (output_csv_path, input_tokens, output_tokens).
    Raises on job failure or timeout.
    """
    df = pd.read_csv(shard_csv, encoding="utf-8")
    id_cols = _id_columns_for(df)
    gcs_prefix = _gcs_shard_prefix(run_id, slug, shard_idx, text_type)
    output_prefix = f"gs://{bucket}/{gcs_prefix}/output"

    # Upload input JSONL to GCS
    input_uri = build_and_upload_input(
        df=df,
        id_columns=id_cols,
        gcs_client=gcs_client,
        bucket=bucket,
        gcs_prefix=gcs_prefix,
        **_upload_kwargs(type_cfg),
    )

    # Submit batch job
    display_name = f"{run_id}__{slug}__shard{shard_idx:03d}__{text_type}"
    job_name = submit_gcs_batch_job(
        gemini_client, type_cfg["model"], input_uri, output_prefix,
        display_name=display_name,
    )
    logger.info(f"  Batch job submitted: {job_name}")

    # Poll until terminal
    waited = 0
    while True:
        status = check_job_status(gemini_client, job_name)
        if status in TERMINAL_STATES:
            break
        if waited >= max_wait_seconds:
            raise RuntimeError(
                f"Batch job {job_name} did not complete within "
                f"{max_wait_seconds // 3600}h. Status: {status}"
            )
        logger.info(
            f"  [{slug}] shard {shard_idx} {text_type} — {status} "
            f"(waited {waited // 60}m, polling every {poll_interval // 60}m)"
        )
        time.sleep(poll_interval)
        waited += poll_interval

    if status in FAILED_STATES:
        raise RuntimeError(f"Batch job {job_name} failed with state: {status}")

    # Download results and extract token counts
    _, gcs_output_path = _strip_gs_uri(output_prefix)
    translations, input_tokens, output_tokens = _download_shard_results(
        gcs_client, bucket, gcs_output_path
    )
    logger.info(
        f"  [{slug}] shard {shard_idx} {text_type} — "
        f"{input_tokens:,} input tokens, {output_tokens:,} output tokens"
    )

    # Write translated CSV
    write_translated_csv(translations, shard_csv, output_path)
    return output_path, input_tokens, output_tokens


def _append_to_accumulated(shard_out_csv: str, accumulated_csv: str) -> int:
    """Append translated shard rows to accumulated CSV. Returns total row count."""
    if not os.path.exists(shard_out_csv):
        logger.warning(f"Shard output not found: {shard_out_csv}")
        return 0
    shard_df = pd.read_csv(shard_out_csv, encoding="utf-8")
    header_needed = not os.path.exists(accumulated_csv)
    os.makedirs(os.path.dirname(accumulated_csv) or ".", exist_ok=True)
    shard_df.to_csv(accumulated_csv, mode="a", header=header_needed, index=False, encoding="utf-8")
    # Count lines without fully deserializing (header line = 1 overhead)
    with open(accumulated_csv, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


# ── QA ─────────────────────────────────────────────────────────────────────────

def _ladder_qa(
    accumulated_csv: str,
    slug: str,
    text_type: str,
    config: dict,
    run_dir: str,
    stage_idx: int,
) -> dict:
    """
    Sample from accumulated_csv, judge with LLM.
    Returns {"passed": bool, "score_mean": float|None, "score_std": float|None, "n": int}.
    """
    qa_cfg = config.get("qa", {})
    if not qa_cfg.get("enabled", False):
        return {"passed": True, "score_mean": None, "score_std": None, "n": 0}

    from translation.api.evaluate_translations import run_evaluate_translations
    from translation.qa_phase import DATASET_EVAL_PROMPTS

    min_score      = qa_cfg.get("min_score", 3.5)
    sample_size    = qa_cfg.get("sample_size", 25)
    sample_seed    = qa_cfg.get("sample_seed", 42)
    judge_model    = qa_cfg.get("judge_model", "gemini-2.5-pro")
    judge_location = qa_cfg.get("judge_location")
    sleep_time     = qa_cfg.get("sleep_time", 0)

    df = pd.read_csv(accumulated_csv, encoding="utf-8")
    df = df[df["translation"].notna()]
    if df.empty:
        logger.warning(f"[qa] No translated rows in {accumulated_csv}")
        return {"passed": True, "score_mean": None, "score_std": None, "n": 0}

    sample = df.sample(n=min(sample_size, len(df)), random_state=sample_seed + stage_idx)
    text_col = "text" if text_type == "query" else "segment_text"

    suffix = f"_s{stage_idx:03d}_{text_type}.csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as tmp:
        sample.to_csv(tmp, index=False)
        tmp_path = tmp.name

    prompt_file = DATASET_EVAL_PROMPTS.get(
        slug,
        "prompts/translation/api/evaluation/translation_evaluation_nogold_v20250406.yaml",
    )
    tmp_out_dir = tempfile.mkdtemp()

    prev_location = os.environ.get("GEMINI_LOCATION")
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
            force=True,
            sleep_time=sleep_time,
            english_key=text_col,
            hebrew_key="translation",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        if judge_location:
            if prev_location is not None:
                os.environ["GEMINI_LOCATION"] = prev_location
            else:
                os.environ.pop("GEMINI_LOCATION", None)

    if evaluated is None or "score" not in evaluated.columns:
        logger.warning(f"[qa] No scores returned for {slug}/{text_type} at stage {stage_idx}")
        return {"passed": True, "score_mean": None, "score_std": None, "n": 0}

    valid = evaluated["score"].dropna()
    if valid.empty:
        return {"passed": True, "score_mean": None, "score_std": None, "n": 0}

    mean = float(valid.mean())
    std  = float(valid.std()) if len(valid) > 1 else 0.0
    n    = int(len(valid))

    # Baseline comparison (if configured); fall back to absolute min_score threshold
    baseline_csv = qa_cfg.get("baseline_csv", "")
    passed = True
    reason = ""
    if baseline_csv and os.path.exists(baseline_csv):
        try:
            from translation.qa_phase import compare_scores, load_baseline
            bdf = load_baseline(
                baseline_csv,
                qa_cfg.get("baseline_model", ""),
                qa_cfg.get("baseline_prompt", ""),
                judge_model,
            )
            bstats = bdf.groupby(["dataset_slug", "text_type"])["score"].agg(["mean", "std"])
            idx = (slug, text_type)
            if idx in bstats.index:
                b_mean = float(bstats.loc[idx, "mean"])
                b_std  = float(bstats.loc[idx, "std"])
                degraded, reason = compare_scores(mean, b_mean, b_std)
                passed = not degraded
                if passed and mean < min_score:
                    passed = False
                    reason = f"score {mean:.2f} < min_score {min_score}"
            else:
                passed = mean >= min_score
        except Exception as e:
            logger.warning(f"[qa] Baseline comparison failed ({e}); using min_score threshold")
            passed = mean >= min_score
    else:
        passed = mean >= min_score

    status = "PASS" if passed else "FAIL"
    logger.info(
        f"  [qa] {slug}/{text_type} stage={stage_idx}: [{status}] "
        f"score={mean:.3f}±{std:.3f} n={n}"
        + (f"  ← {reason}" if reason else "")
    )
    _save_ladder_qa_history(run_dir, config, slug, text_type, stage_idx, n, mean, std, passed)
    return {"passed": passed, "score_mean": mean, "score_std": std, "n": n}


def _save_ladder_qa_history(
    run_dir: str, config: dict, slug: str, text_type: str,
    stage: int, n: int, mean: float, std: float, passed: bool,
) -> None:
    run_id = config.get("run_id", "")
    cfg_key = "queries" if text_type == "query" else "documents"
    translation_model = config.get(cfg_key, {}).get("model", "")
    judge_model = config.get("qa", {}).get("judge_model", "")
    history_path = str(Path(run_dir).parent.parent / "qa_history.csv")
    row = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "dataset_slug": slug,
        "text_type": text_type,
        "stage": stage,
        "translation_model": translation_model,
        "judge_model": judge_model,
        "sample_size": n,
        "score_mean": round(mean, 4),
        "score_std": round(std, 4),
        "passed": passed,
    }
    write_header = not os.path.exists(history_path)
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_qa_scores(
    run_dir: str,
    run_id: str,
    slug: str,
    stage: int,
    shard_meta: dict,
    q_result: dict,
    d_result: dict,
    cumulative_q_rows: int,
    cumulative_d_rows: int,
    q_input_tokens: int = 0,
    q_output_tokens: int = 0,
    d_input_tokens: int = 0,
    d_output_tokens: int = 0,
    shard_cost_usd: float = 0.0,
    cumulative_cost_usd: float = 0.0,
) -> None:
    scores_path = os.path.join(run_dir, "qa_scores.csv")
    overall = q_result.get("passed", True) and d_result.get("passed", True)

    def _fmt(v):
        return round(v, 4) if v is not None else ""

    row = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "dataset_slug": slug,
        "stage": stage,
        "shard_size": shard_meta.get("rows", 0),
        "cumulative_q_rows": cumulative_q_rows,
        "cumulative_d_rows": cumulative_d_rows,
        "q_score_mean": _fmt(q_result.get("score_mean")),
        "q_score_std":  _fmt(q_result.get("score_std")),
        "q_passed": q_result.get("passed"),
        "d_score_mean": _fmt(d_result.get("score_mean")),
        "d_score_std":  _fmt(d_result.get("score_std")),
        "d_passed": d_result.get("passed"),
        "overall_passed": overall,
        "q_input_tokens": q_input_tokens,
        "q_output_tokens": q_output_tokens,
        "d_input_tokens": d_input_tokens,
        "d_output_tokens": d_output_tokens,
        "shard_cost_usd": round(shard_cost_usd, 6),
        "cumulative_cost_usd": round(cumulative_cost_usd, 6),
    }
    write_header = not os.path.exists(scores_path)
    with open(scores_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Dry run ────────────────────────────────────────────────────────────────────

def _dry_run(config: dict, dataset_filter: Optional[str]) -> None:
    candidates_base = _ladder_candidates_base(config)
    datasets = config["datasets"]["names"]
    if dataset_filter:
        datasets = [d for d in datasets if _dataset_slug(d) == dataset_filter or d == dataset_filter]
    print(f"\nDry-run shard plan — candidates: {candidates_base}\n")
    for dname in datasets:
        slug = _dataset_slug(dname)
        manifest_path = os.path.join(candidates_base, slug, "shard_manifest.json")
        if not os.path.exists(manifest_path):
            print(f"  {slug}: [NO MANIFEST] — build candidates with --shard-size first")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        q_shards = manifest["types"]["queries"]
        d_shards = manifest["types"]["documents"]
        total_q = sum(s["rows"] for s in q_shards)
        total_d = sum(s["rows"] for s in d_shards)
        shard_sz = manifest.get("shard_size", "?")
        shard_sz_fmt = f"{shard_sz:,}" if isinstance(shard_sz, int) else shard_sz
        print(
            f"  {slug}: {len(q_shards)} query shards (shard size={shard_sz_fmt}) "
            f"= {total_q:,} queries | "
            f"{len(d_shards)} doc shards = {total_d:,} documents"
        )
    print()


# ── Main ladder loop ───────────────────────────────────────────────────────────

def run_ladder(
    config: dict,
    run_dir: str,
    progress: dict,
    dataset_filter: Optional[str],
    gemini_client,
    gcs_client,
    bucket: str,
) -> None:
    run_id = progress["run_id"]
    candidates_base = _ladder_candidates_base(config)
    q_cfg = config["queries"]
    d_cfg = config["documents"]
    batch_cfg = config.get("batch", {})
    poll_interval   = batch_cfg.get("poll_interval_seconds", 3600)
    max_wait_seconds = int(batch_cfg.get("max_wait_hours", 72) * 3600)

    datasets = config["datasets"]["names"]
    if dataset_filter:
        datasets = [d for d in datasets if _dataset_slug(d) == dataset_filter or d == dataset_filter]

    cumulative_cost_usd = progress.get("total_cost_usd", 0.0)

    for dataset_name in datasets:
        slug = _dataset_slug(dataset_name)
        entry = progress["datasets"].setdefault(slug, _empty_ladder_entry(dataset_name))

        if entry.get("ladder_all_done"):
            logger.info(f"[{slug}] All shards done — skipping.")
            continue
        if entry.get("ladder_stopped"):
            logger.info(
                f"[{slug}] Ladder stopped — skipping. "
                f"Reason: {entry.get('ladder_stop_reason')}"
            )
            continue

        manifest_path = os.path.join(candidates_base, slug, "shard_manifest.json")
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            logger.warning(
                f"[{slug}] No shard manifest at {manifest_path} — skipping. "
                f"Build candidates with --shard-size first."
            )
            continue

        q_shards = manifest["types"]["queries"]
        d_shards = manifest["types"]["documents"]
        logger.info(
            f"[{slug}] Starting ladder: "
            f"{len(q_shards)} query shards, {len(d_shards)} document shards"
        )

        dataset_run_dir = os.path.join(run_dir, slug)
        shard_out_dir   = os.path.join(dataset_run_dir, "shards")
        os.makedirs(shard_out_dir, exist_ok=True)

        q_accumulated = os.path.join(dataset_run_dir, "queries_accumulated.csv")
        d_accumulated = os.path.join(dataset_run_dir, "documents_accumulated.csv")

        current_stage = entry.get("ladder_current_stage", 0)
        dataset_stopped = False

        for shard_meta in q_shards:
            idx = shard_meta["index"]
            if idx < current_stage:
                continue  # already processed in a prior run

            d_shard_meta = d_shards[idx] if idx < len(d_shards) else None
            q_rows = shard_meta["rows"]
            d_rows = d_shard_meta["rows"] if d_shard_meta else 0
            logger.info(f"[{slug}] Shard {idx}: {q_rows} query rows + {d_rows} document rows")

            # ── Translate queries shard via GCS batch ─────────────────────────
            q_shard_csv = os.path.join(candidates_base, slug, shard_meta["file"])
            if not os.path.exists(q_shard_csv):
                logger.error(f"[{slug}] Query shard file missing: {q_shard_csv}")
                entry["ladder_stopped"] = True
                entry["ladder_stop_reason"] = f"shard {idx}: query shard file missing"
                save_progress(run_dir, progress)
                dataset_stopped = True
                break

            q_out_path = os.path.join(shard_out_dir, shard_meta["file"].replace(".csv", "_translated.csv"))
            logger.info(f"[{slug}] Submitting queries shard {idx} to GCS batch")
            try:
                q_out, q_in_tok, q_out_tok = _translate_shard_batch(
                    shard_csv=q_shard_csv, output_path=q_out_path,
                    text_type="queries", type_cfg=q_cfg,
                    run_id=run_id, slug=slug, shard_idx=idx,
                    gemini_client=gemini_client, gcs_client=gcs_client, bucket=bucket,
                    poll_interval=poll_interval, max_wait_seconds=max_wait_seconds,
                    config=config,
                )
            except Exception as e:
                logger.error(f"[{slug}] Query shard {idx} batch failed: {e}")
                entry["ladder_stopped"] = True
                entry["ladder_stop_reason"] = f"shard {idx}: query batch failed: {e}"
                save_progress(run_dir, progress)
                dataset_stopped = True
                break

            # ── Translate documents shard via GCS batch ───────────────────────
            d_out = None
            d_in_tok = d_out_tok = 0
            if d_shard_meta:
                d_shard_csv = os.path.join(candidates_base, slug, d_shard_meta["file"])
                if not os.path.exists(d_shard_csv):
                    logger.error(f"[{slug}] Document shard file missing: {d_shard_csv}")
                    entry["ladder_stopped"] = True
                    entry["ladder_stop_reason"] = f"shard {idx}: document shard file missing"
                    save_progress(run_dir, progress)
                    dataset_stopped = True
                    break
                d_out_path = os.path.join(shard_out_dir, d_shard_meta["file"].replace(".csv", "_translated.csv"))
                logger.info(f"[{slug}] Submitting documents shard {idx} to GCS batch")
                try:
                    d_out, d_in_tok, d_out_tok = _translate_shard_batch(
                        shard_csv=d_shard_csv, output_path=d_out_path,
                        text_type="documents", type_cfg=d_cfg,
                        run_id=run_id, slug=slug, shard_idx=idx,
                        gemini_client=gemini_client, gcs_client=gcs_client, bucket=bucket,
                        poll_interval=poll_interval, max_wait_seconds=max_wait_seconds,
                        config=config,
                    )
                except Exception as e:
                    logger.error(f"[{slug}] Document shard {idx} batch failed: {e}")
                    entry["ladder_stopped"] = True
                    entry["ladder_stop_reason"] = f"shard {idx}: document batch failed: {e}"
                    save_progress(run_dir, progress)
                    dataset_stopped = True
                    break

            # ── Cost tracking ─────────────────────────────────────────────────
            shard_cost = _compute_cost(q_in_tok + d_in_tok, q_out_tok + d_out_tok, config)
            cumulative_cost_usd += shard_cost
            progress["total_cost_usd"] = round(cumulative_cost_usd, 6)
            logger.info(
                f"[{slug}] Shard {idx} cost: ${shard_cost:.4f}  "
                f"(run total: ${cumulative_cost_usd:.4f})"
            )

            # ── Accumulate ────────────────────────────────────────────────────
            cumulative_q = _append_to_accumulated(q_out, q_accumulated)
            cumulative_d = _append_to_accumulated(d_out, d_accumulated) if d_out else 0
            logger.info(
                f"[{slug}] Shard {idx} accumulated: "
                f"{cumulative_q:,} query rows, {cumulative_d:,} document rows"
            )

            # ── Judge ─────────────────────────────────────────────────────────
            q_result = _ladder_qa(q_accumulated, slug, "query",    config, run_dir, idx)
            d_result = _ladder_qa(d_accumulated, slug, "document", config, run_dir, idx) \
                if d_out else {"passed": True, "score_mean": None, "score_std": None, "n": 0}

            # ── Persist ───────────────────────────────────────────────────────
            _append_qa_scores(
                run_dir, run_id, slug, idx, shard_meta,
                q_result, d_result, cumulative_q, cumulative_d,
                q_input_tokens=q_in_tok, q_output_tokens=q_out_tok,
                d_input_tokens=d_in_tok, d_output_tokens=d_out_tok,
                shard_cost_usd=shard_cost,
                cumulative_cost_usd=cumulative_cost_usd,
            )
            entry["ladder_stage_scores"][str(idx)] = {
                "q_score_mean": q_result["score_mean"],
                "q_score_std":  q_result["score_std"],
                "d_score_mean": d_result["score_mean"],
                "d_score_std":  d_result["score_std"],
                "passed": q_result["passed"] and d_result["passed"],
                "cumulative_q_rows": cumulative_q,
                "cumulative_d_rows": cumulative_d,
                "shard_cost_usd": round(shard_cost, 6),
                "cumulative_cost_usd": round(cumulative_cost_usd, 6),
                "timestamp": datetime.now().isoformat(),
            }
            entry["ladder_current_stage"] = idx + 1
            save_progress(run_dir, progress)

            # ── Plots ─────────────────────────────────────────────────────────
            try:
                render_plots(run_dir, progress, config)
            except Exception as e:
                logger.warning(f"[{slug}] Plot update failed (non-fatal): {e}")

            # ── Gate ──────────────────────────────────────────────────────────
            if not (q_result["passed"] and d_result["passed"]):
                qm = f"{q_result['score_mean']:.2f}" if q_result.get("score_mean") is not None else "N/A"
                dm = f"{d_result['score_mean']:.2f}" if d_result.get("score_mean") is not None else "N/A"
                entry["ladder_stopped"] = True
                entry["ladder_stop_reason"] = (
                    f"stage {idx} QA failed (q_score={qm}, d_score={dm})"
                )
                save_progress(run_dir, progress)
                logger.warning(f"[{slug}] Ladder stopped: {entry['ladder_stop_reason']}")
                dataset_stopped = True
                break

        if not dataset_stopped:
            entry["ladder_all_done"] = True
            save_progress(run_dir, progress)
            logger.info(f"[{slug}] All {len(q_shards)} shards completed.")

    logger.info(f"Run total cost: ${cumulative_cost_usd:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-shard ladder translation pipeline for BeIR datasets."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset", default=None,
        help="Run only one dataset (accepts slug like 'BeIR_nfcorpus' or name 'BeIR/nfcorpus').",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Explicitly resume an in-progress run. "
            "Required when restarting after a kill — this is your deliberate choice."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print the shard plan without translating.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = config.get("run_id", "ladder_run")
    runs_base = _ladder_runs_base(config)

    if args.dry_run:
        _dry_run(config, args.dataset)
        return

    existing = _find_existing_run_dir(runs_base, run_id)

    if args.resume:
        if existing is None:
            print(
                f"ERROR: --resume specified but no existing run found for "
                f"run_id='{run_id}' in {runs_base}",
                file=sys.stderr,
            )
            sys.exit(1)
        run_dir = existing
        print(f"Resuming run: {run_dir}")
    else:
        if existing is not None:
            print(
                f"\nERROR: Found an in-progress run at:\n"
                f"  {existing}\n\n"
                f"This run was interrupted. It is YOUR decision what to do:\n"
                f"  • Resume it:     pass --resume to continue from where it stopped\n"
                f"  • Start fresh:   change 'run_id' in the config to create a new run\n\n"
                f"Exiting without changes.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_base, f"{ts}_{run_id}")

    os.makedirs(run_dir, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"Ladder pipeline started. Run dir: {run_dir}")

    _validate_gcs_auth()
    project, bucket, location = _get_gcs_config(config)
    os.environ["GEMINI_PROJECT"] = project
    os.environ["GEMINI_LOCATION"] = location
    gemini_client = _make_gemini_client(project, location)
    gcs_client    = get_gcs_client(project)
    logger.info(f"GCS batch: project={project}, bucket={bucket}, location={location}")

    progress = _load_or_init_progress(run_dir, config, run_id)
    run_ladder(config, run_dir, progress, args.dataset, gemini_client, gcs_client, bucket)

    total_cost = progress.get("total_cost_usd", 0.0)
    logger.info(f"Ladder pipeline complete. Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()

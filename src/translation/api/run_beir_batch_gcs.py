"""
BeIR GCS Batch Translation Pipeline.

Three-phase pipeline that submits all Gemini batch jobs simultaneously via GCS,
instead of the sequential blocking approach in run_beir_translation_pipeline.py.

Phases:
  pilot   — translate first N rows synchronously, run QA gates per dataset
  submit  — upload full JSONL to GCS, fire all batch jobs at once
  collect — check job status, download results, QA, export; print rich report

Usage:
  python -m translation.api.run_beir_batch_gcs --config config/translation/full_corpus.yaml pilot
  python -m translation.api.run_beir_batch_gcs --config config/translation/full_corpus.yaml submit --yes
  python -m translation.api.run_beir_batch_gcs --config config/translation/full_corpus.yaml collect
  python -m translation.api.run_beir_batch_gcs --config config/translation/full_corpus.yaml collect --wait

Required env vars:
  GEMINI_PROJECT  — GCP project (already in .env)
  GEMINI_API_KEY must be UNSET — this pipeline uses Vertex AI ADC auth only.

GCS bucket and location are read from config YAML (gcs.bucket / gcs.location),
falling back to GCS_BUCKET env var if the yaml field is blank.

Auth: run once before using this pipeline:
  gcloud auth application-default login
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.table import Table

load_dotenv()

# ── Reuse all helpers from the existing pipeline ──────────────────────────────
from translation.api.run_beir_translation_pipeline import (
    load_config,
    make_run_id,
    load_or_init_progress as _load_or_init_progress,
    save_progress as _save_progress,
    _phase_build_candidates,
    _run_dataset_qa,
    _apply_cache_and_dedup,
    _expand_dedup_and_update_cache,
    _dataset_slug,
    _empty_dataset_entry,
    _estimate_and_confirm,
    _console,
    _log,
    _slugify,
)
from translation.api.translate import run_translation_pipeline as run_parallel_pipeline
from translation.api.translate_batch_gemini import TERMINAL_STATES, FAILED_STATES
from translation.api.translate_batch_gemini_gcs import (
    get_gcs_client,
    build_and_upload_input,
    submit_gcs_batch_job,
    check_job_status,
    download_and_parse_results,
    write_translated_csv,
    _strip_gs_uri,
)
from translation.api.utils import load_data, Translation
from translation.beir.export import export_to_beir_jsonl
from translation.beir.cache import TranslationCache


# ── Progress schema extensions ────────────────────────────────────────────────

GCS_PROGRESS_KEYS = {
    "queries_gcs_input_uri":    None,
    "queries_gcs_output_prefix": None,
    "queries_batch_job_name":   None,
    "queries_submitted_at":     None,
    "queries_completed_at":     None,

    "docs_gcs_input_uri":       None,
    "docs_gcs_output_prefix":   None,
    "docs_batch_job_name":      None,
    "docs_submitted_at":        None,
    "docs_completed_at":        None,

    "titles_gcs_input_uri":     None,
    "titles_gcs_output_prefix": None,
    "titles_batch_job_name":    None,
    "titles_submitted_at":      None,
    "titles_completed_at":      None,
}


# This pipeline keeps its own progress file inside the shared run dir, so it
# never collides with the ladder's progress.json (different entry schema).
_BATCH_PROGRESS_FILE = "progress.batch.json"


def load_or_init_progress(run_dir: str, config: dict, run_id: str) -> dict:
    return _load_or_init_progress(run_dir, config, run_id, filename=_BATCH_PROGRESS_FILE)


def save_progress(run_dir: str, progress: dict) -> None:
    _save_progress(run_dir, progress, filename=_BATCH_PROGRESS_FILE)


def _patch_gcs_keys(progress: dict) -> None:
    """Ensure all dataset entries have the GCS-specific progress keys."""
    for entry in progress["datasets"].values():
        for k, v in GCS_PROGRESS_KEYS.items():
            entry.setdefault(k, v)


# ── GCS path helpers ──────────────────────────────────────────────────────────

def _gcs_prefix(run_id: str, dataset_slug: str, text_type: str) -> str:
    return f"translation/{run_id}/corpus/{dataset_slug}/{text_type}"


def _gcs_input_uri(bucket: str, run_id: str, dataset_slug: str, text_type: str) -> str:
    return f"gs://{bucket}/{_gcs_prefix(run_id, dataset_slug, text_type)}/input.jsonl"


def _gcs_output_uri(bucket: str, run_id: str, dataset_slug: str, text_type: str) -> str:
    return f"gs://{bucket}/{_gcs_prefix(run_id, dataset_slug, text_type)}/output"


# ── Auth validation ───────────────────────────────────────────────────────────

def _validate_gcs_auth() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GCS batch mode requires Vertex AI (gcloud ADC) authentication.\n"
            "Please unset GEMINI_API_KEY and run: gcloud auth application-default login"
        )


def _get_gcs_config(config: dict) -> tuple:
    """Return (project, bucket, location) from config, falling back to env vars."""
    gcs = config.get("gcs", {})
    project = gcs.get("project") or os.environ.get("GEMINI_PROJECT", "")
    bucket = gcs.get("bucket") or os.environ.get("GCS_BUCKET", "")
    location = gcs.get("location") or os.environ.get("GEMINI_LOCATION", "us-central1")

    if not project:
        raise RuntimeError(
            "GCP project not configured. Set 'gcs.project' in your config YAML "
            "or export GEMINI_PROJECT=your-project in .env"
        )
    if not bucket:
        raise RuntimeError(
            "GCS bucket not configured. Set 'gcs.bucket' in your config YAML "
            "or export GCS_BUCKET=your-bucket-name in .env"
        )
    return project, bucket, location


def _make_gemini_client(project: str, location: str):
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


# ── Shared per-dataset helpers ────────────────────────────────────────────────

def _build_run_kwargs(type_cfg: dict, exec_cfg: dict, output_dir: str, force: bool = False) -> dict:
    """Build run_parallel_pipeline kwargs from a per-type config block."""
    p = type_cfg["prompt"]
    hk = p.get("hebrew_key", "Hebrew")
    return dict(
        output_dir=output_dir,
        prompt_file_name=p["file"],
        model_name=type_cfg["model"],
        id_field="_id",
        force=force or exec_cfg.get("force_translation", False),
        parallel=True,
        num_workers=exec_cfg.get("num_workers", 8),
        english_key=p.get("english_key", "Text"),
        hebrew_key=hk,
        context_key=p.get("context_key", "Context"),
        hebrew_key_query=hk,
        hebrew_key_document=hk,
        response_format=Translation,
        sleep_time=exec_cfg.get("sleep_time", 0),
    )


def _upload_kwargs(type_cfg: dict) -> dict:
    """Build build_and_upload_input kwargs from a per-type config block."""
    p = type_cfg["prompt"]
    hk = p.get("hebrew_key", "Hebrew")
    return dict(
        prompt_file=p["file"],
        prompt_type=p["type"],
        model_name=type_cfg["model"],
        temperature=type_cfg.get("temperature", 0.0),
        english_key=p.get("english_key", "Text"),
        hebrew_key=hk,
        context_key=p.get("context_key", "Context"),
        hebrew_key_query=hk,
        hebrew_key_document=hk,
    )


def _id_columns_for(df: pd.DataFrame) -> list:
    cols = ["_id"]
    if "segment_id" in df.columns:
        cols.append("segment_id")
    return cols


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_str(start_iso: Optional[str], end_iso: Optional[str] = None) -> str:
    if not start_iso:
        return ""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso) if end_iso else datetime.now(timezone.utc)
        minutes = int((end - start).total_seconds() / 60)
        h, m = divmod(minutes, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"
    except Exception:
        return ""


def _flatten_shards_to_csv(slug_dir: str, prefix: str, out_csv: str) -> bool:
    """Concatenate the ladder's sharded candidates into one flat CSV.

    {prefix}_shard_000.csv, _001.csv, … → out_csv (e.g. queries.csv). This lets
    the pilot reuse the exact same candidate rows the corpus ladder uses — one
    source of truth — instead of re-fetching/rebuilding from HuggingFace.
    Returns True if shards were found and flattened.
    """
    import glob
    shards = sorted(glob.glob(os.path.join(slug_dir, f"{prefix}_shard_*.csv")))
    if not shards:
        return False
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return True


# ── Phase 1: Pilot ────────────────────────────────────────────────────────────

def run_pilot(
    config: dict,
    run_id: str,
    run_dir: str,
    progress: dict,
    cache: TranslationCache,
    yes: bool,
    dataset_filter: Optional[str],
    qa_only: bool = False,
) -> None:
    """
    Translate first pilot_n rows synchronously per dataset, run QA gates.
    With qa_only=True, skips translation and only re-runs QA on existing translations.
    """
    pilot_n = config.get("progression", {}).get("pilot_n", 100)
    pilot_qa = config.get("progression", {}).get("pilot_qa", True)
    pilot_location = config.get("pilot", {}).get("location")
    candidates_base = os.path.join(run_dir, "candidates")
    exec_cfg = config["execution"]
    q_cfg = config["queries"]
    d_cfg = config["documents"]

    dataset_names = config["datasets"]["names"]
    if dataset_filter:
        dataset_names = [n for n in dataset_names if _dataset_slug(n) == dataset_filter or n == dataset_filter]
        if not dataset_names:
            raise ValueError(f"Dataset '{dataset_filter}' not found in config.")

    mode_label = "QA only" if qa_only else f"{pilot_n} rows per dataset, synchronous"
    _console.print(f"\n[bold]Pilot Phase[/bold] — {mode_label}\n")

    for dataset_name in dataset_names:
        slug = _dataset_slug(dataset_name)
        entry = progress["datasets"].setdefault(slug, _empty_dataset_entry(dataset_name))
        _patch_gcs_keys(progress)

        dataset_run_dir = os.path.join(run_dir, "pilot", slug)
        queries_translated = os.path.join(dataset_run_dir, "queries_translated.csv")
        documents_translated = os.path.join(dataset_run_dir, "documents_translated.csv")

        if qa_only:
            # Skip datasets with no translations yet
            if not os.path.exists(queries_translated) and not os.path.exists(documents_translated):
                _console.print(f"  [dim]SKIP[/dim] {dataset_name} — no translations yet (run without --qa first)")
                continue
            qa_done = (not pilot_qa or
                       (entry.get("queries_pilot_qa_passed") is True and
                        entry.get("documents_pilot_qa_passed") is True))
            if qa_done:
                _console.print(f"  [dim]SKIP[/dim] {dataset_name} — QA already passed")
                continue
        else:
            qa_done = (not pilot_qa or
                       (entry.get("queries_pilot_qa_passed") is True and
                        entry.get("documents_pilot_qa_passed") is True))
            if entry.get("queries_pilot_done") and entry.get("documents_pilot_done") and qa_done:
                _console.print(f"  [dim]SKIP[/dim] {dataset_name} — pilot already done")
                continue

        if (entry.get("error") or "").startswith("pilot_qa_failed"):
            entry["error"] = None  # clear so we can retry QA

        _console.print(f"  [cyan]{dataset_name}[/cyan]")

        if not qa_only:
            # ── Build candidates ───────────────────────────────────────────────
            slug_dir = os.path.join(candidates_base, slug)
            queries_csv = os.path.join(slug_dir, "queries.csv")
            documents_csv = os.path.join(slug_dir, "documents.csv")
            if not entry["candidates_built"] or exec_cfg.get("force_candidates"):
                if os.path.exists(queries_csv) and os.path.exists(documents_csv) and not exec_cfg.get("force_candidates"):
                    entry["candidates_built"] = True
                    save_progress(run_dir, progress)
                elif os.path.exists(os.path.join(slug_dir, "shard_manifest.json")):
                    # Reuse the corpus ladder's sharded candidates (single source
                    # of truth) — flatten them into queries.csv/documents.csv.
                    _flatten_shards_to_csv(slug_dir, "queries", queries_csv)
                    _flatten_shards_to_csv(slug_dir, "documents", documents_csv)
                    entry["candidates_built"] = True
                    save_progress(run_dir, progress)
                    _console.print(f"    [dim]candidates flattened from shards[/dim]")
                else:
                    try:
                        _phase_build_candidates(dataset_name, slug, config)
                        entry["candidates_built"] = True
                        save_progress(run_dir, progress)
                    except Exception as e:
                        entry["error"] = f"candidates: {e}"
                        save_progress(run_dir, progress)
                        _console.print(f"    [red]ERROR[/red] building candidates: {e}")
                        continue

            os.makedirs(dataset_run_dir, exist_ok=True)

            q_text_col = q_cfg["prompt"].get("text_col", "text")
            d_text_col = d_cfg["prompt"].get("text_col", "segment_text")
            q_context_col = q_cfg["prompt"].get("context_col")

            # ── Queries translation ────────────────────────────────────────────
            if not entry.get("queries_pilot_done"):
                _apply_cache_and_dedup(
                    source_csv=queries_csv,
                    output_csv=queries_translated,
                    cache=cache,
                    model_name=q_cfg["model"],
                    prompt_file=q_cfg["prompt"]["file"],
                    text_col=q_text_col,
                    context_col=q_context_col if q_context_col and _has_col(queries_csv, q_context_col) else None,
                )
                _prev_loc = os.environ.get("GEMINI_LOCATION")
                if pilot_location:
                    os.environ["GEMINI_LOCATION"] = pilot_location
                try:
                    run_parallel_pipeline(
                        source_file_path=queries_csv,
                        limit=pilot_n,
                        **_build_run_kwargs(q_cfg, exec_cfg, dataset_run_dir, force=True),
                    )
                    entry["queries_pilot_done"] = True
                    save_progress(run_dir, progress)
                except Exception as e:
                    entry["error"] = f"pilot_queries: {e}"
                    save_progress(run_dir, progress)
                    _console.print(f"    [red]ERROR[/red] pilot queries: {e}")
                    continue
                finally:
                    if pilot_location:
                        if _prev_loc is not None:
                            os.environ["GEMINI_LOCATION"] = _prev_loc
                        else:
                            os.environ.pop("GEMINI_LOCATION", None)

            # ── Documents translation ──────────────────────────────────────────
            if not entry.get("documents_pilot_done"):
                _apply_cache_and_dedup(
                    source_csv=documents_csv,
                    output_csv=documents_translated,
                    cache=cache,
                    model_name=d_cfg["model"],
                    prompt_file=d_cfg["prompt"]["file"],
                    text_col=d_text_col,
                    context_col=None,
                )
                _prev_loc = os.environ.get("GEMINI_LOCATION")
                if pilot_location:
                    os.environ["GEMINI_LOCATION"] = pilot_location
                try:
                    run_parallel_pipeline(
                        source_file_path=documents_csv,
                        limit=pilot_n,
                        **_build_run_kwargs(d_cfg, exec_cfg, dataset_run_dir, force=True),
                    )
                    entry["documents_pilot_done"] = True
                    save_progress(run_dir, progress)
                except Exception as e:
                    entry["error"] = f"pilot_documents: {e}"
                    save_progress(run_dir, progress)
                    _console.print(f"    [red]ERROR[/red] pilot documents: {e}")
                    continue
                finally:
                    if pilot_location:
                        if _prev_loc is not None:
                            os.environ["GEMINI_LOCATION"] = _prev_loc
                        else:
                            os.environ.pop("GEMINI_LOCATION", None)

        # ── Pilot QA ───────────────────────────────────────────────────────────
        q_score, d_score = "-", "-"

        if pilot_qa and not entry.get("queries_pilot_qa_passed"):
            if not os.path.exists(queries_translated):
                _console.print(f"    [yellow]SKIP queries QA[/yellow] — no translated file")
            else:
                q_pass = _run_dataset_qa(queries_translated, slug, "query", config, run_dir)
                entry["queries_pilot_qa_passed"] = q_pass
                q_score = "[green]PASS[/green]" if q_pass else "[red]FAIL[/red]"
                save_progress(run_dir, progress)
                if not q_pass:
                    entry["error"] = "pilot_qa_failed_queries"
                    save_progress(run_dir, progress)
                    _console.print(f"    [red]Pilot QA FAILED[/red] for queries")
                    if not qa_only:
                        continue
        else:
            q_score = "[green]PASS[/green]" if entry.get("queries_pilot_qa_passed") else "skipped"

        if pilot_qa and not entry.get("documents_pilot_qa_passed"):
            if not os.path.exists(documents_translated):
                _console.print(f"    [yellow]SKIP documents QA[/yellow] — no translated file")
            else:
                d_pass = _run_dataset_qa(documents_translated, slug, "document", config, run_dir)
                entry["documents_pilot_qa_passed"] = d_pass
                d_score = "[green]PASS[/green]" if d_pass else "[red]FAIL[/red]"
                save_progress(run_dir, progress)
                if not d_pass:
                    entry["error"] = "pilot_qa_failed_documents"
                    save_progress(run_dir, progress)
                    _console.print(f"    [red]Pilot QA FAILED[/red] for documents")
        else:
            d_score = "[green]PASS[/green]" if entry.get("documents_pilot_qa_passed") else "skipped"

        _console.print(f"    queries QA: {q_score}   documents QA: {d_score}")

    _console.print("\n[bold]Pilot complete.[/bold] Run [cyan]submit[/cyan] to fire batch jobs.")


def _has_col(csv_path: str, col: str) -> bool:
    """Cheap check: read header only to see if column exists."""
    try:
        header = pd.read_csv(csv_path, nrows=0)
        return col in header.columns
    except Exception:
        return False


# ── Phase 2: Submit ───────────────────────────────────────────────────────────

def run_submit(
    config: dict,
    run_id: str,
    run_dir: str,
    progress: dict,
    gemini_client,
    gcs_client,
    bucket: str,
    yes: bool,
    dataset_filter: Optional[str],
) -> None:
    """
    Upload full input JSONLs to GCS and submit all batch jobs at once.
    Only datasets that passed pilot QA (or have pilot disabled) are submitted.
    """
    candidates_base = os.path.join(run_dir, "candidates")
    exec_cfg = config["execution"]
    q_cfg = config["queries"]
    d_cfg = config["documents"]
    t_cfg = config["titles"]
    pilot_enabled = config.get("progression", {}).get("pilot_n", 0) > 0
    pilot_qa = config.get("progression", {}).get("pilot_qa", False)

    dataset_names = config["datasets"]["names"]
    if dataset_filter:
        dataset_names = [n for n in dataset_names if _dataset_slug(n) == dataset_filter or n == dataset_filter]

    pending_datasets = []
    for dataset_name in dataset_names:
        slug = _dataset_slug(dataset_name)
        entry = progress["datasets"].get(slug, {})

        if pilot_enabled and pilot_qa:
            if not entry.get("queries_pilot_qa_passed") or not entry.get("documents_pilot_qa_passed"):
                _console.print(f"  [yellow]SKIP[/yellow] {dataset_name} — pilot QA not passed. Run pilot first.")
                continue

        if (entry.get("queries_batch_job_name") and
                entry.get("docs_batch_job_name")):
            _console.print(f"  [dim]SKIP[/dim] {dataset_name} — already submitted")
            continue

        pending_datasets.append(dataset_name)

    if not pending_datasets:
        _console.print("Nothing to submit.")
        return

    _console.print(f"\n[bold]Submit Phase[/bold] — {len(pending_datasets)} datasets\n")

    submitted = 0
    total_cost = 0.0

    for dataset_name in pending_datasets:
        slug = _dataset_slug(dataset_name)
        entry = progress["datasets"].setdefault(slug, _empty_dataset_entry(dataset_name))
        _patch_gcs_keys(progress)
        dataset_run_dir = os.path.join(run_dir, "corpus", slug)
        os.makedirs(dataset_run_dir, exist_ok=True)

        _console.print(f"  [cyan]{dataset_name}[/cyan]")

        queries_csv = os.path.join(candidates_base, slug, "queries.csv")
        documents_csv = os.path.join(candidates_base, slug, "documents.csv")
        queries_translated = os.path.join(dataset_run_dir, "queries_translated.csv")
        documents_translated = os.path.join(dataset_run_dir, "documents_translated.csv")

        # ── Queries ────────────────────────────────────────────────────────────
        if not entry.get("queries_batch_job_name"):
            src = queries_translated if os.path.exists(queries_translated) else queries_csv
            pending_q = load_data(src, 0, False, ignore_populated_column="translation")
            if pending_q is None or pending_q.empty:
                _console.print("    [dim]queries already fully translated[/dim]")
                entry["queries_translated"] = True
            else:
                id_cols = _id_columns_for(pending_q)
                _estimate_and_confirm(src, q_cfg["prompt"].get("text_col", "text"), config, yes)
                try:
                    input_uri = build_and_upload_input(
                        df=pending_q,
                        id_columns=id_cols,
                        gcs_client=gcs_client,
                        bucket=bucket,
                        gcs_prefix=_gcs_prefix(run_id, slug, "queries"),
                        **_upload_kwargs(q_cfg),
                    )
                    output_uri = _gcs_output_uri(bucket, run_id, slug, "queries")
                    job_name = submit_gcs_batch_job(
                        gemini_client, q_cfg["model"], input_uri, output_uri,
                        display_name=f"{run_id}__{slug}__queries",
                    )
                    entry["queries_gcs_input_uri"] = input_uri
                    entry["queries_gcs_output_prefix"] = output_uri
                    entry["queries_batch_job_name"] = job_name
                    entry["queries_submitted_at"] = _now_iso()
                    save_progress(run_dir, progress)
                    submitted += 1
                except Exception as e:
                    entry["error"] = f"submit_queries: {e}"
                    save_progress(run_dir, progress)
                    _console.print(f"    [red]ERROR[/red] submitting queries: {e}")
                    continue

        # ── Documents ──────────────────────────────────────────────────────────
        if not entry.get("docs_batch_job_name"):
            src = documents_translated if os.path.exists(documents_translated) else documents_csv
            pending_d = load_data(src, 0, False, ignore_populated_column="translation")
            if pending_d is None or pending_d.empty:
                _console.print("    [dim]documents already fully translated[/dim]")
                entry["documents_translated"] = True
            else:
                id_cols = _id_columns_for(pending_d)
                _estimate_and_confirm(src, d_cfg["prompt"].get("text_col", "segment_text"), config, yes)
                try:
                    input_uri = build_and_upload_input(
                        df=pending_d,
                        id_columns=id_cols,
                        gcs_client=gcs_client,
                        bucket=bucket,
                        gcs_prefix=_gcs_prefix(run_id, slug, "documents"),
                        **_upload_kwargs(d_cfg),
                    )
                    output_uri = _gcs_output_uri(bucket, run_id, slug, "documents")
                    job_name = submit_gcs_batch_job(
                        gemini_client, d_cfg["model"], input_uri, output_uri,
                        display_name=f"{run_id}__{slug}__documents",
                    )
                    entry["docs_gcs_input_uri"] = input_uri
                    entry["docs_gcs_output_prefix"] = output_uri
                    entry["docs_batch_job_name"] = job_name
                    entry["docs_submitted_at"] = _now_iso()
                    save_progress(run_dir, progress)
                    submitted += 1
                except Exception as e:
                    entry["error"] = f"submit_documents: {e}"
                    save_progress(run_dir, progress)
                    _console.print(f"    [red]ERROR[/red] submitting documents: {e}")
                    continue

        # ── Titles ─────────────────────────────────────────────────────────────
        if not entry.get("titles_batch_job_name") and config["datasets"].get("translate_titles", True):
            titles_csv = os.path.join(dataset_run_dir, "titles.csv")
            titles_translated = os.path.join(dataset_run_dir, "titles_translated.csv")

            if not os.path.exists(titles_csv) or exec_cfg.get("force_translation"):
                titles_df = _build_titles_df(documents_csv)
                if not titles_df.empty:
                    os.makedirs(dataset_run_dir, exist_ok=True)
                    titles_df.to_csv(titles_csv, index=False, encoding="utf-8")

            if os.path.exists(titles_csv):
                src = titles_translated if os.path.exists(titles_translated) else titles_csv
                pending_t = load_data(src, 0, False, ignore_populated_column="translation")
                if pending_t is not None and not pending_t.empty:
                    id_cols = ["_id"]
                    try:
                        input_uri = build_and_upload_input(
                            df=pending_t,
                            id_columns=id_cols,
                            gcs_client=gcs_client,
                            bucket=bucket,
                            gcs_prefix=_gcs_prefix(run_id, slug, "titles"),
                            **_upload_kwargs(t_cfg),
                        )
                        output_uri = _gcs_output_uri(bucket, run_id, slug, "titles")
                        job_name = submit_gcs_batch_job(
                            gemini_client, t_cfg["model"], input_uri, output_uri,
                            display_name=f"{run_id}__{slug}__titles",
                        )
                        entry["titles_gcs_input_uri"] = input_uri
                        entry["titles_gcs_output_prefix"] = output_uri
                        entry["titles_batch_job_name"] = job_name
                        entry["titles_submitted_at"] = _now_iso()
                        save_progress(run_dir, progress)
                        submitted += 1
                    except Exception as e:
                        entry["error"] = f"submit_titles: {e}"
                        save_progress(run_dir, progress)
                        _console.print(f"    [red]ERROR[/red] submitting titles: {e}")

    _console.print(f"\n[bold]Submit complete.[/bold] {submitted} jobs submitted.")
    _console.print("Run [cyan]collect[/cyan] periodically to check status and retrieve results.")


def _build_titles_df(documents_csv: str) -> pd.DataFrame:
    """Build deduplicated titles dataframe from documents CSV (memory-efficient)."""
    try:
        docs = pd.read_csv(documents_csv, usecols=["_id", "title"], encoding="utf-8")
        titles = (
            docs.drop_duplicates(subset=["_id"])
            .query("title.notna() and title.str.strip() != ''", engine="python")
            .rename(columns={"title": "segment_text"})
            .reset_index(drop=True)
        )
        return titles
    except Exception:
        return pd.DataFrame()


# ── Phase 3: Collect ──────────────────────────────────────────────────────────

def run_collect(
    config: dict,
    run_id: str,
    run_dir: str,
    progress: dict,
    cache: TranslationCache,
    gemini_client,
    gcs_client,
    bucket: str,
    dataset_filter: Optional[str],
    wait: bool,
    poll_interval: int,
) -> None:
    """
    Check job status, download completed results, run QA, export to BeIR JSONL.
    Prints a rich status table. Re-run until all jobs are terminal.
    """
    dataset_names = config["datasets"]["names"]
    if dataset_filter:
        dataset_names = [n for n in dataset_names if _dataset_slug(n) == dataset_filter or n == dataset_filter]

    candidates_base = os.path.join(run_dir, "candidates")
    q_cfg = config["queries"]
    d_cfg = config["documents"]

    while True:
        statuses = {}

        for dataset_name in dataset_names:
            slug = _dataset_slug(dataset_name)
            entry = progress["datasets"].get(slug, {})
            _patch_gcs_keys(progress)
            dataset_run_dir = os.path.join(run_dir, "corpus", slug)
            os.makedirs(dataset_run_dir, exist_ok=True)

            queries_csv = os.path.join(candidates_base, slug, "queries.csv")
            documents_csv = os.path.join(candidates_base, slug, "documents.csv")
            queries_translated = os.path.join(dataset_run_dir, "queries_translated.csv")
            documents_translated = os.path.join(dataset_run_dir, "documents_translated.csv")
            titles_csv = os.path.join(dataset_run_dir, "titles.csv")
            titles_translated = os.path.join(dataset_run_dir, "titles_translated.csv")

            ds_statuses = {}

            # ── Check and collect each text type ──────────────────────────────
            for text_type, job_key, gcs_output_key, translated_key, source_csv, output_csv, text_col, type_cfg in [
                ("queries",   "queries_batch_job_name", "queries_gcs_output_prefix",
                 "queries_translated",   queries_csv,   queries_translated,
                 q_cfg["prompt"].get("text_col", "text"), q_cfg),
                ("documents", "docs_batch_job_name",    "docs_gcs_output_prefix",
                 "documents_translated", documents_csv, documents_translated,
                 d_cfg["prompt"].get("text_col", "segment_text"), d_cfg),
                ("titles",    "titles_batch_job_name",  "titles_gcs_output_prefix",
                 "titles_translated",    titles_csv,    titles_translated,
                 config["titles"]["prompt"].get("text_col", "segment_text"), config["titles"]),
            ]:
                job_name = entry.get(job_key)
                if not job_name:
                    ds_statuses[text_type] = None  # not submitted
                    continue

                if entry.get(translated_key):
                    ds_statuses[text_type] = "JOB_STATE_SUCCEEDED"
                    continue

                try:
                    status = check_job_status(gemini_client, job_name)
                except Exception as e:
                    status = f"ERROR: {e}"

                ds_statuses[text_type] = status

                if status == "JOB_STATE_SUCCEEDED" and not entry.get(translated_key):
                    _console.print(f"  [green]✓[/green] {dataset_name}/{text_type} — downloading results...")
                    try:
                        _, gcs_output_path = _strip_gs_uri(entry[gcs_output_key])
                        results = download_and_parse_results(gcs_client, bucket, gcs_output_path)
                        write_translated_csv(results, source_csv, output_csv)

                        # Cache and dedup expansion
                        p = type_cfg["prompt"]
                        ctx_col = p.get("context_col") if text_type == "queries" else None
                        _expand_dedup_and_update_cache(
                            translated_csv=output_csv,
                            cache=cache,
                            model_name=type_cfg["model"],
                            prompt_file=p["file"],
                            text_col=text_col,
                            context_col=ctx_col if ctx_col and _has_col(source_csv, ctx_col) else None,
                        )

                        entry[translated_key] = True
                        entry[job_key.replace("_job_name", "_completed_at")] = _now_iso()
                        save_progress(run_dir, progress)

                    except Exception as e:
                        entry["error"] = f"collect_{text_type}: {e}"
                        save_progress(run_dir, progress)
                        _console.print(f"  [red]ERROR[/red] collecting {text_type} for {dataset_name}: {e}")

                elif status in FAILED_STATES:
                    entry["error"] = f"{text_type}_job_failed: {status}"
                    save_progress(run_dir, progress)

            statuses[slug] = ds_statuses

            # ── Merge titles + QA + export when both translated ────────────────
            q_done = entry.get("queries_translated")
            d_done = entry.get("documents_translated")
            t_done = entry.get("titles_translated") or not entry.get("titles_batch_job_name")

            if q_done and d_done and not entry.get("exported_to_beir"):
                if t_done and os.path.exists(titles_translated):
                    _merge_title_translations(documents_translated, titles_translated)

                if config.get("qa", {}).get("enabled"):
                    q_pass = _run_dataset_qa(queries_translated, slug, "query", config, run_dir)
                    d_pass = _run_dataset_qa(documents_translated, slug, "document", config, run_dir)
                    entry["queries_qa_passed"] = q_pass
                    entry["documents_qa_passed"] = d_pass
                    save_progress(run_dir, progress)
                    if not (q_pass and d_pass):
                        entry["error"] = "post_qa_failed"
                        save_progress(run_dir, progress)
                        _console.print(f"  [red]QA FAILED[/red] for {dataset_name}")
                        continue

                try:
                    run_metadata = {
                        "run_id": run_id,
                        "queries_model": q_cfg["model"],
                        "documents_model": d_cfg["model"],
                        "queries_prompt_file": q_cfg["prompt"]["file"],
                        "documents_prompt_file": d_cfg["prompt"]["file"],
                        "queries_temperature": q_cfg.get("temperature", 0.0),
                        "documents_temperature": d_cfg.get("temperature", 0.0),
                    }
                    export_to_beir_jsonl(
                        translated_queries_csv=queries_translated,
                        translated_documents_csv=documents_translated,
                        dataset_name=dataset_name,
                        output_dir=dataset_run_dir,
                        run_metadata=run_metadata,
                        segment_separator=config.get("export", {}).get("segment_separator", " "),
                        force=config["execution"].get("force_export", False),
                    )
                    entry["exported_to_beir"] = True
                    save_progress(run_dir, progress)
                    _console.print(f"  [green]✓[/green] {dataset_name} exported to BeIR JSONL")
                except Exception as e:
                    entry["error"] = f"export: {e}"
                    save_progress(run_dir, progress)
                    _console.print(f"  [red]ERROR[/red] exporting {dataset_name}: {e}")

        # ── Print status report ────────────────────────────────────────────────
        _print_report(progress, statuses, dataset_names, run_id)

        # ── Check if done ──────────────────────────────────────────────────────
        all_terminal = _all_terminal(progress, dataset_names, statuses)
        if not wait or all_terminal:
            break

        _console.print(f"\n[dim]Sleeping {poll_interval // 3600}h "
                       f"{(poll_interval % 3600) // 60}m before next poll...[/dim]")
        time.sleep(poll_interval)


def _merge_title_translations(documents_translated: str, titles_translated: str) -> None:
    """Merge title_translation column from titles into documents_translated CSV."""
    if not os.path.exists(documents_translated) or not os.path.exists(titles_translated):
        return
    try:
        docs = pd.read_csv(documents_translated, encoding="utf-8")
        titles = pd.read_csv(titles_translated, encoding="utf-8")[["_id", "translation"]]
        titles = titles.rename(columns={"translation": "title_translation"})
        if "title_translation" in docs.columns:
            docs = docs.drop(columns=["title_translation"])
        docs = docs.merge(titles, on="_id", how="left")
        docs.to_csv(documents_translated, index=False, encoding="utf-8")
    except Exception as e:
        print(f"  Warning: could not merge title translations: {e}")


def _has_col(csv_path: str, col: str) -> bool:
    try:
        return col in pd.read_csv(csv_path, nrows=0).columns
    except Exception:
        return False


def _all_terminal(progress: dict, dataset_names: list, statuses: dict) -> bool:
    for name in dataset_names:
        slug = _dataset_slug(name)
        entry = progress["datasets"].get(slug, {})
        ds_st = statuses.get(slug, {})
        for status in ds_st.values():
            if status is not None and status not in TERMINAL_STATES and not status.startswith("ERROR"):
                return False
        if not entry.get("exported_to_beir") and not entry.get("error"):
            return False
    return True


# ── Status report ─────────────────────────────────────────────────────────────

def _fmt_job_status(
    status: Optional[str],
    submitted_at: Optional[str],
    completed_at: Optional[str],
    translated: bool,
) -> str:
    if translated or status == "JOB_STATE_SUCCEEDED":
        elapsed = _elapsed_str(submitted_at, completed_at)
        return f"[green]SUCCEEDED[/green] ({elapsed})" if elapsed else "[green]SUCCEEDED[/green]"
    if status is None:
        return "[dim]—[/dim]"
    if status in FAILED_STATES or (status and status.startswith("ERROR")):
        return f"[red]{status}[/red]"
    if status:
        elapsed = _elapsed_str(submitted_at)
        short = status.replace("JOB_STATE_", "")
        return f"[yellow]{short}[/yellow] ({elapsed})" if elapsed else f"[yellow]{short}[/yellow]"
    return "[dim]—[/dim]"


def _print_report(progress: dict, statuses: dict, dataset_names: list, run_id: str) -> None:
    table = Table(title=f"Batch Status — {run_id}", show_lines=True)
    table.add_column("Dataset", style="cyan", min_width=22)
    table.add_column("Queries",   min_width=22)
    table.add_column("Documents", min_width=22)
    table.add_column("Titles",    min_width=14)
    table.add_column("QA",        min_width=6)
    table.add_column("Export",    min_width=6)

    complete = failed = running = 0

    for name in dataset_names:
        slug = _dataset_slug(name)
        entry = progress["datasets"].get(slug, {})
        ds_st = statuses.get(slug, {})

        q_cell = _fmt_job_status(
            ds_st.get("queries"), entry.get("queries_submitted_at"),
            entry.get("queries_completed_at"), entry.get("queries_translated", False),
        )
        d_cell = _fmt_job_status(
            ds_st.get("documents"), entry.get("docs_submitted_at"),
            entry.get("docs_completed_at"), entry.get("documents_translated", False),
        )
        t_cell = _fmt_job_status(
            ds_st.get("titles"), entry.get("titles_submitted_at"),
            entry.get("titles_completed_at"), entry.get("titles_translated", False),
        )

        qa_q = entry.get("queries_qa_passed")
        qa_d = entry.get("documents_qa_passed")
        if qa_q is None and qa_d is None:
            qa_cell = "[dim]—[/dim]"
        elif qa_q and qa_d:
            qa_cell = "[green]PASS[/green]"
        elif qa_q is False or qa_d is False:
            qa_cell = "[red]FAIL[/red]"
        else:
            qa_cell = "[dim]—[/dim]"

        export_cell = "[green]done[/green]" if entry.get("exported_to_beir") else "[dim]—[/dim]"

        table.add_row(name, q_cell, d_cell, t_cell, qa_cell, export_cell)

        if entry.get("exported_to_beir"):
            complete += 1
        elif entry.get("error") and "failed" in entry.get("error", ""):
            failed += 1
        else:
            running += 1

    _console.print(table)
    _console.print(
        f"[bold]{complete}[/bold] complete · "
        f"[yellow]{running}[/yellow] in progress · "
        f"[red]{failed}[/red] failed\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BeIR GCS Batch Translation Pipeline — pilot / submit / collect"
    )
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    sub = parser.add_subparsers(dest="phase", required=True)

    p_pilot = sub.add_parser("pilot", help="Translate pilot_n rows synchronously and run QA gates.")
    p_pilot.add_argument("--yes", "-y", action="store_true", help="Skip cost confirmation.")
    p_pilot.add_argument("--dataset", help="Run pilot for a single dataset slug only.")
    p_pilot.add_argument("--qa", action="store_true",
                         help="Skip translation, only re-run QA on existing pilot translations.")

    p_submit = sub.add_parser("submit", help="Upload JSONL to GCS and submit all batch jobs.")
    p_submit.add_argument("--yes", "-y", action="store_true", help="Skip cost confirmation.")
    p_submit.add_argument("--dataset", help="Submit a single dataset slug only.")

    p_collect = sub.add_parser("collect", help="Check status, download results, export.")
    p_collect.add_argument("--dataset", help="Collect a single dataset slug only.")
    p_collect.add_argument("--wait", action="store_true",
                           help="Loop until all jobs are terminal.")
    p_collect.add_argument("--poll-interval", type=int, default=3600,
                           help="Seconds between polls when --wait is set (default 3600).")

    args = parser.parse_args()

    config = load_config(args.config)
    run_id = make_run_id(config)
    config["run_id"] = run_id

    runs_base = config["paths"]["runs_base"]
    run_dir = os.path.join(runs_base, run_id)

    progress = load_or_init_progress(run_dir, config, run_id)
    _patch_gcs_keys(progress)

    # Cache key built from queries model + prompt file (representative for the run)
    q_model_slug = _slugify(config["queries"]["model"])
    q_prompt_slug = _slugify(Path(config["queries"]["prompt"]["file"]).stem)
    cache_path = os.path.join(runs_base, "cache", f"{q_model_slug}__{q_prompt_slug}.jsonl")
    cache = TranslationCache(cache_path)

    project, bucket, location = _get_gcs_config(config)

    # Propagate to env so child processes (parallel translation workers) can read them
    os.environ["GEMINI_PROJECT"] = project
    os.environ["GEMINI_LOCATION"] = location

    _console.print(f"\n[bold]BeIR GCS Batch Pipeline[/bold]  phase=[cyan]{args.phase}[/cyan]")
    _console.print(f"  Run ID:          {run_id}")
    _console.print(f"  Queries model:   {config['queries']['model']}")
    _console.print(f"  Documents model: {config['documents']['model']}")
    _console.print(f"  Project:         {project}")
    _console.print(f"  Location:        {location}")
    _console.print(f"  Output:          {run_dir}")

    if args.phase == "pilot":
        run_pilot(config, run_id, run_dir, progress, cache,
                  yes=args.yes, dataset_filter=args.dataset,
                  qa_only=getattr(args, "qa", False))

    elif args.phase == "submit":
        _validate_gcs_auth()
        gemini_client = _make_gemini_client(project, location)
        gcs_client = get_gcs_client(project)
        _console.print(f"  Bucket:   gs://{bucket}\n")
        run_submit(config, run_id, run_dir, progress, gemini_client, gcs_client, bucket,
                   yes=args.yes, dataset_filter=args.dataset)

    elif args.phase == "collect":
        _validate_gcs_auth()
        gemini_client = _make_gemini_client(project, location)
        gcs_client = get_gcs_client(project)
        _console.print(f"  Bucket:   gs://{bucket}\n")
        run_collect(config, run_id, run_dir, progress, cache, gemini_client, gcs_client, bucket,
                    dataset_filter=args.dataset, wait=args.wait,
                    poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()

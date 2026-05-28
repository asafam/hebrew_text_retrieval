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
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from translation.api.run_beir_translation_pipeline import (
    load_config,
    _dataset_slug,
    save_progress,
)
from translation.api.run_beir_batch_gcs import _build_run_kwargs
from translation.api.translate import run_translation_pipeline as run_parallel_pipeline
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


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_shard(
    shard_csv: str,
    output_dir: str,
    type_cfg: dict,
    exec_cfg: dict,
) -> Optional[str]:
    """Translate one shard CSV. Returns path to the translated output file."""
    run_kwargs = _build_run_kwargs(type_cfg, exec_cfg, output_dir, force=True)
    run_parallel_pipeline(source_file_path=shard_csv, **run_kwargs)
    out_path = (
        os.path.join(output_dir, os.path.basename(shard_csv))
        .replace(".csv", "_translated.csv")
    )
    return out_path if os.path.exists(out_path) else None


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

def run_ladder(config: dict, run_dir: str, progress: dict, dataset_filter: Optional[str]) -> None:
    run_id = progress["run_id"]
    candidates_base = _ladder_candidates_base(config)
    exec_cfg = config.get("execution", {})
    q_cfg = config["queries"]
    d_cfg = config["documents"]

    datasets = config["datasets"]["names"]
    if dataset_filter:
        datasets = [d for d in datasets if _dataset_slug(d) == dataset_filter or d == dataset_filter]

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

            # ── Translate queries shard ───────────────────────────────────────
            q_shard_csv = os.path.join(candidates_base, slug, shard_meta["file"])
            if not os.path.exists(q_shard_csv):
                logger.error(f"[{slug}] Query shard file missing: {q_shard_csv}")
                entry["ladder_stopped"] = True
                entry["ladder_stop_reason"] = f"shard {idx}: query shard file missing"
                save_progress(run_dir, progress)
                dataset_stopped = True
                break

            logger.info(f"[{slug}] Translating queries shard {idx}: {q_shard_csv}")
            q_out = _translate_shard(q_shard_csv, shard_out_dir, q_cfg, exec_cfg)
            if q_out is None:
                logger.error(f"[{slug}] Query shard {idx} translation produced no output")
                entry["ladder_stopped"] = True
                entry["ladder_stop_reason"] = f"shard {idx}: query translation produced no output"
                save_progress(run_dir, progress)
                dataset_stopped = True
                break

            # ── Translate documents shard ─────────────────────────────────────
            d_out = None
            if d_shard_meta:
                d_shard_csv = os.path.join(candidates_base, slug, d_shard_meta["file"])
                if not os.path.exists(d_shard_csv):
                    logger.error(f"[{slug}] Document shard file missing: {d_shard_csv}")
                    entry["ladder_stopped"] = True
                    entry["ladder_stop_reason"] = f"shard {idx}: document shard file missing"
                    save_progress(run_dir, progress)
                    dataset_stopped = True
                    break
                logger.info(f"[{slug}] Translating documents shard {idx}: {d_shard_csv}")
                d_out = _translate_shard(d_shard_csv, shard_out_dir, d_cfg, exec_cfg)
                if d_out is None:
                    logger.error(f"[{slug}] Document shard {idx} translation produced no output")
                    entry["ladder_stopped"] = True
                    entry["ladder_stop_reason"] = f"shard {idx}: document translation produced no output"
                    save_progress(run_dir, progress)
                    dataset_stopped = True
                    break

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
            )
            entry["ladder_stage_scores"][str(idx)] = {
                "q_score_mean": q_result["score_mean"],
                "q_score_std":  q_result["score_std"],
                "d_score_mean": d_result["score_mean"],
                "d_score_std":  d_result["score_std"],
                "passed": q_result["passed"] and d_result["passed"],
                "cumulative_q_rows": cumulative_q,
                "cumulative_d_rows": cumulative_d,
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

    progress = _load_or_init_progress(run_dir, config, run_id)
    run_ladder(config, run_dir, progress, args.dataset)

    logger.info("Ladder pipeline complete.")


if __name__ == "__main__":
    main()

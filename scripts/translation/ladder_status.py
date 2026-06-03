#!/usr/bin/env python3
"""
Show the cumulative ladder status for a run — every cadence step with its
shards, queries/documents judge scores, and cumulative cost. Read-only; safe
to run against a live run, including parallel runs with per-dataset files.

Usage:
    # One-shot snapshot
    python scripts/translation/ladder_status.py --run-dir outputs/translation/runs/<run_id>
    python scripts/translation/ladder_status.py --run-dir <RUN_DIR> --dataset BeIR/nfcorpus

    # Live: refresh every N seconds (default 30) until all datasets are done
    python scripts/translation/ladder_status.py --run-dir <RUN_DIR> --live
    python scripts/translation/ladder_status.py --run-dir <RUN_DIR> --live --interval 10

Parallel runs: when each dataset has its own progress.<slug>.json (produced by
--parallel or --dataset), this script finds and merges all of them automatically.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from translation.api.run_beir_ladder_pipeline import _build_ladder_table
from translation.api.run_beir_translation_pipeline import load_config, _dataset_slug


def _load_progress_all(run_dir: str) -> dict:
    """Merge progress from all progress.*.json files + shared progress.json."""
    merged = {"datasets": {}}
    run_path = Path(run_dir)

    # Shared (serial) file first
    shared = run_path / "progress.json"
    if shared.exists():
        data = json.loads(shared.read_text())
        merged["run_id"] = data.get("run_id", "")
        merged["total_cost_usd"] = data.get("total_cost_usd", 0.0)
        merged["datasets"].update(data.get("datasets", {}))

    # Per-dataset files (parallel mode) — override shared entries
    total_cost = merged.get("total_cost_usd", 0.0)
    for p in sorted(run_path.glob("progress.BeIR_*.json")):
        data = json.loads(p.read_text())
        if not merged.get("run_id"):
            merged["run_id"] = data.get("run_id", "")
        for slug, entry in data.get("datasets", {}).items():
            merged["datasets"][slug] = entry
        total_cost += data.get("total_cost_usd", 0.0)

    if list(run_path.glob("progress.BeIR_*.json")):
        merged["total_cost_usd"] = total_cost

    return merged


def _summary_table(run_dir: str, config: dict, progress: dict,
                   dataset_filter: str = None) -> Table:
    """One-line-per-dataset summary table."""
    cand_base = os.path.join(run_dir, "candidates")
    datasets_cfg = config["datasets"]["names"]
    slugs = [_dataset_slug(n) for n in datasets_cfg]
    if dataset_filter:
        want = _dataset_slug(dataset_filter)
        slugs = [s for s in slugs if s == want or s == dataset_filter]

    total_cost = progress.get("total_cost_usd", 0.0)
    t = Table(
        title=f"Ladder status — {progress.get('run_id','')}  ·  total ${total_cost:.2f}",
        show_header=True, header_style="bold", expand=False,
    )
    t.add_column("Dataset",     min_width=24)
    t.add_column("Steps",       justify="right")
    t.add_column("Queries QA",  justify="right", min_width=14)
    t.add_column("Docs QA",     justify="right", min_width=14)
    t.add_column("Status",      min_width=14)
    t.add_column("Cost",        justify="right")

    for slug in slugs:
        entry = progress["datasets"].get(slug, {})
        ss = entry.get("ladder_stage_scores", {})
        last = ss[max(ss, key=int)] if ss else {}
        q = f"{last['q_score_mean']:.2f}±{last.get('q_score_std',0):.2f}" if last else "—"
        d = f"{last['d_score_mean']:.2f}±{last.get('d_score_std',0):.2f}" if last else "—"
        cost = f"${entry.get('shards',{}) and sum(v.get(t2,{}).get('cumulative_cost_usd',0) for v in entry.get('shards',{}).values() for t2 in v):.2f}" if entry.get("shards") else "—"
        cum_cost = last.get("cumulative_cost_usd")
        cost_str = f"${cum_cost:.2f}" if cum_cost is not None else "—"

        if entry.get("ladder_all_done"):
            status = Text("✓ done", style="green")
        elif entry.get("ladder_stopped"):
            status = Text("✗ stopped", style="red")
        elif ss:
            status = Text(f"⟳ step {entry.get('ladder_cadence_step',0)}", style="yellow")
        else:
            status = Text("· pending", style="dim")

        t.add_row(slug.replace("BeIR_",""), str(len(ss)), q, d, status, cost_str)

    return t


def main():
    ap = argparse.ArgumentParser(description="Ladder status — supports parallel per-dataset runs.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="config/translation/full_corpus.yaml")
    ap.add_argument("--dataset", default=None, help="Single dataset filter.")
    ap.add_argument("--detail", action="store_true",
                    help="Show per-shard detail table for each dataset.")
    ap.add_argument("--live", action="store_true",
                    help="Refresh display every --interval seconds until all done.")
    ap.add_argument("--interval", type=int, default=30,
                    help="Refresh interval in seconds for --live (default: 30).")
    args = ap.parse_args()

    config = load_config(args.config)
    console = Console()

    def _render():
        progress = _load_progress_all(args.run_dir)
        console.print(_summary_table(args.run_dir, config, progress, args.dataset))
        if args.detail:
            cand_base = os.path.join(args.run_dir, "candidates")
            slugs = [_dataset_slug(n) for n in config["datasets"]["names"]]
            if args.dataset:
                want = _dataset_slug(args.dataset)
                slugs = [s for s in slugs if s == want]
            for slug in slugs:
                entry = progress["datasets"].get(slug, {})
                manifest_path = os.path.join(cand_base, slug, "shard_manifest.json")
                if not os.path.exists(manifest_path):
                    continue
                m = json.load(open(manifest_path))
                q_by_idx = {s["index"]: s for s in m["types"]["queries"]}
                d_by_idx = {s["index"]: s for s in m["types"]["documents"]}
                console.print(_build_ladder_table(slug, config, entry, q_by_idx, d_by_idx))
        return progress

    if not args.live:
        _render()
        return

    # Live mode: refresh until all datasets are done or stopped
    console.print(f"[dim]Live mode — refreshing every {args.interval}s. Ctrl-C to exit.[/dim]")
    while True:
        console.clear()
        progress = _render()
        ds = progress.get("datasets", {})
        relevant = [v for k, v in ds.items()
                    if not args.dataset or _dataset_slug(args.dataset) == k]
        all_terminal = all(v.get("ladder_all_done") or v.get("ladder_stopped")
                          for v in relevant if v.get("ladder_stage_scores") or v.get("ladder_all_done"))
        if all_terminal and relevant:
            console.print("[green]All datasets reached terminal state.[/green]")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

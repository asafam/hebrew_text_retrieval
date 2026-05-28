#!/usr/bin/env python3
"""
Build sharded translation candidates for all BeIR datasets in parallel.

Reads all configuration from a YAML file. Each dataset runs as an
independent subprocess so failures are isolated and a live progress
table shows the status of every dataset at a glance.

Usage:
    python scripts/translation/build_ladder_candidates.py
    python scripts/translation/build_ladder_candidates.py --config config/translation/candidates.yaml
    python scripts/translation/build_ladder_candidates.py --dataset nfcorpus   # partial name match
    python scripts/translation/build_ladder_candidates.py --force
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CONFIG = "config/translation/candidates.yaml"
console = Console()


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def shard_size_for(slug: str, cfg: dict) -> int:
    ds = cfg.get("datasets", {})
    return ds.get("shard_sizes", {}).get(slug, ds.get("default_shard_size", 10000))


# ── Manifest reader ─────────────────────────────────────────────────────────────

def _read_manifest(output_path: str, slug: str) -> dict | None:
    p = os.path.join(output_path, slug, "shard_manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ── Progress table ─────────────────────────────────────────────────────────────

STATUS_STYLE = {
    "pending":  "dim",
    "running":  "bold cyan",
    "done":     "bold green",
    "skipped":  "dim green",
    "failed":   "bold red",
}

STATUS_ICON = {
    "pending":  "·",
    "running":  "⟳",
    "done":     "✓",
    "skipped":  "✓",
    "failed":   "✗",
}


def _build_table(jobs: list, output_path: str, elapsed: float) -> Table:
    table = Table(
        title=f"Building ladder candidates  [dim]({elapsed:.0f}s)[/dim]",
        show_header=True,
        header_style="bold",
        show_lines=False,
        expand=False,
    )
    table.add_column("Dataset",      min_width=26)
    table.add_column("Status",       min_width=10)
    table.add_column("Shard size",   justify="right", min_width=10)
    table.add_column("Q shards",     justify="right", min_width=9)
    table.add_column("D shards",     justify="right", min_width=9)
    table.add_column("Queries",      justify="right", min_width=10)
    table.add_column("Documents",    justify="right", min_width=11)
    table.add_column("Time",         justify="right", min_width=7)

    for j in jobs:
        status = j["status"]
        style  = STATUS_STYLE.get(status, "")
        icon   = STATUS_ICON.get(status, "?")
        label  = f"{icon} {status}"

        q_shards = d_shards = queries = documents = shard_sz = "-"

        if status in ("done", "skipped"):
            m = _read_manifest(output_path, j["slug"])
            if m:
                qs = m["types"]["queries"]
                ds = m["types"]["documents"]
                q_shards  = str(len(qs))
                d_shards  = str(len(ds))
                queries   = f"{sum(s['rows'] for s in qs):,}"
                documents = f"{sum(s['rows'] for s in ds):,}"
                shard_sz  = f"{m['shard_size']:,}"

        elapsed_job = "-"
        if j.get("started_at") and j.get("ended_at"):
            elapsed_job = f"{j['ended_at'] - j['started_at']:.0f}s"
        elif j.get("started_at"):
            elapsed_job = f"{time.time() - j['started_at']:.0f}s"

        table.add_row(
            j["dataset_name"], label, shard_sz,
            q_shards, d_shards, queries, documents, elapsed_job,
            style=style,
        )

    return table


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build sharded candidates for all BeIR datasets in parallel.")
    parser.add_argument("--config",  default=DEFAULT_CONFIG, help="YAML candidates config.")
    parser.add_argument("--dataset", default=None,           help="Partial name filter (e.g. 'nfcorpus').")
    parser.add_argument("--force",   action="store_true",    help="Rebuild even if manifest already exists.")
    parser.add_argument("--split",   default="all",
                        help="Qrel split to include (test/train/validation/dev/all). Default: all.")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    ds_cfg      = cfg.get("datasets", {})
    paths_cfg   = cfg.get("paths",    {})
    output_path = paths_cfg.get("output", "outputs/translation/candidates")
    log_dir     = paths_cfg.get("logs",   os.path.join(output_path, "logs"))
    os.makedirs(log_dir, exist_ok=True)

    all_datasets = ds_cfg.get("names", [])
    if not all_datasets:
        console.print("[red]No datasets found in config.[/red]")
        sys.exit(1)

    if args.dataset:
        all_datasets = [d for d in all_datasets if args.dataset.lower() in d.lower()]
    if not all_datasets:
        console.print(f"[red]No datasets matched filter '{args.dataset}'.[/red]")
        sys.exit(1)

    python = sys.executable

    # ── Build job list ─────────────────────────────────────────────────────────
    jobs = []
    for name in all_datasets:
        slug     = name.replace("/", "_")
        manifest = os.path.join(output_path, slug, "shard_manifest.json")
        if not args.force and os.path.exists(manifest):
            status = "skipped"
        else:
            status = "pending"
        jobs.append({
            "dataset_name": name,
            "slug":         slug,
            "status":       status,
            "log":          os.path.join(log_dir, f"{slug}.log"),
            "proc":         None,
            "started_at":   None,
            "ended_at":     None,
        })

    console.print(f"\n[bold]Building ladder candidates[/bold]")
    console.print(f"  Config:  {args.config}")
    console.print(f"  Output:  {output_path}")
    console.print(f"  Logs:    {log_dir}")
    console.print(f"  Datasets: {len(jobs)} ({sum(1 for j in jobs if j['status']=='skipped')} already done)\n")

    start_time = time.time()

    # ── Launch all pending jobs ────────────────────────────────────────────────
    for j in jobs:
        if j["status"] != "pending":
            continue
        log_fh = open(j["log"], "w")
        cmd = [
            python, "-m", "translation.build_translation_candidates",
            "--config",        args.config,
            "--dataset_names", j["dataset_name"],
            "--split",         args.split,
        ]
        if args.force:
            cmd.append("--force")
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        j["proc"]       = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT)
        j["log_fh"]     = log_fh
        j["status"]     = "running"
        j["started_at"] = time.time()

    # ── Live progress loop ─────────────────────────────────────────────────────
    with Live(console=console, refresh_per_second=2) as live:
        while True:
            for j in jobs:
                if j["status"] != "running":
                    continue
                rc = j["proc"].poll()
                if rc is not None:
                    j["ended_at"] = time.time()
                    j["log_fh"].close()
                    j["status"] = "done" if rc == 0 else "failed"

            live.update(_build_table(jobs, output_path, time.time() - start_time))

            if all(j["status"] in ("done", "skipped", "failed") for j in jobs):
                break
            time.sleep(0.5)

    # ── Final summary ──────────────────────────────────────────────────────────
    failed = [j for j in jobs if j["status"] == "failed"]
    done   = [j for j in jobs if j["status"] == "done"]
    skip   = [j for j in jobs if j["status"] == "skipped"]

    console.print()
    if failed:
        console.print(f"[bold red]FAILED ({len(failed)} dataset(s)):[/bold red]")
        for j in failed:
            console.print(f"  {j['dataset_name']}  → {j['log']}")
            try:
                lines = Path(j["log"]).read_text().splitlines()
                for line in lines[-5:]:
                    console.print(f"    [dim]{line}[/dim]")
            except OSError:
                pass
        console.print()
        sys.exit(1)
    else:
        console.print(
            f"[bold green]All done.[/bold green]  "
            f"{len(done)} built, {len(skip)} already existed  "
            f"({time.time() - start_time:.0f}s total)"
        )
        console.print()
        console.print("Next step:")
        console.print(
            "  python -m translation.api.run_beir_ladder_pipeline "
            f"--config config/translation/full_corpus.yaml --dry-run"
        )
        console.print()


if __name__ == "__main__":
    main()

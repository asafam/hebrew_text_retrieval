#!/usr/bin/env python3
"""
Show the cumulative ladder status for a run — every cadence step with its
shards, queries/documents judge scores, and cumulative cost. Read-only; safe
to run against a live run.

Usage:
    python scripts/translation/ladder_status.py --run-dir outputs/translation/runs/<run_id>
    python scripts/translation/ladder_status.py --run-dir <RUN_DIR> --dataset BeIR/nfcorpus
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from translation.api.run_beir_ladder_pipeline import _build_ladder_table
from translation.api.run_beir_translation_pipeline import load_config, _dataset_slug


def main():
    ap = argparse.ArgumentParser(description="Cumulative ladder status per dataset.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="config/translation/full_corpus.yaml")
    ap.add_argument("--dataset", default=None, help="Single dataset (name or slug).")
    args = ap.parse_args()

    config = load_config(args.config)
    progress = json.load(open(os.path.join(args.run_dir, "progress.json")))
    cand_base = os.path.join(args.run_dir, "candidates")
    console = Console()

    slugs = list(progress["datasets"].keys())
    if args.dataset:
        want = _dataset_slug(args.dataset)
        slugs = [s for s in slugs if s == want or s == args.dataset]

    for slug in slugs:
        entry = progress["datasets"][slug]
        manifest_path = os.path.join(cand_base, slug, "shard_manifest.json")
        if not os.path.exists(manifest_path):
            continue
        m = json.load(open(manifest_path))
        q_by_idx = {s["index"]: s for s in m["types"]["queries"]}
        d_by_idx = {s["index"]: s for s in m["types"]["documents"]}
        console.print(_build_ladder_table(slug, config, entry, q_by_idx, d_by_idx))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate zero-shot BEIR evaluation results into a Markdown table.

Walks outputs/eval/beir_zeroshot/ (or --results_dir) for results.json files,
then prints a model × dataset table of NDCG@10 values.

Usage:
    python src/model/eval/collect_beir_results.py
    python src/model/eval/collect_beir_results.py --results_dir outputs/eval/beir_zeroshot --output_csv results.csv
"""

import os
import json
import argparse
from pathlib import Path


def collect(results_dir):
    """Return list of result dicts from all results.json files under results_dir."""
    results = []
    for path in sorted(Path(results_dir).rglob("results.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            results.append({
                "model": data.get("model", str(path.parent.parent.name)),
                "dataset": data.get("dataset", str(path.parent.name)),
                "metrics": data.get("metrics", {}),
                "path": str(path),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {path} ({e})")
    return results


def build_table(results, metric="ndcg_at_10"):
    """Build model × dataset pivot table for the given metric."""
    models = []
    datasets = []
    seen_models = set()
    seen_datasets = set()

    for r in results:
        if r["model"] not in seen_models:
            models.append(r["model"])
            seen_models.add(r["model"])
        if r["dataset"] not in seen_datasets:
            datasets.append(r["dataset"])
            seen_datasets.add(r["dataset"])

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["model"], r["dataset"])] = r["metrics"].get(metric)

    return models, datasets, lookup


def print_markdown(models, datasets, lookup, metric="ndcg_at_10"):
    col_w = max(len(d) for d in datasets) + 2
    model_w = max(len(m) for m in models) + 2

    header = f"| {'Model':<{model_w}} |" + "".join(f" {d:<{col_w}} |" for d in datasets)
    sep = f"|{'-' * (model_w + 2)}|" + "".join(f"{'-' * (col_w + 2)}|" for _ in datasets)

    print(f"\n## {metric.upper().replace('_', '@')} Results\n")
    print(header)
    print(sep)
    for model in models:
        row = f"| {model:<{model_w}} |"
        for dataset in datasets:
            val = lookup.get((model, dataset))
            cell = f"{val:.4f}" if val is not None else "—"
            row += f" {cell:<{col_w}} |"
        print(row)
    print()


def write_csv(models, datasets, lookup, output_csv, metric="ndcg_at_10"):
    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + datasets)
        for model in models:
            row = [model] + [lookup.get((model, d), "") for d in datasets]
            writer.writerow(row)
    print(f"CSV saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate BEIR zero-shot eval results.")
    parser.add_argument("--results_dir", default="outputs/eval/beir_zeroshot",
                        help="Root directory to scan for results.json files.")
    parser.add_argument("--metric", default="ndcg_at_10",
                        choices=["ndcg_at_10", "ndcg_at_100", "recall_at_100", "mrr"],
                        help="Metric to display in the table (default: ndcg_at_10).")
    parser.add_argument("--output_csv", default=None,
                        help="Optional CSV output path.")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return

    results = collect(args.results_dir)
    if not results:
        print(f"No results.json files found under {args.results_dir}")
        return

    print(f"Found {len(results)} result file(s).")
    models, datasets, lookup = build_table(results, metric=args.metric)
    print_markdown(models, datasets, lookup, metric=args.metric)

    if args.output_csv:
        write_csv(models, datasets, lookup, args.output_csv, metric=args.metric)


if __name__ == "__main__":
    main()

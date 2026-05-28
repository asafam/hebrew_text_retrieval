#!/usr/bin/env python3
"""
Export a finished ladder translation run to BeIR-format JSONL.

Reads accumulated CSVs from a ladder run directory and produces
HuggingFace-ready BeIR files per dataset:

    <out>/<slug>/beir/corpus.jsonl   {_id, title (HE), title_en, text (HE), text_en}
    <out>/<slug>/beir/queries.jsonl  {_id, text (HE), text_en}
    <out>/<slug>/beir/qrels/<split>.tsv
    <out>/<slug>/beir/metadata.json

Usage:
    python scripts/translation/build_hf_dataset.py --run-dir <RUN_DIR>
    python scripts/translation/build_hf_dataset.py --run-dir <RUN_DIR> --dataset nfcorpus
    python scripts/translation/build_hf_dataset.py --run-dir <RUN_DIR> --require-complete --force

HF upload is intentionally NOT part of this script — run `huggingface-cli upload`
or `huggingface_hub.HfApi().upload_folder(...)` against the resulting beir/ dir.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from translation.beir.export import export_to_beir_jsonl


def _find_dataset_slugs(run_dir: Path, name_filter: str | None) -> list:
    """Return slugs of datasets with both accumulated CSVs present."""
    slugs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "queries_accumulated.csv").exists():
            continue
        if not (child / "documents_accumulated.csv").exists():
            continue
        if name_filter and name_filter.lower() not in child.name.lower():
            continue
        slugs.append(child.name)
    return slugs


def _slug_to_dataset_name(slug: str) -> str:
    # "BeIR_nfcorpus" -> "BeIR/nfcorpus"  (only the first underscore)
    return slug.replace("_", "/", 1)


def _load_progress(run_dir: Path) -> dict:
    p = run_dir / "progress.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Export a ladder run's accumulated CSVs to BeIR JSONL.")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Path to a ladder run directory (under outputs/translation/runs/).")
    parser.add_argument("--dataset", default=None,
                        help="Partial slug filter, e.g. 'nfcorpus'.")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Override export root (default: write under each dataset's run subdir).")
    parser.add_argument("--segment-separator", default=" ",
                        help="Joiner between translated document segments (default: single space).")
    parser.add_argument("--require-complete", action="store_true",
                        help="Skip datasets whose ladder did not finish all shards.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing beir/ outputs.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"Run dir not found: {run_dir}")

    progress = _load_progress(run_dir)
    datasets_progress = progress.get("datasets", {})
    run_metadata = {
        "run_id":      progress.get("run_id", run_dir.name),
        "config_file": progress.get("config_file"),
        "started_at":  progress.get("started_at"),
        "run_dir":     str(run_dir),
    }

    slugs = _find_dataset_slugs(run_dir, args.dataset)
    if not slugs:
        sys.exit("No datasets with both queries_accumulated.csv and documents_accumulated.csv found.")

    print(f"Exporting {len(slugs)} dataset(s) from {run_dir}\n")
    skipped, exported = [], []
    for slug in slugs:
        dataset_name = _slug_to_dataset_name(slug)
        entry = datasets_progress.get(slug, {})
        all_done = entry.get("ladder_all_done", False)
        if args.require_complete and not all_done:
            print(f"  [skip] {dataset_name} — ladder_all_done=False")
            skipped.append(slug)
            continue

        dataset_dir = run_dir / slug
        out_dir = (args.output_dir / slug) if args.output_dir else dataset_dir
        try:
            export_to_beir_jsonl(
                translated_queries_csv=str(dataset_dir / "queries_accumulated.csv"),
                translated_documents_csv=str(dataset_dir / "documents_accumulated.csv"),
                dataset_name=dataset_name,
                output_dir=str(out_dir),
                run_metadata={**run_metadata, "ladder_all_done": all_done},
                segment_separator=args.segment_separator,
                force=args.force,
            )
            exported.append(slug)
        except Exception as e:
            print(f"  [fail] {dataset_name}: {e}")

    print()
    print(f"Exported: {len(exported)}  ·  Skipped: {len(skipped)}  ·  Total candidates: {len(slugs)}")
    if exported:
        print("\nNext step (HF upload, run yourself):")
        sample_slug = exported[0]
        sample_dir = (args.output_dir / sample_slug if args.output_dir else run_dir / sample_slug) / "beir"
        print(f"  huggingface-cli upload <your-org/{sample_slug}-he> {sample_dir} . --repo-type dataset")


if __name__ == "__main__":
    main()

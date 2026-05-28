from typing import List, Optional
import json
import pandas as pd
import argparse
import os
from tqdm import tqdm
from data.translation_candidates import build_data


DEFAULT_OUTPUT_PATH = "outputs/translation/candidates"


def _write_shards(df: pd.DataFrame, slug_dir: str, prefix: str, shard_size: int) -> list:
    """Split df into fixed-size shard CSVs; return manifest shard list."""
    shards = []
    total = len(df)
    idx = 0
    shard_idx = 0
    while idx < total:
        chunk = df.iloc[idx: idx + shard_size]
        fname = f"{prefix}_shard_{shard_idx:03d}.csv"
        fpath = os.path.join(slug_dir, fname)
        chunk.to_csv(fpath, index=False, encoding="utf-8")
        shards.append({"index": shard_idx, "file": fname, "rows": len(chunk)})
        idx += shard_size
        shard_idx += 1
    return shards


def build_dataset_candidates(
    dataset_names: List[str],
    num_samples: int,
    output_path: str,
    max_document_segment_tokens: int,
    model_name_or_path: str,
    tokenizer_name_or_path: str = None,
    split: str = "test",
    force: bool = False,
    random_state: int = 42,
    shard_size: Optional[int] = None,
) -> None:
    """Build candidate CSVs for each dataset.

    When shard_size is given, outputs fixed-size shard files and a
    shard_manifest.json instead of a single queries/documents.csv.
    Without shard_size, the original single-file behaviour is preserved.
    """
    for dataset_name in tqdm(dataset_names, desc="Processing datasets: "):
        print(f"Processing dataset: {dataset_name}")

        dataset_name_slug = dataset_name.replace("/", "_")
        slug_dir = os.path.join(output_path, dataset_name_slug)

        if shard_size:
            manifest_path = os.path.join(slug_dir, "shard_manifest.json")
            if not force and os.path.exists(manifest_path):
                print(f"Skipping {dataset_name} — shard_manifest.json exists (use --force to rebuild).")
                continue
        else:
            queries_output_path = os.path.join(slug_dir, split, "queries.csv")
            documents_output_path = os.path.join(slug_dir, split, "documents.csv")
            if not force and os.path.exists(queries_output_path) and os.path.exists(documents_output_path):
                print(f"Skipping {dataset_name} — files exist (use --force to rebuild).")
                continue

        data = build_data(
            dataset_name=dataset_name,
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            n=num_samples,
            max_tokens=max_document_segment_tokens,
            split=split,
            random_state=random_state,
        )
        queries, documents = data

        queries_df = pd.DataFrame(queries)
        documents_df = pd.DataFrame(documents)
        for df in [queries_df, documents_df]:
            df["dataset_name"] = dataset_name
            df["tokenizer"] = tokenizer_name_or_path or model_name_or_path

        os.makedirs(slug_dir, exist_ok=True)

        if shard_size:
            q_shards = _write_shards(queries_df, slug_dir, "queries", shard_size)
            d_shards = _write_shards(documents_df, slug_dir, "documents", shard_size)
            manifest = {
                "shard_size": shard_size,
                "types": {"queries": q_shards, "documents": d_shards},
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Saved {len(queries_df)} queries → {len(q_shards)} shards")
            print(f"Saved {len(documents_df)} documents → {len(d_shards)} shards")
            print(f"Manifest: {manifest_path}")
        else:
            split_dir = os.path.join(slug_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            qp = os.path.join(split_dir, "queries.csv")
            dp = os.path.join(split_dir, "documents.csv")
            queries_df.to_csv(qp, index=False, encoding="utf-8")
            print(f"Saved {len(queries_df)} queries to {qp}")
            documents_df.to_csv(dp, index=False, encoding="utf-8")
            print(f"Saved {len(documents_df)} documents to {dp}")


def main():
    parser = argparse.ArgumentParser(description="Build translation candidate CSVs from BeIR datasets.")

    parser.add_argument("--dataset_names", nargs="+", required=True,
                        help="Dataset names (e.g. BeIR/nfcorpus).")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Rows to sample per dataset (0 = all).")
    parser.add_argument("--max_document_segment_tokens", type=int, default=512,
                        help="Max tokens per document segment.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name for tokenisation.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Tokeniser override (defaults to model_name_or_path).")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"Output base directory (default: {DEFAULT_OUTPUT_PATH}).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--shard-size", type=int, default=None, dest="shard_size",
                        help="If set, split output into fixed-size shard CSVs and write a shard_manifest.json.")

    args = parser.parse_args()

    build_dataset_candidates(
        dataset_names=args.dataset_names,
        num_samples=args.num_samples,
        max_document_segment_tokens=args.max_document_segment_tokens,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        output_path=args.output_path,
        split=args.split,
        force=args.force,
        random_state=args.random_state,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()

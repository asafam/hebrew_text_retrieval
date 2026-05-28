from typing import List, Optional
import glob
import json
import pandas as pd
import argparse
import os
import re
from tqdm import tqdm
from data.translation_candidates import build_data


# Columns the BeIR candidate builder can emit conditionally (drop if no prompt
# template references them). Add new optional columns here as the schema grows.
_OPTIONAL_COLUMNS = {
    "context_id":   "include_context",
    "context_text": "include_context",
}


def _placeholders_in_prompts(prompt_files: list) -> set:
    """Return the set of {placeholder} names used across the given prompt YAMLs."""
    pat = re.compile(r"\{(\w+)\}")
    names: set = set()
    for pf in prompt_files or []:
        if not os.path.isfile(pf):
            print(f"  [warn] prompt file not found, ignoring: {pf}")
            continue
        with open(pf, encoding="utf-8") as f:
            names.update(pat.findall(f.read()))
    return names


def _builder_flags_from_prompts(prompt_files: list) -> dict:
    """Map prompt-placeholder usage to builder kwargs (e.g. include_context)."""
    used = _placeholders_in_prompts(prompt_files)
    flags: dict = {}
    for col, flag in _OPTIONAL_COLUMNS.items():
        if col in used:
            flags[flag] = True
    return flags


DEFAULT_OUTPUT_PATH = "outputs/translation/candidates"


def _write_shards(df: pd.DataFrame, slug_dir: str, prefix: str, shard_size: int) -> list:
    """Split df into fixed-size shard CSVs; return manifest shard list.

    Removes any pre-existing {prefix}_shard_*.csv first so a smaller rebuild
    (e.g. switching --split from 'all' to 'test') doesn't leave stale shards.
    """
    for stale in glob.glob(os.path.join(slug_dir, f"{prefix}_shard_*.csv")):
        os.remove(stale)

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
    builder_flags: Optional[dict] = None,
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
            **(builder_flags or {}),
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


def _load_candidates_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Build translation candidate CSVs from BeIR datasets.")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (e.g. config/translation/candidates.yaml). "
                             "Provides dataset names, shard sizes, and other defaults. "
                             "When given, --dataset_names and --model_name_or_path are optional.")
    parser.add_argument("--dataset_names", nargs="+", default=None,
                        help="Dataset names (e.g. BeIR/nfcorpus). Overrides config when given.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Rows to sample per dataset (0 = all).")
    parser.add_argument("--max_document_segment_tokens", type=int, default=None,
                        help="Max tokens per document segment.")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Model name for tokenisation.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Tokeniser override (defaults to model_name_or_path).")
    parser.add_argument("--output_path", type=str, default=None,
                        help=f"Output base directory (default: {DEFAULT_OUTPUT_PATH}).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test). For BeIR datasets, "
                             "use 'all' to union train/dev/test queries.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files.")
    parser.add_argument("--random_state", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--shard-size", type=int, default=None, dest="shard_size",
                        help="If set, split output into fixed-size shard CSVs and write a shard_manifest.json. "
                             "Ignored when --config is given (shard sizes come from config per dataset).")

    args = parser.parse_args()

    # ── Resolve parameters (CLI > config > defaults) ──────────────────────────
    cfg = _load_candidates_config(args.config) if args.config else {}
    ds_cfg = cfg.get("datasets", {})
    paths_cfg = cfg.get("paths", {})

    dataset_names = args.dataset_names or ds_cfg.get("names") or []
    num_samples = args.num_samples if args.num_samples is not None else ds_cfg.get("num_samples", 0)
    max_tokens = args.max_document_segment_tokens or ds_cfg.get("max_document_segment_tokens", 512)
    model = args.model_name_or_path or ds_cfg.get("tokenizer_model")
    output_path = args.output_path or paths_cfg.get("output", DEFAULT_OUTPUT_PATH)
    random_state = args.random_state if args.random_state is not None else ds_cfg.get("random_seed", 42)

    # Scan listed prompts to decide which optional columns to emit
    # (e.g. context_id/context_text only if a template references them).
    prompt_files  = cfg.get("prompts", [])
    builder_flags = _builder_flags_from_prompts(prompt_files)
    if prompt_files:
        print(f"Scanned {len(prompt_files)} prompt(s) -> builder flags: {builder_flags or '{}  (no optional columns)'}")

    if not dataset_names:
        parser.error("--dataset_names is required when --config is not given or has no datasets.names.")
    if not model:
        parser.error("--model_name_or_path is required when --config is not given or has no datasets.tokenizer_model.")

    if args.config:
        # Config mode: build each dataset with its own shard size from the YAML.
        shard_sizes = ds_cfg.get("shard_sizes", {})
        default_shard = ds_cfg.get("default_shard_size", 10000)
        for dataset_name in dataset_names:
            slug = dataset_name.replace("/", "_")
            shard_size = shard_sizes.get(slug, default_shard)
            build_dataset_candidates(
                dataset_names=[dataset_name],
                num_samples=num_samples,
                max_document_segment_tokens=max_tokens,
                model_name_or_path=model,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                output_path=output_path,
                split=args.split,
                force=args.force,
                random_state=random_state,
                shard_size=shard_size,
                builder_flags=builder_flags,
            )
    else:
        build_dataset_candidates(
            dataset_names=dataset_names,
            num_samples=num_samples,
            max_document_segment_tokens=max_tokens,
            model_name_or_path=model,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            output_path=output_path,
            split=args.split,
            force=args.force,
            random_state=random_state,
            shard_size=args.shard_size,
            builder_flags=builder_flags,
        )


if __name__ == "__main__":
    main()

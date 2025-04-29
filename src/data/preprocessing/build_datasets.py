from typing import List, Dict, Optional
import os
from itertools import chain
from pathlib import Path
import shutil
import argparse
import yaml
import dotenv
from datasets import IterableDataset, interleave_datasets
from data.preprocessing.utils import save_as_mds, save_as_txt
from data.preprocessing.dataset_formats.dataset_format_jsonl import DatasetFormatJSONL
from data.preprocessing.dataset_formats.dataset_format_hf import DatasetFormatHF

# Load environment variables from .env file
dotenv.load_dotenv()

COLUMNS = {
    "text": "str",  # Store JSON as a string
    "_source": "str",  # Filename source metadata
    "_row_number": "int"  # Row index metadata
}

def build(
        config_file: str,
        output_path: str,
        split: str,
        format: str,
        shard_size_limit: int = 67108864,
        buffer_size: int = 10000,
        random_state: int = 42
    ) -> None:
    """
    Build a dataset from multpiple sources

    Args:
        config_file (str): Path to the YAML configuration file.
        output_path (str): Path to the output directory.
        split (str): Split type (train/validation/test).
        format (str): Output format (mds/txt).
        shard_size_limit (int): Number of records per shard.
        buffer_size (int): Buffer size for shuffling.
        random_state (int): Random seed for sampling.h
    """
    # Load config file
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)

    datasets = []
    for config in configs:
        print(f"Processing {config['name']}...")
        if config["type"] == "hf":
            # Process HF datasets
            dataset = DatasetFormatHF(config.get("name"))\
                    .stream(
                        hf_dataset_args=config.get("args"),
                        filter_criteria=config.get("filter_criteria", {}),
                        limit=config.get("limit", 0),
                        tokens_limit=config.get("tokens_limit", 0),
                        split=split,
                        split_ratio=config.get("split_ratio", {}).get("validation", 0.1),
                        shuffle=True,
                        random_state=random_state,
                    )
            # dataset = IterableDataset.from_generator(generator_func)
            datasets.append(dataset)
        elif config["type"] == "jsonl":
            # Process JSONL files
            jsonl_files = config.get("files") or config["dir"]
            dataset = DatasetFormatJSONL(config.get("name"))\
                    .stream(
                        jsonl_files=jsonl_files,
                        exclude_jsonl_files=config.get("exclude_files", []),
                        split=split,
                        split_ratio=config.get("split_ratio", {}).get("validation", 0.1),
                        limit=config.get("limit", 0),
                        random_state=random_state
                    )
            # dataset = IterableDataset.from_generator(generator_func)
            datasets.append(dataset)

    merged_dataset = chain(*datasets)
    # Interleave datasets
    # interleaved_dataset = interleave_datasets(datasets, probabilities=None) # iterate over each dataset until it is exhausted
    # shuffled_dataset = interleaved_dataset.shuffle(buffer_size=buffer_size, seed=random_state)
    
    if format.lower() == "mds":
        # Ensure output directory is clean
        output_dir = os.path.join(output_path, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True) # Ensure the output_dir exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir) # Remove existing directory
        # Save dataset shards and generate index.json 
        save_as_mds(dataset=merged_dataset,
                    columns=COLUMNS,
                    output_dir=output_dir,
                    shard_size_limit=shard_size_limit) # Save the dataset in MDS format
    elif format.lower() == "txt":
        save_as_txt(dataset=merged_dataset,
                    column="text",
                    output_file=output_path) # Save the dataset in TXT format

def main():
    parser = argparse.ArgumentParser(description="Process JSONL files into MDS format with metadata")
    parser.add_argument("--config_file", type=str, required=True, help="Config yaml file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the data")
    parser.add_argument("--split", type=str, default="train", help="Split type (train/validation/test)")
    parser.add_argument("--format", type=str, default="mds", choices=["mds", "txt"], help="Output format (mds/txt)")
    parser.add_argument("--shard_size_limit", type=int, default=67108864, help="Number of records per shard")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Buffer size for shuffling")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()

    build(config_file=args.config_file,
          output_path=args.output_path,
          split=args.split,
          format=args.format,
          shard_size_limit=args.shard_size_limit,
          buffer_size=args.buffer_size,
          random_state=args.random_state)

if __name__ == "__main__":
    main()
from typing import List, Dict, Optional
import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
import shutil
import itertools
import argparse
from tqdm import tqdm
from streaming import MDSWriter
import random


COLUMNS = {
    "text": "str",  # Store JSON as a string
    "_source": "str",  # Filename source metadata
    "_row_number": "int"  # Row index metadata
}

def stream_jsonl_files(jsonl_files, split_ratio=0.1, random_state=42):
    """ 
    Generator to stream JSONL rows as either train or validation split.
    
    Parameters:
        jsonl_files (list): List of JSONL file paths.
        split_ratio (float): Proportion of samples to use for validation.
        random_state (int): Random seed for reproducibility.
    
    Yields:
        dict: {"text": ..., "_source": ..., "_row_number": ...}
    """
    assert split in {"train", "validation"}, "split must be 'train' or 'validation'"
    
    random.seed(random_state)
    
    for file_path in tqdm(jsonl_files, desc="Reading JSONL files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            samples_count = sum(1 for _ in f)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            validation_size = int(split_ratio * samples_count)
            validation_indexes = random.sample(range(0, samples_count), validation_size)
            validation_indexes_stack = sorted(validation_indexes)
            for idx, line in tqdm(enumerate(f), desc=f"Reading {file_path}"):
                try:
                    data = json.loads(line)
                    record = {
                        "text": data["text"],
                        "_source": os.path.basename(file_path),
                        "_row_number": idx
                    }

                    # Yield record based on the split type
                    if validation_indexes_stack and idx == validation_indexes_stack[0]:
                        split = "validation"
                        validation_indexes_stack.pop(0)
                    else:
                        split = "train"
                        
                    yield record, split
                    
                except json.JSONDecodeError:
                    print(f"Skipping malformed line {idx} in {file_path}")


def save_as_mds(jsonl_files: List[str], 
                columns: Dict[str, str], 
                output_dir: str, 
                size_limit: int, 
                compression:Optional[int] = None):
    print(f"ℹ️ Processing {len(jsonl_files)} JSONL files into MDS format")

    # Save split dir data
    split_dir = {}
    for split in ["train", "validation"]:
        split_dir[split] = os.path.join(output_dir, split)
        os.makedirs(split_dir[split], exist_ok=True)

    with MDSWriter(out=split_dir['train'], columns=columns, size_limit=size_limit, compression=compression) as train_writer, \
        MDSWriter(out=split_dir['validation'], columns=columns, size_limit=size_limit, compression=compression) as val_writer:
        for record, split in tqdm(stream_jsonl_files(jsonl_files), desc="Processing JSONL files"):
            if split == "train":
                train_writer.write(record)
            elif split == "validation":
                val_writer.write(record)

    print(f"✅ Dataset saved in MDS format at {output_dir}")


def transform(jsonl_files: List[str]|str,
              output_dir: str,
              size_limit: int = 500000,
              exclude_jsonl_files: Optional[List[str]] = None) -> None:
    """ Transform JSONL files to MDS format and save dataset shards """
    # Get list of JSONL files
    if len(jsonl_files) == 1 and os.path.isdir(jsonl_files[0]):
        jsonl_files_path = jsonl_files[0]
        jsonl_files = [str(f) for f in list(Path(jsonl_files_path).glob("*.jsonl"))]
    
    # Exclude files if needed
    if exclude_jsonl_files:
        jsonl_files = [file for file in jsonl_files if file not in exclude_jsonl_files]

    # Ensure output directory is clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Remove existing directory
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset shards and generate index.json
    save_as_mds(jsonl_files=jsonl_files,
                columns=COLUMNS,
                output_dir=output_dir,
                size_limit=size_limit) # Stream the data and save in MDS format in chunks


def main():
    parser = argparse.ArgumentParser(description="Process JSONL files into MDS format with metadata")
    parser.add_argument("--jsonl_files", type=str, nargs='+', required=True, help="List of input JSONL files or a single directory")
    parser.add_argument("--exclude_jsonl_files", type=str, nargs='+', required=False, help="List of JSONL files to exclude")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for MDS dataset")
    parser.add_argument("--size_limit", type=int, default=500000, help="Number of records per shard")

    args = parser.parse_args()

    transform(jsonl_files=args.jsonl_files, 
              output_dir=args.output_dir,
              size_limit=args.size_limit,
              exclude_jsonl_files=args.exclude_jsonl_files)

if __name__ == "__main__":
    main()
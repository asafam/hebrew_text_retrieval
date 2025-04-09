from typing import List, Dict, Optional
import os
import shutil
import argparse
import yaml
from tqdm import tqdm
import tiktoken
from streaming import MDSWriter
from data.preprocessing.dataset_formats.dataset_format_jsonl import DatasetFormatJSONL
from data.preprocessing.dataset_formats.dataset_format_hf import DatasetFormatHF



COLUMNS = {
    "text": "str",  # Store JSON as a string
    "_source": "str",  # Filename source metadata
    "_row_number": "int"  # Row index metadata
}

tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize(text: str) -> List[str]:
    return tokenizer.encode(text)

def save_as_mds(dataset_streams: List[str],
                columns: Dict[str, str], 
                output_dir: str, 
                size_limit: int, 
                compression: Optional[int] = None,
                tokens_limit_map: dict[str, int] = {}):
    print(f"ℹ️ Processing {len(dataset_streams)} streams into MDS format")
    os.makedirs(output_dir, exist_ok=True) # Ensure the output_dir exists
    with MDSWriter(out=output_dir, columns=columns, size_limit=size_limit, compression=compression) as writer:
        iterators = [iter(ds) for ds in dataset_streams]
        active = [True] * len(iterators)
        total_tokens = [0] * len(iterators)
        tokens_limits = [tokens_limit_map.get(ds.name, 0) for ds in dataset_streams]

        while any(active):
            for i, it in enumerate(iterators):
                if not active[i]:
                    continue
                try:
                    # Ensure the tokens_limit for this stream is not exceeded
                    if total_tokens[i] > tokens_limits[i]:
                        active[i] = False
                        print(f"⚠️ {dataset_streams[i].name} stream exceeded token limit, stopping.")
                        continue
                    
                    # Write the next record to MDS
                    record = next(it)
                    writer.write(record)

                    # Update total tokens for the current stream
                    total_tokens[i] += len(tokenize(record["text"])) if tokenizer else 0
                except StopIteration:
                    active[i] = False

    print(f"✅ Data saved in MDS format at {output_dir}")


def transform(config_file: str,
              split: str,
              output_dir: str,
              size_limit: int = 67108864) -> None:
    """ Transform JSONL files to MDS format and save dataset shards """
    # Load config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure output directory is clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Remove existing directory
    os.makedirs(output_dir, exist_ok=True)


    dataset_streams = []
    for dataset in config:
        if dataset["type"] == "hf":
            # Process HF datasets
            hf_dataset_stream = DatasetFormatHF(config.get("name"))\
                .stream(split=split, **config.get("args", {}))
            dataset_streams.append(hf_dataset_stream)
        elif dataset["type"] == "jsonl":
            # Process JSONL files
            jsonl_files = dataset.get("files") or dataset["dir"]
            exclude_jsonl_files = config.get("exclude_files", [])
            split_ratio = config.get("split_ratio", {}).get("validation", 0.1)
            jsonl_dataset_stream = DatasetFormatJSONL(config.get("name"))\
                .stream(
                    jsonl_files=jsonl_files,
                    exclude_jsonl_files=exclude_jsonl_files,
                    split=split,
                    split_ratio=split_ratio
                )
            dataset_streams.append(jsonl_dataset_stream)
    
    # Save dataset shards and generate index.json 
    save_as_mds(dataset_streams=dataset_streams,
                columns=COLUMNS,
                output_dir=os.path.join(output_dir, split),
                tokens_limit_map={x["name"]: x["tokens_limit"]["split"] for x in config.keys()},
                size_limit=size_limit) # Stream the data and save in MDS format in chunks

def tokenize(text: str) -> List[str]:
    return tokenizer.encode(text)

def main():
    parser = argparse.ArgumentParser(description="Process JSONL files into MDS format with metadata")
    parser.add_argument("--config_file", type=str, required=True, help="Config yaml file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for MDS dataset")
    parser.add_argument("--size_limit", type=int, default=67108864, help="Number of records per shard")

    args = parser.parse_args()

    transform(config_file=args.config_file,
              output_dir=args.output_dir,
              size_limit=args.size_limit)

if __name__ == "__main__":
    main()
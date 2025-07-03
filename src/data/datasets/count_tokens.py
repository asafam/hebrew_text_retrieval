from typing import List, Dict, Any
import argparse
import os
import glob
from datasets import load_dataset
import sentencepiece as spm
from transformers import AutoTokenizer
from tqdm import tqdm

def get_tokenizer_func(tokenizer_path: str = None):
    if tokenizer_path is None:
        print("No tokenizer path provided, using space delimiter tokenizer.")
        def encode(text, **kwargs):
            return {"input_ids": text.split(), "attention_mask": [1] * len(text.split())}
        tokenizer = encode
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def main(data_path: str,
         tokenizer_path: str = None,
         exclude_source: List[str] = [],
         include_source: List[str] = []):
    # Paths to all files
    print(f"Searching for JSON files in {data_path}")
    all_files = glob.glob(data_path, recursive=True)
    assert len(all_files) > 0, "No JSON files found in the specified directory."
    print(f"Found {len(all_files):,} JSON files in {data_path}")

    # Filter files based on include/exclude source lists
    if include_source or exclude_source:
        print(f"Filtering files based on include/exclude source lists: {include_source}, {exclude_source}")
        files = [file for file in all_files if any(source in file for source in include_source) and not any(source in file for source in exclude_source)]
        print(f"After filtering, {len(all_files):,} files remain.")
    else:
        print("No include/exclude source lists provided, using all files.")
        files = all_files
    
    # Load dataset in streaming mode
    dataset = load_dataset("json", data_files=files, split="train", streaming=True)

    # Load tokenizer
    tokenizer = get_tokenizer_func(tokenizer_path)

    # Define the field you want to tokenize
    text_field = "text"  # change to the correct column name

    # Count tokens
    token_count = 0
    example_count = 0

    pbar = tqdm(dataset, desc="Tokenizing", dynamic_ncols=True, unit=" examples")

    for example in pbar:
        text = example[text_field]
        tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False, add_special_tokens=True, max_length=8192, truncation=True)
        token_count += len(tokens["input_ids"])
        example_count += 1
        pbar.set_postfix_str(f"tokens={token_count:,}")

    print(f"Total tokens: {token_count:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dolma dataset")
    parser.add_argument("--data_path", type=str, help="Path to the Dolma dataset files")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer model file (optional)")
    parser.add_argument("--exclude_source", type=str, nargs='*', default=[], help="List of sources to exclude from the dataset")
    parser.add_argument("--include_source", type=str, nargs='*', default=[], help="List of sources to include in the dataset")
    args = parser.parse_args()
    
    main(data_path=args.data_path,
         tokenizer_path=args.tokenizer_path,
         exclude_source=args.exclude_source,
         include_source=args.include_source)
    print("Finished processing and saving the dataset.")
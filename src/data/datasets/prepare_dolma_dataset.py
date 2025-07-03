from typing import List, Dict, Any
import argparse
import os
import glob
from datasets import load_dataset, IterableDataset
from itertools import islice, chain
import json
import sentencepiece as spm
from transformers import AutoTokenizer
from datetime import datetime, timedelta
from tqdm import tqdm

def get_tokenizer_func(tokenizer_path: str = None):
    if tokenizer_path is None or tokenizer_path.endswith('.model'):
        model_file = "/home/nlp/achimoa/workspace/hebrew_text_retrieval/outputs/tokenizer/HebrewModernBERT_mixed_1M_100K.model"
        print(f"Loading SentencePiece tokenizer from {model_file}")
        tokenizer = spm.SentencePieceProcessor(model_file=model_file).encode
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def filter_valid_text(example):
    return example.get("text") not in [None, "", "null"]

def keep_text_source(example):
    return {
        "id": example["id"],
        "text": example["text"],
        "source": example.get("source")
    }

def load_and_sample_by_tokens(files, token_budget, tokenizer, shuffle_buffer, seed: int = 42, text_field: str = "text"):
    print(f"Loading dataset from {len(files):,} files with token budget: {token_budget:,}")
    dataset = (
        load_dataset("json", data_files=files, split="train", streaming=True)
        # .filter(filter_valid_text)
        # .map(keep_text_source)
        .shuffle(buffer_size=shuffle_buffer, seed=seed)
    )

    def generator():
        token_count = 0
        example_count = 0

        pbar = tqdm(dataset, desc="Tokenizing", dynamic_ncols=True, unit=" examples")

        start_datetime = datetime.now()
        for example in pbar:
            text = example[text_field]
            tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False, add_special_tokens=True, max_length=8192, truncation=True)
            token_count += len(tokens["input_ids"])
            example_count += 1
            pbar.set_postfix_str(f"tokens={token_count:,}")
                
            if token_count > token_budget:
                print(f"Reached token budget of {token_budget:,} tokens after processing {example_count:,} examples.")
                elapsed_time = datetime.now() - start_datetime
                print(f"Elapsed time: {timedelta(seconds=elapsed_time.total_seconds())}")
                print(f"Total examples processed: {example_count:,}")
                print(f"Total tokens counted: {token_count:,}")
                print(f"Stopping data generation.")
                break

            data = {
                'id': example['id'],
                'text': example['text'],
                'source': example['source'],
            }
            yield data

    return generator

def main(output_file: str,
         token_budget: int,
         data_path: str,
         tokenizer_path: str = None,
         exclude_source: List[str] = [],
         include_source: List[str] = [],
         shuffle_buffer: int = 1_000_000,
         random_state: int = 42):
    #  Load the tokenizer
    print(f"Loading tokenizer...")
    tokenizer = get_tokenizer_func(tokenizer_path)

    # Paths to all files
    print(f"Searching for JSON files in {data_path}")
    files = glob.glob(os.path.join(data_path, '**/*.json.gz'), recursive=True)
    assert len(files) > 0, "No JSON files found in the specified directory."
    print(f"Found {len(files):,} JSON files in {data_path}")

    # Filter files based on include/exclude source lists
    if include_source or exclude_source:
        print(f"Filtering files based on include/exclude source lists: {include_source}, {exclude_source}")
        if include_source:
            files = [file for file in files if any(source in file for source in include_source)]

        if exclude_source:
            files = [file for file in files if not any(source in file for source in exclude_source)]
        print(f"After filtering, {len(files):,} files remain.")
    else:
        print("No include/exclude source lists provided, using all files.")

    print(f"Sampling {token_budget:,} tokens from other files.")
    data_generator = load_and_sample_by_tokens(
        files=files,
        token_budget=token_budget,
        tokenizer=tokenizer,
        shuffle_buffer=shuffle_buffer,
        seed=random_state
    )

    generator_dataset = IterableDataset.from_generator(generator=data_generator)

    # output_file = f"data/dolma/corpus_sampled_50B.jsonl"
    with open(output_file, "w") as f:
        for sample in tqdm(generator_dataset):
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dolma dataset")
    parser.add_argument("--output_file", type=str, help="Output file for the dataset")
    parser.add_argument("--token_budget", type=int, default=25_000_000_000, help="Token budget for sampling")
    parser.add_argument("--data_path", type=str, default="data/dolma", help="Path to the Dolma dataset files")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer model file (optional)")
    parser.add_argument("--exclude_source", type=str, nargs='*', default=[], help="List of sources to exclude from the dataset")
    parser.add_argument("--include_source", type=str, nargs='*', default=[], help="List of sources to include in the dataset")
    parser.add_argument("--shuffle_buffer", type=int, default=1_000_000, help="Buffer size for shuffling the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    
    main(output_file=args.output_file,
         token_budget=args.token_budget,
         data_path=args.data_path,
         tokenizer_path=args.tokenizer_path,
         exclude_source=args.exclude_source,
         include_source=args.include_source,
         shuffle_buffer=args.shuffle_buffer,
         random_state=args.seed)
    print("Finished processing and saving the dataset.")
from typing import List, Dict, Any
import argparse
import glob
from datasets import load_dataset, IterableDataset
from itertools import islice, chain
import json
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from datetime import datetime, timedelta
from tqdm import tqdm

model_file = "/home/nlp/achimoa/workspace/hebrew_text_retrieval/outputs/tokenizer/HebrewModernBERT_mixed_1M_100K.model"
tokenizer = spm.SentencePieceProcessor(model_file=model_file)

def tokenize(text: str) -> List[str]:
    return tokenizer.encode(text)

def filter_valid_text(example):
    return example.get("text") not in [None, "", "null"]

def keep_text_source(example):
    return {
        "id": example["id"],
        "text": example["text"],
        "source": example.get("source")
    }

def load_and_sample_by_tokens(files, token_budget, tokenizer, shuffle_buffer, seed):
    dataset = (
        load_dataset("json", data_files=files, split="train", streaming=True)
        # .filter(filter_valid_text)
        # .map(keep_text_source)
        .shuffle(buffer_size=shuffle_buffer, seed=seed)
    )

    def generator():
        total_tokens = 0
        start_datetime = datetime.now()
        for i, example in enumerate(dataset):
            token_count = len(tokenizer.encode(example["text"]))
            if i % 1_000_000_000 == 0 or i in [10, 100, 1000, 10_000]:
                elapsed_time = (datetime.now() - start_datetime).total_seconds()
                formatted_time = str(timedelta(seconds=int(elapsed_time)))
                print(f"[{formatted_time}] Processed {i:,} samples, {total_tokens:,} tokens")
                
            if total_tokens + token_count > token_budget:
                print(f"Reached {total_tokens:,} token budget limit ({i+1} samples). Stopping.")
                break
            total_tokens += token_count
            data = {
                'id': example['id'],
                'text': example['text'],
                'source': example['source'],
            }
            yield data

    return generator

def main(token_budget, output_file, random_state):
    # Paths to all files
    all_files = glob.glob('data/**/*.json.gz', recursive=True)

    assert len(all_files) > 0, "No JSON files found in the specified directory."

    # Separate StarCoder files from the rest
    starcoder_files = [f for f in all_files if 'starcoder' in f]
    other_files = [f for f in all_files if 'starcoder' not in f]

    def combined_generator():
        other_gen = load_and_sample_by_tokens(
            other_files,
            token_budget=token_budget,
            tokenizer=tokenizer,
            shuffle_buffer=1_000_000,
            seed=random_state
        )
        starcoder_gen = load_and_sample_by_tokens(
            starcoder_files,
            token_budget=token_budget,
            tokenizer=tokenizer,
            shuffle_buffer=100_000,
            seed=random_state
        )
        return chain(starcoder_gen(), other_gen())

    combined_dataset = IterableDataset.from_generator(combined_generator)

    shuffled_dataset = combined_dataset.shuffle(buffer_size=1_000_000, seed=42)

    # output_file = f"data/dolma/corpus_sampled_50B.jsonl"
    with open(output_file, "w") as f:
        for sample in tqdm(shuffled_dataset):
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dolma dataset")
    parser.add_argument("--output_file", type=str, help="Output file for the dataset")
    parser.add_argument("--token_budget", type=int, default=25_000_000_000, help="Token budget for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    
    main(token_budget=args.token_budget,
         output_file=args.output_file, 
         random_state=args.seed)
    print("Finished processing and saving the dataset.")
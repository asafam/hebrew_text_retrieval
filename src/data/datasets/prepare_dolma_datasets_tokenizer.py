import glob
from datasets import load_dataset, interleave_datasets
from itertools import chain

# Paths to all files
all_files = glob.glob('/path/to/data/**/*.json.gz', recursive=True)

# Separate StarCoder files from the rest
starcoder_files = [f for f in all_files if 'starcoder' in f]
other_files = [f for f in all_files if 'starcoder' not in f]

# Load StarCoder dataset
starcoder_dataset = load_dataset(
    'json',
    data_files={'train': starcoder_files},
    split='train',
    streaming=True
)

# Load other dataset
other_dataset = load_dataset(
    'json',
    data_files={'train': other_files},
    split='train',
    streaming=True
)

starcoder_sample = starcoder_dataset.shuffle(buffer_size=100_000, seed=42).take(250_000)
other_sample = other_dataset.shuffle(buffer_size=1_000_000, seed=42).take(250_000)

datasets = [starcoder_sample, other_sample]
interleaved_dataset = interleave_datasets(datasets, probabilities=None) # iterate over each dataset until it is exhausted
shuffled_dataset = interleaved_dataset.shuffle(buffer_size=100000, seed=42)

import json
from pathlib import Path

output_path = Path("data/dolma/tokenizer_corpus_500K.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for record in shuffled_dataset:
        f.write(json.dumps(record['text']) + "\n")

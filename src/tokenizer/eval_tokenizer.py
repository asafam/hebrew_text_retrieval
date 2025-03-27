from typing import Dict
import sentencepiece as spm
import argparse
import re
import os
import pandas as pd
from tqdm import tqdm

def eval_compression(model_file: str, validation_file: str) -> Dict:
    # Load trained tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=model_file)
    print(f"Loaded tokenizer from {model_file}")

    # Define validation text samples
    with open(validation_file, "r") as f:
        validation_texts = f.readlines()

    # Compute compression rate
    total_chars = sum(len(text) for text in tqdm(validation_texts, desc="Counting characters"))
    total_tokens = sum(len(tokenizer.encode(text, out_type=int)) for text in tqdm(validation_texts, desc="Tokenizing"))

    compression_rate = total_chars / total_tokens
    print(f"Compression Rate: {compression_rate:.2f} characters per token")

    return dict(compression_rate=compression_rate, 
                total_chars=total_chars, 
                total_tokens=total_tokens,)


def convert_to_number(value):
    """Convert 'XM' or 'XK' to actual numbers."""
    multiplier = {'M': 1_000_000, 'K': 1_000}
    return int(value[:-1]) * multiplier[value[-1]]


def main(model_file, eval_file, output_file):
    df = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame(columns=["Train Size", "Vocab Size", "Compression Rate", "model_file"])

    if df[df['model_file'] == model_file].shape[0] > 0:
        print(f"Skipping {model_file}")
        return
    
    print(model_file)

    matches = re.findall(r'\d+[MK]', model_file)
    numbers = [convert_to_number(match) for match in matches]
    train_size, vocab_size = numbers

    results = eval_compression(model_file, eval_file)
    
    data = {
        "Train Size": train_size,
        "Vocab Size": vocab_size,
        "Compression Rate": results["compression_rate"],
        "model_file": model_file
    }
    print(data)
    df.loc[len(df)] = data

    # Save the results
    df.to_csv(output_file, index=False)
    print(f"Saved results for {model_file} in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    main(
        model_file=args.model_file, 
        eval_file=args.eval_file,
        output_file=args.output_file
    ) 

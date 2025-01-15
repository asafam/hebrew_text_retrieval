import argparse
import torch
from tqdm import tqdm
from translation.translate import *


def main():
    parser = argparse.ArgumentParser(description="Translate queries and documents using a specified model.")

    parser.add_argument('--source_file_path', type=str, nargs='+', required=True, help="Paths to the source input files containing queries.")
    parser.add_argument('--prompt_file_name', type=str, required=True, help="File name for the translation prompt.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the translation model.")
    parser.add_argument('--batch_size', type=int, default=32, help="Number of queries to translate in each batch.")
    parser.add_argument('--max_new_tokens', type=int, default=None, help="Maximum number of tokens to generate per query.")
    parser.add_argument('--use_cached_prefix', action='store_true', help="Use cached results for translation.")
    parser.add_argument('--force', action='store_true', help="Force re-translation if output file exists.")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for data_file_path in tqdm(args.data_file_paths, desc="Data files"):
        print(f"Translating {data_file_path}...")
        run_translation_pipeline(
            source_file_path=args.source_file_path,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_cached_prefix=args.use_cached_prefix,
            device=device,
            force=args.force
        )
            

if __name__ == "__main__":
    main()
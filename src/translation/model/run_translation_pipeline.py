import argparse
import torch
from tqdm import tqdm
from translation.model.translate import *


def main():
    parser = argparse.ArgumentParser(description="Translate queries and documents using a specified model.")

    parser.add_argument('--source_file_paths', type=str, nargs='+', required=True, help="Paths to the source input files containing queries.")
    parser.add_argument('--prompt_file_name', type=str, required=True, help="File name for the translation prompt.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the translation model.")
    parser.add_argument('--batch_size', type=int, default=32, help="Number of queries to translate in each batch.")
    parser.add_argument('--max_new_tokens', type=int, default=None, help="Maximum number of tokens to generate per query.")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of texts to translate.")
    parser.add_argument('--use_cached_prefix', action='store_true', help="Use cached results for translation.")
    parser.add_argument('--force', action='store_true', help="Force re-translation if output file exists.")
    parser.add_argument('--english_key', type=str, default="אנגלית", help="Key for the English translation in the prompt file.")
    parser.add_argument('--hebrew_key', type=str, default="עברית", help="Key for the Hebrew translation in the prompt file.")
    parser.add_argument('--context_key', type=str, default="הקשר", help="Key for the English context translation in the prompt file.")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for source_file_path in tqdm(args.source_file_paths, desc="Data files"):
        print(f"Translating {source_file_path}...")
        run_translation_pipeline(
            source_file_path=source_file_path,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            limit=args.limit,
            use_cached_prefix=args.use_cached_prefix,
            force=args.force,
            english_key=args.english_key,
            hebrew_key=args.hebrew_key,
            context_key=args.context_key
        )
            

if __name__ == "__main__":
    main()
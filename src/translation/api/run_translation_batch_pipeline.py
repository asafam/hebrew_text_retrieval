import argparse
import torch
from tqdm import tqdm
from translation.api.translate_batch import *


def main():
    parser = argparse.ArgumentParser(description="Translate queries and documents using a specified model.")

    parser.add_argument('--source_file_paths', type=str, nargs='+', required=True, help="Paths to the source input files containing the queries and docuemnts.")
    parser.add_argument('--prompt_file_name', type=str, required=True, help="File name for the translation prompt.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the translation model.")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of texts to translate.")
    parser.add_argument('--force', action='store_true', help="Force re-translation if output file exists.")
    parser.add_argument('--text_key', type=str, default="{source_lang}", help="Key for the {source_lang} translation in the prompt file.")
    parser.add_argument('--translation_key', type=str, default="Hebrew", help="Key for the Hebrew translation in the prompt file.")
    parser.add_argument('--context_key', type=str, default="Context", help="Key for the {source_lang} context translation in the prompt file.")

    args = parser.parse_args()

    for source_file_path in tqdm(args.source_file_paths, desc="Data files"):
        print(f"Translating {source_file_path}...")
        run_translation_pipeline(
            source_file_path=source_file_path,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            limit=args.limit,
            force=args.force,
            text_key=args.text_key,
            translation_key=args.translation_key,
            context_key=args.context_key
        )
            

if __name__ == "__main__":
    main()
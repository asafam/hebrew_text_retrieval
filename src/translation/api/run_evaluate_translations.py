import argparse
from tqdm import tqdm
from translation.api.evaluate_translations import *


def main():
    parser = argparse.ArgumentParser(description="Translate queries and documents using a specified model.")

    parser.add_argument('--source_file_paths', type=str, nargs='+', required=True, help="Paths to the source input files containing the queries and docuemnts.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for the translated queries and documents.")
    parser.add_argument('--gold_file_path', type=str, required=False, help="Path to the gold translation file containing the ground truth translated queries and docuemnts.")
    parser.add_argument('--prompt_file_name', type=str, required=True, help="Translation prompt file name.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the translation model.")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of texts to translate.")
    parser.add_argument('--english_key', type=str, default="Text", help="Key for the English translation in the prompt file.")
    parser.add_argument('--hebrew_key', type=str, default="Translation", help="Key for the Hebrew translation in the prompt file.")
    parser.add_argument('--force', action='store_true', help="Force re-translation if output file exists.")
    parser.add_argument('--parallel', action='store_true', help="Evaluate translations in parallel.")

    args = parser.parse_args()

    for source_file_path in tqdm(args.source_file_paths, desc="Data files"):
        print(f"Translating {source_file_path}...")
        run_evaluate_translations(
            source_file_path=source_file_path,
            output_dir=args.output_dir,
            gold_file_path=args.gold_file_path,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            limit=args.limit,
            english_key=args.english_key,
            hebrew_key=args.hebrew_key,
            force=args.force,
            parallel=args.parallel
        )
            
if __name__ == "__main__":
    main()
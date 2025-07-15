import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from translation.api.translate import *

# Automatically load .env from current directory
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Translate queries and documents using a specified model.")

    parser.add_argument('--source_file_paths', type=str, nargs='+', required=True, help="Paths to the source input files containing the queries and docuemnts.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for the translated queries and documents.")
    parser.add_argument('--prompt_file_name', type=str, required=True, help="Translation prompt file name.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the translation model.")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of texts to translate.")
    parser.add_argument('--english_key', type=str, default="English", help="Key for the English translation in the prompt file.")
    parser.add_argument('--hebrew_key', type=str, default="Hebrew", help="Key for the Hebrew translation in the prompt file.")
    parser.add_argument('--context_key', type=str, default="Context", help="Key for the English context translation in the prompt file.")
    parser.add_argument('--hebrew_key_query', type=str, default="Hebrew Query", help="Key for the Hebrew query translation in the prompt file.")
    parser.add_argument('--hebrew_key_document', type=str, default="Hebrew Document", help="Key for the Hebrew document translation in the prompt file.")
    parser.add_argument('--response_format', type=str, default="Translation", choices=("Translation", "UnifiedTranslation", "UnifiedSingleSentenceTranslation"), help="Response format for the translation.")
    parser.add_argument('--sleep_time', type=int, default=0, help="Sleep time between requests to avoid rate limiting.")
    parser.add_argument('--force', action='store_true', help="Force re-translation if output file exists.")
    parser.add_argument('--parallel', action='store_true', help="Translate in parallel.")

    args = parser.parse_args()

    for source_file_path in tqdm(args.source_file_paths, desc="Data files"):
        print(f"Translating {source_file_path}...")
        run_translation_pipeline(
            source_file_path=source_file_path,
            output_dir=args.output_dir,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            limit=args.limit,
            english_key=args.english_key,
            hebrew_key=args.hebrew_key,
            context_key=args.context_key,
            hebrew_key_query=args.hebrew_key_query,
            hebrew_key_document=args.hebrew_key_document,
            response_format=args.response_format,
            sleep_time=args.sleep_time,
            force=args.force,
            parallel=args.parallel
        )
            
if __name__ == "__main__":
    main()
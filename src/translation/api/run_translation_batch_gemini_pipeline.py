import argparse
from tqdm import tqdm
from translation.api.translate_batch_gemini import run_translation_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Submit translation batch jobs to the Gemini Batch API."
    )
    parser.add_argument("--source_file_paths", type=str, nargs="+", required=True,
                        help="Paths to source CSV files (queries/documents).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for translated output CSV files.")
    parser.add_argument("--prompt_file_name", type=str, required=True,
                        help="YAML file with system/user prompt templates.")
    parser.add_argument("--model_name", type=str,
                        default="gemini-3.1-flash-lite",
                        help="Gemini model name for batch inference.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit the number of texts to translate (0 = all).")
    parser.add_argument("--force", action="store_true",
                        help="Re-translate even if output already exists.")
    parser.add_argument("--text_key", type=str, default="{source_lang}",
                        help="Column key for source text in the prompt template.")
    parser.add_argument("--translation_key", type=str, default="Hebrew",
                        help="Column key for the translation field.")
    parser.add_argument("--context_key", type=str, default="Context",
                        help="Column key for context in the prompt template.")

    args = parser.parse_args()

    for source_file_path in tqdm(args.source_file_paths, desc="Data files"):
        print(f"Submitting batch job for {source_file_path}...")
        run_translation_pipeline(
            source_file_path=source_file_path,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            output_dir=args.output_dir,
            limit=args.limit,
            force=args.force,
            text_key=args.text_key,
            translation_key=args.translation_key,
            context_key=args.context_key,
        )


if __name__ == "__main__":
    main()

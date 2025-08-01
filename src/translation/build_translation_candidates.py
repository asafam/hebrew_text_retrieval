from typing import List
import pandas as pd
import argparse
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from data.translation_candidates import build_data


# Function to load and process datasets
def build_dataset_candidates(dataset_names: List[str], 
                             num_samples: int, 
                             output_path: str, 
                             max_document_segment_tokens: int, 
                             model_name_or_path: str, 
                             tokenizer_name_or_path: str = None,
                             split: str = 'test',
                             force: bool = False,
                             random_state: int = 42) -> None:
    for dataset_name in tqdm(dataset_names, desc="Processing datasets: "):
        print(f"Processing dataset: {dataset_name}")
        
        # Determine file paths
        dataset_name_slug = dataset_name.replace("/", "_")
        queries_output_path = f"{output_path}/{dataset_name_slug}/{split}/queries.csv"
        documents_output_path = f"{output_path}/{dataset_name_slug}/{split}/documents.csv"

        if not force and os.path.exists(queries_output_path) and os.path.exists(documents_output_path):
            print(f"Skipping {dataset_name} as files already exist and --force is not set.")
            continue

        # Build the data
        data = build_data(dataset_name=dataset_name, 
                          model_name_or_path=model_name_or_path,
                          tokenizer_name_or_path=tokenizer_name_or_path,
                          n=num_samples, 
                          max_tokens=max_document_segment_tokens,
                          split=split,
                          random_state=random_state)
        queries, documents = data

        # Convert to DataFrames
        queries_df = pd.DataFrame(queries)
        documents_df = pd.DataFrame(documents)

        for df in [queries_df, documents_df]:
            df['dataset_name'] = dataset_name
            df['tokenizer'] = tokenizer_name_or_path or model_name_or_path

        
        # Save to CSV
        os.makedirs(os.path.dirname(queries_output_path), exist_ok=True) # Create the output path folder if it does not exist
        queries_df.to_csv(queries_output_path, index=False, encoding='utf-8')
        print(f"Saved {len(queries_df)} queries to {queries_output_path}")

        os.makedirs(os.path.dirname(documents_output_path), exist_ok=True) # Create the output path folder if it does not exist
        documents_df.to_csv(documents_output_path, index=False, encoding='utf-8')
        print(f"Saved {len(documents_df)} documents to {documents_output_path}")


# Main function
def main():
    parser = argparse.ArgumentParser(description="Process and tokenize datasets.")
    
    parser.add_argument('--dataset_names', nargs='+', required=True, help="List of space separated dataset names to process.")
    parser.add_argument('--num_samples', type=int, required=True, help="Number of samples to load from each dataset.")
    parser.add_argument('--max_document_segment_tokens', type=int, default=2048, required=True, help="Maximum number of tokens per document segment.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Name of the model to use for tokenization.")
    parser.add_argument('--tokenizer_name_or_path', type=str, default=None, help="Name of the tokenizer to use for tokenization. If not provided, model_name_or_path will be used.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output CSV files.")
    parser.add_argument('--split', type=str, default='test', help="Dataset split to use (default: 'test').")
    parser.add_argument('--force', action='store_true', help="Override existing files for the datasets.")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    build_dataset_candidates(
        dataset_names=args.dataset_names,
        num_samples=args.num_samples,
        max_document_segment_tokens=args.max_document_segment_tokens,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        output_path=args.output_path,
        split=args.split,
        force=args.force,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()

from typing import List
import pandas as pd
import argparse
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from data.beir import build_data


BeIR = {
    'Misc': ['BeIR/msmarco'], 
    'Fact checking': ['BeIR/fever', 'BeIR/climate-fever', 'BeIR/scifact'],
    'Citation-Prediction': ['BeIR/scidocs'],
    'Duplicate question retrieval': ['BeIR/quora'], # CQADupStack
    'Argument retrieval': ['BeIR/arguana'], # Touche-2020
    'News retrieval': [], # TREC-NEWS, Robust04
    'Question answering': ['BeIR/nq', 'BeIR/hotpotqa'], # FiQA-2018
    'Tweet retrieval': [], # Signal-1M
    'Bio-medical IR': ['BeIR/trec-covid', 'BeIR/nfcorpus'], # BioASQ
    'Entity retrieval': ['BeIR/dbpedia-entity'],
}


# Function to load and process datasets
def build_dataset_candidates(dataset_names: List[str], 
                             num_samples: int, 
                             max_document_segment_tokens: int, 
                             model_name: str, 
                             output_path: str, 
                             force: bool = False,
                             random_seed: int = 42) -> None:
    for dataset_name in tqdm(dataset_names, desc="Processing datasets: "):
        print(f"Processing dataset: {dataset_name}")
        
        # Determine file paths
        dataset_name_slug = dataset_name.replace("/", "_")
        queries_output_path = f"{output_path}/{dataset_name_slug}/queries.csv"
        documents_output_path = f"{output_path}/{dataset_name_slug}/documents.csv"

        if not force and os.path.exists(queries_output_path) and os.path.exists(documents_output_path):
            print(f"Skipping {dataset_name} as files already exist and --force is not set.")
            continue

        # Build the data
        data = build_data(dataset_name=dataset_name, 
                          model_name=model_name,
                          n=num_samples, 
                          max_tokens=max_document_segment_tokens,
                          random_seed=random_seed)
        queries, documents = data

        # Convert to DataFrames
        queries_df = pd.DataFrame(queries)
        documents_df = pd.DataFrame(documents)

        for df in [queries_df, documents_df]:
            category_name = next((k for k, v in BeIR.items() if dataset_name in v), None)
            df['category'] = category_name
            df['dataset_name'] = dataset_name
            df['tokenizer'] = model_name

        # Create the output path folder if it does not exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save to CSV
        queries_df.to_csv(queries_output_path, index=False, encoding='utf-8')
        print(f"Saved queries to {queries_output_path}")

        documents_df.to_csv(documents_output_path, index=False, encoding='utf-8')
        print(f"Saved documents to {documents_output_path}")


# Main function
def main():
    parser = argparse.ArgumentParser(description="Process and tokenize datasets.")
    
    parser.add_argument('--dataset_names', nargs='+', required=True, help="List of space separated dataset names to process.")
    parser.add_argument('--num_samples', type=int, required=True, help="Number of samples to load from each dataset.")
    parser.add_argument('--max_document_segment_tokens', type=int, default=2048, required=True, help="Maximum number of tokens per document segment.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to use for tokenization.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output CSV files.")
    parser.add_argument('--force', action='store_true', help="Override existing files for the datasets.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    build_dataset_candidates(
        dataset_names=args.dataset_names,
        num_samples=args.num_samples,
        max_document_segment_tokens=args.max_document_segment_tokens,
        model_name=args.model_name,
        output_path=args.output_path,
        force=args.force,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()

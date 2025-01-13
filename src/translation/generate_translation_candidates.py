from typing import List
import pandas as pd
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from data.beir import build_data

# Function to load and process datasets
def build_dataset_candidates(dataset_names: List[str], 
                             num_samples: int, 
                             max_document_segment_tokens: int, 
                             model_name: str, 
                             output_path: str, 
                             random_seed: int = 42) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Process each sample
        for i, dataset_name in tqdm(enumerate(dataset_names), desc=f"Processing {dataset_name}"):
            data = build_data(dataset_name=dataset_name, 
                              tokenizer=tokenizer, 
                              n=num_samples, 
                              max_tokens=max_document_segment_tokens,
                              random_seed=random_seed)
            queries, documents = data

            # Convert to DataFrames
            queries_df = pd.DataFrame(queries, columns=['query'])
            queries_df['dataset_name'] = dataset_name
            queries_df['tokenizer'] = model_name
            documents_df = pd.DataFrame(documents, columns=['document'])
            documents_df['dataset_name'] = dataset_name
            documents_df['tokenizer'] = model_name
            
            # Save to CSV
            queries_output_path = f"{output_path}/queries_{dataset_name}.csv"
            documents_output_path = f"{output_path}/documents_{dataset_name}.csv"
            
            queries_df.to_csv(queries_output_path, index=False)
            documents_df.to_csv(documents_output_path, index=False)
            
            print(f"Saved queries to {queries_output_path}")
            print(f"Saved documents to {documents_output_path}")


# Main function
def main():
    parser = argparse.ArgumentParser(description="Process and tokenize datasets.")
    
    parser.add_argument('--dataset_names', nargs='+', required=True, help="List of space separated dataset names to process.")
    parser.add_argument('--num_samples', type=int, required=True, help="Number of samples to load from each dataset.")
    parser.add_argument('--max_document_segment_tokens', type=int, required=True, help="Maximum number of tokens per document segment.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to use for tokenization.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output CSV files.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    build_dataset_candidates(
        dataset_names=args.dataset_names,
        num_samples=args.num_samples,
        max_document_segment_tokens=args.max_document_segment_tokens,
        model_name=args.model_name,
        output_path=args.output_path,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()

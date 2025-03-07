import json
import glob
import argparse
import random
from tqdm import tqdm

def build(jsonl_files_path: str, limit: int, output_file: str, random_state: int = 42):
    # List all JSONL files
    jsonl_files = glob.glob(jsonl_files_path)

    samples_per_file = limit // len(jsonl_files)

    random.seed(random_state)

    # Sample from each file and write results
    with open(output_file, "w", encoding="utf-8-sig") as out_f:
        for file in tqdm(jsonl_files, desc="Sampling JSONL files"):
            sampled_lines = reservoir_sample(file, samples_per_file)
            out_f.writelines(sampled_lines)  # Stream writing to avoid memory issues

    sampled_records_count = sum(1 for _ in open(output_file, "r", encoding="utf-8"))
    print(f"Successfully saved {sampled_records_count} sampled records to {output_file}.")


def reservoir_sample(file_path, sample_size):
    """
    Performs reservoir sampling on a large JSONL file without loading it all into memory.
    """
    sampled_lines = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i < sample_size:
                record = json.loads(line)
                sampled_lines.append(record["text"])  # Fill initial sample
            else:
                # Replace with decreasing probability
                j = random.randint(0, i)
                if j < sample_size:
                    record = json.loads(line)
                    sampled_lines[j] = record["text"]  # Replace an existing sample

    return sampled_lines



# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_files_path", type=str, help="Path to JSONL files")
    parser.add_argument("--limit", type=int, help="Limit the number of records to process")
    parser.add_argument("--output_file", type=str, help="Path to output tokenizer corpus file")
    
    args = parser.parse_args()
    
    build(args.jsonl_files_path, args.limit, args.output_file)  # Adjust the path


if __name__ == "__main__":
    main()
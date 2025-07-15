import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from datasets import load_from_disk
import numpy as np
import json
import argparse
from pathlib import Path


def main(model_name_or_path,
         tokenizer_name_or_path,
         documents_path,
         output_file,
         batch_size=32,
         max_length=None,
         text_field = "text"):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
    print(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    print(f"Loading model from {model_name_or_path}")
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading dataset from {documents_path}")
    data_files = {
        "documents": {
            "test": documents_path
        }
    }
    dataset = load_dataset("json", data_files=data_files["documents"], split="test")
    unique_contexts = set()
    deduped_contexts = []
    contexts = [item for item in tqdm(dataset, desc="Loading documents")]
    # Deduplicate contexts
    contexts.sort(key=lambda x: 0 if x["_source"] == "heq" else 1)
    for i, context in tqdm(enumerate(contexts), desc="Deduplicating contexts"):
        if context["guid"] not in unique_contexts:
            unique_contexts.add(context["guid"])
            deduped_contexts.append(context)
    documents = [c[text_field] for c in deduped_contexts]
    print(len(deduped_contexts), "unique contexts found.")

    total_tokens = 0
    total_time = 0.0
    latencies = []

    # Optional: clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for i in tqdm(range(0, len(documents), batch_size), desc="Batch encoding"):
        batch = documents[i:i+batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=(max_length or tokenizer.model_max_length)
        ).to(device)
        input_ids = tokens["input_ids"]
        total_tokens += input_ids.numel()

        tokens = {k: v.to(device) for k, v in tokens.items()}

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**tokens)
            _ = outputs.last_hidden_state[:, 0]  # CLS pooling
        end_time = time.perf_counter()

        duration = end_time - start_time
        total_time += duration
        latencies.extend([duration / len(batch)] * len(batch))

    # Metrics summary
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    throughput = total_tokens / total_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    results = {
        "total_examples": len(documents),
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "avg_latency_ms": avg_latency * 1000,
        "throughput_tokens_per_sec": throughput,
        "peak_memory_mb": peak_mem,
        "latency_p50_ms": p50 * 1000,
        "latency_p95_ms": p95 * 1000,
        "batch_size": batch_size
    }

    # Save evaluation results to json file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run encoding inference on a dataset.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=False, default=None, help="Path to the tokenizer.")
    parser.add_argument("--documents_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the evaluation results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length for tokenization. If None, uses tokenizer's max length.")
    args = parser.parse_args()

    main(model_name_or_path=args.model_name_or_path, 
         tokenizer_name_or_path =args.tokenizer_name_or_path,
         documents_path=args.documents_path,
         output_file=args.output_file,
         batch_size=args.batch_size,
         max_length=args.max_length)

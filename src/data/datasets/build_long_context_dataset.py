import argparse
import os
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer
from copy import deepcopy
from tqdm import tqdm
from data.heq import HeQDatasetBuilder, HeQTranslatedDatasetBuilder
from data.squad_v2 import SquadV2DatasetBuilder
    

def get_distractor_files(all_files, exclude_substrings=("Geektime", "Wikipedia")):
    """
    Given a list of file paths, returns those not containing any substring in exclude_substrings.
    Case-insensitive.
    """
    filtered = [
        path for path in all_files
        if not any(ex.lower() in os.path.basename(path).lower() for ex in exclude_substrings)
    ]
    print(f"Selected {len(filtered)} distractor files (excluded: {exclude_substrings}):")
    for f in filtered:
        print(" -", f)
    return filtered


def reservoir_sample_lines(file_path, num_samples, text_key="text", min_length=50, seed=42):
    """
    Uniformly samples num_samples lines from a large file (JSONL or txt), returns list of sampled paragraphs.
    """
    random.seed(seed)
    reservoir = []
    count = 0

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Use 'text' or 'content' as appropriate
                text = item.get(text_key) or item.get("content") or str(item)
            except Exception:
                text = line
            if len(text) < min_length:
                continue
            count += 1
            if len(reservoir) < num_samples:
                reservoir.append(text)
            else:
                idx = random.randint(0, count - 1)
                if idx < num_samples:
                    reservoir[idx] = text
    print(f"  [{os.path.basename(file_path)}] Sampled {len(reservoir)} from {count} valid lines")
    return reservoir


def build_distractor_pool(
    file_paths,
    num_samples_per_file=5000,
    text_key="text",
    min_length=50,
    max_workers=None,
    seed=42
):
    print(f"\nBuilding distractor pool in parallel from {len(file_paths)} files...")
    total_files = len(file_paths)
    total_seen = 0
    all_samples = []
    
    # Prepare argument tuples for each file
    task_args = [
        (file_path, num_samples_per_file, text_key, min_length, seed + i)
        for i, file_path in enumerate(file_paths)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(reservoir_sample_lines, *args): idx for idx, args in enumerate(task_args)}
        # tqdm with manual update as futures complete
        with tqdm(total=total_files, desc="Files processed", unit="file") as pbar:
            for future in as_completed(futures):
                file_idx = futures[future]
                file_path = file_paths[file_idx]
                try:
                    samples = future.result()
                except Exception as e:
                    print(f"Error in processing {file_path}: {e}")
                    samples = []
                all_samples.extend(samples)
                total_seen += len(samples)
                pbar.update(1)
                print(f"  [{os.path.basename(file_path)}] Added {len(samples)} samples. Total so far: {total_seen}")

    print(f"\nFinished! Total distractors sampled: {len(all_samples)}\n")
    return all_samples


def create_long_document(
    gt_document: str,
    distractor_gen,
    context_window_tokens: int,
    tokenizer,
    gt_location: str = "random",
    output_field: str = "long_document",
    min_num_distractors: int = 2
):
    gt_tokens = tokenizer(gt_document, add_special_tokens=False)["input_ids"]
    gt_len = len(gt_tokens)

    distractor_documents = []
    distractor_tokens_list = []
    total_tokens = gt_len
    distractor_count = 0
    while (total_tokens < context_window_tokens or distractor_count < min_num_distractors):
        distractor = next(distractor_gen)
        if distractor.strip() == gt_document.strip():
            continue
        tokens = tokenizer(distractor, add_special_tokens=False)["input_ids"]
        distractor_documents.append(distractor)
        distractor_tokens_list.append(tokens)
        total_tokens += len(tokens)
        distractor_count += 1

    available_tokens = context_window_tokens - gt_len
    truncated_distractors = []
    truncated_tokens_list = []
    curr_len = 0
    for ctx, toks in zip(distractor_documents, distractor_tokens_list):
        if curr_len + len(toks) > available_tokens:
            toks = toks[:available_tokens - curr_len]
            ctx = tokenizer.decode(toks)
        truncated_distractors.append(ctx)
        truncated_tokens_list.append(toks)
        curr_len += len(toks)
        if curr_len >= available_tokens:
            break

    if gt_location == "start":
        documents = [gt_document] + truncated_distractors
        tokens_list = [gt_tokens] + truncated_tokens_list
        gt_offset = 0
    elif gt_location == "end":
        documents = truncated_distractors + [gt_document]
        tokens_list = truncated_tokens_list + [gt_tokens]
        gt_offset = sum(len(t) for t in truncated_tokens_list)
    elif gt_location == "random":
        idx = random.randint(0, len(truncated_distractors))
        documents = truncated_distractors[:idx] + [gt_document] + truncated_distractors[idx:]
        tokens_list = truncated_tokens_list[:idx] + [gt_tokens] + truncated_tokens_list[idx:]
        gt_offset = sum(len(t) for t in tokens_list[:idx])
    else:
        raise ValueError(f"Invalid gt_location: {gt_location}")

    full_document = "".join(documents)
    full_tokens = sum(tokens_list, [])
    full_tokens = full_tokens[:context_window_tokens]
    full_document = tokenizer.decode(full_tokens)
    gt_token_start = gt_offset
    gt_token_end = min(gt_offset + gt_len, context_window_tokens)

    return {
        output_field: full_document,
        "gt_token_start": gt_token_start,
        "gt_token_end": gt_token_end,
        "gt_location": gt_location
    }


def distractor_gen_from_pool(pool):
    # Infinite generator from a list
    while True:
        yield random.choice(pool)


def main(
    dataset,
    distractor_source_folder: str,
    context_window_tokens,
    tokenizer,
    gt_location="random",
    document_field="document",
    output_field="long_document",
    random_state=42
):
    # Seed for reproducibility
    random.seed(random_state)

    # Load all distractor files from the specified folder
    if not os.path.exists(distractor_source_folder):
        raise FileNotFoundError(f"Distractor source folder does not exist: {distractor_source_folder}")
    all_distractors_sources = [
        os.path.join(distractor_source_folder, fname)
        for fname in os.listdir(distractor_source_folder)
        if fname.endswith(".jsonl")
    ]
    # Build the distractor pool
    distractor_files = get_distractor_files(all_distractors_sources, exclude_substrings=["Geektime", "Wikipedia", "Oscar"])
    distractor_pool = build_distractor_pool(
        distractor_files,
        num_samples_per_file=5000,   # or higher/lower based on RAM/dataset size
        text_key="text",
        min_length=50,
        seed=random_state
    )
    distractor_gen = distractor_gen_from_pool(distractor_pool)

    # Ensure the tokenizer does not warn on hardcoded large context window size
    tokenizer.model_max_length = context_window_tokens

    # Create a new dataset with long documents, showing progress
    new_dataset = deepcopy(dataset)
    for i, record in enumerate(tqdm(new_dataset, desc="Building long document dataset", unit="record")):
        result = create_long_document(
            gt_document=record[document_field],
            distractor_gen=distractor_gen,
            context_window_tokens=context_window_tokens,
            tokenizer=tokenizer,
            gt_location=gt_location,
            output_field=output_field,
        )
        new_record = {
            **record,
            **result
        }
        new_dataset[i] = new_record

    return new_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset with long document windows for retrieval evaluation.")
    parser.add_argument("--documents_file_path", required=True, help="Path to the document file (JSONL format).")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer model name or path.")
    parser.add_argument("--context-window", type=int, default=8192, help="Document window size in tokens.")
    parser.add_argument("--gt_location", choices=["start", "end", "random"], default="random", help="Where to place GT document.")
    parser.add_argument("--document_field", default="document", help="Field name for the ground-truth document.")
    parser.add_argument("--output_field", default="long_document", help="Field name for the long document output.")
    parser.add_argument("--distractor_source_folder", required=True, help="Path to a folder containing distractor files (JSONL format).")
    parser.add_argument("--encoding", default="utf-8", help="Encoding for input/output files.")

    args = parser.parse_args()

    # Load document jsonl file
    if not os.path.exists(args.documents_file_path):
        raise FileNotFoundError(f"Document file does not exist: {args.documents_file_path}")
    with open(args.documents_file_path, "r", encoding=args.encoding) as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Augment dataset
    long_document_dataset = main(
        dataset,
        distractor_source_folder=args.distractor_source_folder,
        context_window_tokens=args.context_window,
        tokenizer=tokenizer,
        gt_location=args.gt_location,
        document_field=args.document_field,
        output_field=args.output_field
    )

    # Save to output JSONL
    with open(args.output, "w", encoding="utf-8") as out_f:
        for record in long_document_dataset:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(long_document_dataset)} records to {args.output}")

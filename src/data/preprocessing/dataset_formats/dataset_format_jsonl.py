from typing import List, Optional
import os
import random
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm

class DatasetFormatJSONL:
    def __init__(self,):
        self.name = "jsonl"

    def stream(jsonl_files: List[str],
               exclude_jsonl_files: Optional[List[str]] = None, 
               split: str = "train",
               split_ratio: float = 0.1, 
               random_state: int = 42):
        """ 
        Generator to stream JSONL rows as either train or validation split.
        
        Parameters:
            jsonl_files (list): List of JSONL file paths.
            exclude_jsonl_files (list): List of JSONL file paths to exclude.
            split (str): Split type, either "train" or "validation".
            split_ratio (float): Proportion of samples to use for validation.
            random_state (int): Random seed for reproducibility.
        
        Yields:
            dict: {"text": ..., "_source": ..., "_row_number": ...}
        """
        assert split in {"train", "validation"}, "split must be 'train' or 'validation'"
        
        random.seed(random_state)
        
        # Get list of JSONL files
        if len(jsonl_files) == 1 and os.path.isdir(jsonl_files[0]):
            jsonl_files_path = jsonl_files[0]
            jsonl_files = [str(f) for f in list(Path(jsonl_files_path).glob("*.jsonl"))]

        # Exclude files if needed
        if exclude_jsonl_files:
            jsonl_files = [file for file in jsonl_files if file not in exclude_jsonl_files]
        
        for file_path in tqdm(jsonl_files, desc="Reading JSONL files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                samples_count = sum(1 for _ in f)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                validation_size = int(split_ratio * samples_count)
                # Randomly select validation indexes
                validation_indexes = random.sample(range(0, samples_count), validation_size)
                validation_indexes_stack = sorted(validation_indexes)

                for idx, line in tqdm(enumerate(f), desc=f"Reading {file_path}"):
                    # Check if the current index is in the validation indexes
                    index_in_validation_indexes = validation_indexes_stack and idx == validation_indexes_stack[0]
                    if index_in_validation_indexes and split != "validation":
                        continue # Skip this record for training
                    elif index_in_validation_indexes:
                        # Pop the index from the validation indexes stack
                        validation_indexes_stack.pop(0)

                    try:
                        data = json.loads(line)
                        record = {
                            "text": data["text"],
                            "_source": os.path.basename(file_path),
                            "_row_number": idx
                        }   
                        yield record
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line {idx} in {file_path}")



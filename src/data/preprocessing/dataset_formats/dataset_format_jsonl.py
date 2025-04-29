from typing import List, Optional
import os
import random
import json
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, DatasetDict
from tqdm import tqdm

class DatasetFormatJSONL:
    def __init__(self, name: str):
        self.name = name


    def stream(self, 
        jsonl_files: List[str],
        exclude_jsonl_files: Optional[List[str]] = None, 
        split: str = "train",
        split_ratio: float = 0.1,
        text_field: str = "text",
        limit: int  = 0,
        encoding: str = "utf-8-sig",
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
        print("***** Loading JSONL files...")

        # Expand directories to file list
        json_files_orig = jsonl_files.copy()
        for json_file in json_files_orig:
            print(f"Checking {json_file}...")
            if os.path.isdir(json_file):
                jsonl_files.remove(json_file)
                jsonl_files += [str(f) for f in Path(json_file).glob("*.jsonl")]

        # Exclude files if needed
        if exclude_jsonl_files:
            jsonl_files = [file for file in jsonl_files if file not in exclude_jsonl_files]

        total_yielded = 0

        for file_path in tqdm(jsonl_files, desc="Reading JSONL files..."):
            with open(file_path, 'r', encoding=encoding) as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line {idx} in {file_path}")
                        continue

                    # Decide split randomly
                    is_validation = random.random() < split_ratio
                    if (split == "train" and is_validation) or (split == "validation" and not is_validation):
                        continue

                    record = {
                        "text": data[text_field],
                        "_source": os.path.basename(file_path),
                        "_row_number": idx
                    }

                    yield record

                    total_yielded += 1
                    if limit > 0 and total_yielded >= limit:
                        return  # Stop streaming after reaching the limit


    def stream_(self, 
               jsonl_files: List[str],
               exclude_jsonl_files: Optional[List[str]] = None, 
               split: str = "train",
               split_ratio: float = 0.1,
               text_field: str = "text",
               limit: int  = 0,
               encoding: str = "utf-8-sig",
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

        samples_per_file = limit // len(jsonl_files)

        # Exclude files if needed
        if exclude_jsonl_files:
            jsonl_files = [file for file in jsonl_files if file not in exclude_jsonl_files]
        
        # Iterate over JSONL files and sample records
        for file_path in tqdm(jsonl_files, desc="Reading JSONL files"):
            # Get file lines size for splitting 
            with open(file_path, 'r', encoding=encoding) as f:
                samples_count = sum(1 for _ in f)
            # Reset file pointer
            with open(file_path, 'r', encoding=encoding) as f:
                samples = []

                # Train/validation splits
                validation_size = int(split_ratio * samples_count)
                validation_indexes = random.sample(range(0, samples_count), validation_size)
                validation_indexes_stack = sorted(validation_indexes)
                split_indexes = []
                for idx, line in tqdm(enumerate(f), desc=f"Reading {file_path}"):
                    # Check if the current index is in the validation indexes
                    index_in_validation_indexes = validation_indexes_stack and idx == validation_indexes_stack[0]
                    if index_in_validation_indexes and split != "validation":
                        continue # Skip this record for training
                    elif index_in_validation_indexes:
                        # Pop the index from the validation indexes stack
                        validation_indexes_stack.pop(0)
                    # Keep track of the split indexes
                    split_indexes.append(idx)
                    
                    if samples_per_file == 0:
                        # Do not sample
                        samples.append(line)
                    else:
                        # Sample from each file and yield results
                        if len(split_indexes) < samples_per_file:
                            record = json.loads(line)
                            samples.append(record)  # Fill initial sample
                        else:
                            # Replace with decreasing probability
                            j = random.randint(0, len(split_indexes))
                            if j < samples_per_file:
                                record = json.loads(line)
                                samples[j] = record  # Replace an existing sample
                
                # Yield sampled records
                for idx, line in enumerate(samples):
                    try:
                        data = json.loads(line)
                        record = {
                            "text": data[text_field],
                            "_source": os.path.basename(file_path),
                            "_row_number": idx
                        }   
                        yield record
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line {idx} in {file_path}")
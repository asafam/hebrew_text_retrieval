from typing import List, Optional
import os
import random
import json
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from data.datasets.utils import hash_text

class DatasetFormatJSONL:
    def __init__(self, name: str):
        self.name = name


    def stream(
            self, 
            jsonl_files: List[str],
            exclude_jsonl_files: Optional[List[str]] = None, 
            split: str = "train",
            split_ratio: float = 0.1,
            text_field: str = "text",
            guid_field: Optional[str] = None,
            limit: int  = 0,
            encoding: str = "utf-8-sig",
            random_state: int = 42
        ):
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
        assert split in {"train", "validation", "test"}, "split must be 'train', 'test', or 'validation'"
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
                        **{key.lower(): value for key, value in data.items()},
                        "guid": data[guid_field] if guid_field else hash_text(data[text_field]),
                        "text": data[text_field],
                        "_source": data["_source"] if "_source" in data else f"{self.name}_{os.path.basename(file_path)}",
                        "_file": os.path.basename(file_path),
                        "_row_number": idx,
                    }

                    yield record

                    total_yielded += 1
                    if limit > 0 and total_yielded >= limit:
                        return  # Stop streaming after reaching the limit


    
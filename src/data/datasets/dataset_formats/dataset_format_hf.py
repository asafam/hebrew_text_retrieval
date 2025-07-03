from typing import Optional
import random
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from data.datasets.utils import tokenize, hash_text

dataset_to_text_field = {
    "allenai/c4": "text",
}

class DatasetFormatHF:
    def __init__(self,
                 dataset_name: str):
        self.name = dataset_name

    def stream(self,
               hf_dataset_args: dict,
               text_field: str = "text", 
               filter_criteria: list = [],
               split: str = "train",
               split_ratio: float = 0.1,
               limit: int = 0,
               tokens_limit: int = 0,
               shuffle: bool = True,
               remove_duplicates: bool = False,
               random_state: int = 42):
        """
        Generator to stream datasets as either train or validation split.
        Parameters:
            split (str): Split type, either "train" or "validation".
            text_field (str): Field name for the text data.
        Yields:
            dict: {"text": ..., "_source": ..., "_row_number": ...}
        """
        # Load the dataset
        dataset = load_dataset(**hf_dataset_args)

        # Train/validation splits
        random.seed(random_state)
        dataset_size = sum(1 for _ in dataset)
        validation_size = int(split_ratio * dataset_size)
        validation_indexes = random.sample(range(0, dataset_size), validation_size)
        indexes = validation_indexes.copy()
        
        if split == "train":
            train_indexes = [i for i in range(dataset_size) if i not in validation_indexes]
            indexes = train_indexes

        sorted_indices = sorted(indexes)
        dataset = dataset.select(sorted_indices) # Select the rows corresponding to the sorted indices

        # Check if the dataset is in the dataset_to_text_field mapping
        if filter_criteria:
            filtered_datasets = []
            for criteria in filter_criteria:
                filtered = dataset.filter(
                    lambda example: all(example.get(field) == value for field, value in criteria.items())
                )
                filtered_datasets.append(filtered)
            dataset = concatenate_datasets(filtered_datasets)

        # Shuffle the dataset if required
        if shuffle:
            dataset = dataset.shuffle(seed=random_state)

        # Limit the number of records if specified
        if limit > 0:
            dataset_size = sum(1 for _ in dataset)
            dataset = dataset.select(range(min(limit, dataset_size)))
        
        # Limit the number of tokens if specified
        if tokens_limit > 0:
            total_tokens = 0
            end_tokens_index = 0
            for i, record in enumerate(dataset):
                text = record[text_field]
                tokens = tokenize(text)
                token_count = len(tokens)
                
                # Check if adding this record exceeds the token limit
                if total_tokens + token_count > tokens_limit:
                    end_tokens_index = i
                    break
                
            dataset = dataset.select(range(end_tokens_index)) if end_tokens_index > 0 else dataset

        if remove_duplicates:
            dataset = dataset.unique(text_field)
            
        for idx, record in enumerate(dataset):
            # Create a record with the required fields
            record = {
                **{key.lower(): value for key, value in record.items()},
                "guid": hash_text(record[text_field]),
                "text": record[text_field],
                "_source": self.name,
                "_row_number": idx,
            }
            yield record
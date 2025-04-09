from typing import Optional
from datasets import load_dataset, DatasetDict

dataset_to_text_field = {
    "allenai/c4": "text",
}

class DatasetFormatHF:
    def __init__(self,
                 dataset_name: str):
        self.name = dataset_name

    def stream(self, split: str, text_field: str = "text", **kwargs):
        """
        Generator to stream datasets as either train or validation split.
        Parameters:
            split (str): Split type, either "train" or "validation".
            text_field (str): Field name for the text data.
        Yields:
            dict: {"text": ..., "_source": ..., "_row_number": ...}
        """
        # Load the dataset
        dataset = load_dataset(**kwargs, split=split)
        for idx, record in enumerate(dataset):
            record = {
                "text": record[text_field],
                "_source": self.dataset_name,
                "_row_number": idx
            }
            yield record
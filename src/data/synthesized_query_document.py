from datasets import Dataset, DatasetDict
from tqdm import tqdm
from pathlib import Path
import pickle
import logging
from data import *

class SynthesizedQueryDocumentDatasetBuilder(BaseDatasetBuilder):
    def build_dataset(
            self, 
            data_folder_path='data/synthetic_data_202409', 
            splits=['train', 'validation'], 
            test_size=0.2,
            include_task_token: bool = True
        ):
        logger = logging.getLogger('default')
        logger.info("Transforming synthesized dataset")

        logger.info("Loading synthesize query document dataset from {data_folder_path}")
        data = self._load_synthesized_data_files(data_folder_path=os.path.join(os.getcwd(), data_folder_path))

        def transform_entry(entry):
            # Return the transformed data
            return {
                'anchor_text': f"{TASK_TOKENS[TASK.QUERY_PASSAGE]} {QUERY_TOKEN} {entry['user_query']}" if include_task_token else f"{QUERY_TOKEN} {entry['user_query']}",
                'positive_text': f"{DOCUMENT_TOKEN} {entry['positive_document']}",
                'negative_text': f"{DOCUMENT_TOKEN} {entry['hard_negative_document']}",
            }

        # Apply the transformation to each entry
        transformed_data = list(map(transform_entry, data))

        # Convert the list of dictionaries to a Hugging Face Dataset
        all_dataset = Dataset.from_list(transformed_data)

        # Split the dataset into train and test sets
        train_test_dataset = all_dataset.train_test_split(test_size=test_size)
        # Further split the test set into validation and test
        test_validation_dataset = train_test_dataset['test'].train_test_split(test_size=0.5)  # 50% of 'test' for 'validation'
        train_validation_dataset = DatasetDict({
            'train': train_test_dataset['train'],  # Keep the 'train' split
            'test': test_validation_dataset['test'],  # The remaining 'test' set
            'validation': test_validation_dataset['train'],  # The remaining 'test' set
        })

        dataset = DatasetDict({split: train_validation_dataset[split] for split in splits})
        return dataset
    
    def build_eval_dataset(self, split='test', random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building Synthesized Query Document evaluation dataset")

        tasks_datasets = {
            'TASK_QUERY_DOC': self.build_dataset(splits=[split], include_task_token=False)[split]
        }
        return tasks_datasets
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name == DatasetName.SYNTHESIZED_QUERY_DOCUMENT.value
    
    def _load_synthesized_data_files(self, data_folder_path):
        logger = logging.getLogger('default')
        data = []

        # Method to use pathlib to find all .pkl files
        data_folder = Path(data_folder_path)
        files = [file_path for file_path in data_folder.glob('*.pkl')]

        # Sort the files by name (optional, if you want a specific order)
        files.sort(key=lambda x: x.name)

        # Load and concatenate data from each file
        for file_path in tqdm(files, desc="Loading data files"):
            logger.debug(f"Loading data from {file_path.name}")
            with file_path.open('rb') as f:
                file_data = pickle.load(f)
                if isinstance(data, list):
                    data.extend(file_data)
                else:
                    logger.debug(f"Warning: {file_path.name} does not contain a list.")

        data = [item for item in data if item['success']]
        
        return data



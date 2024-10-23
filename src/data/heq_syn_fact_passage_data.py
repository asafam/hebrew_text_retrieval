from datasets import load_dataset
import logging
from data import *

class HeQDatasetBuilder(BaseDatasetBuilder):
    def build_dataset(self, splits=['train', 'validation']):
        logger = logging.getLogger('default')
        logger.info("Building HeQ Synthesized Fact-Passage dataset")

        # Load the dataset from Hugging Face Datasets
        dataset = load_dataset("asafam/heq_syn_fact_passage", split=splits)

        # Apply the transformation to the train, validation, and test splits
        transformed_dataset = {}
        for split in splits:
            # Transform each subset of the dataset using map (this processes each 'text' entry)
            logger.info(f"Transforming {split} split")
            transformed_split = dataset[split].map(self._transform_entry)
            transformed_dataset[split] = transformed_split

        return transformed_dataset
    
    def build_eval_dataset(self, split='test'):
        logger = logging.getLogger('default')
        logger.info("Building HeQ evaluation dataset")

        dataset = load_dataset("asafam/heq_syn_fact_passage", split=[split])
        transformed_split = dataset[split].map(self._transform_entry)
        return transformed_split
        
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name == DatasetName.HEQ_SYN_FACT_PASSAGE.value
    
    def _transform_entry(self, entry, include_task_token: bool = True):
        # Return the transformed data
        return {
            'anchor_text': f"{TASK_TOKENS[TASK.FACT_PASSAGE]} {QUERY_TOKEN} {entry['fact']}" if include_task_token else f"{QUERY_TOKEN} {entry['fact']}",
            'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}",
        }

    
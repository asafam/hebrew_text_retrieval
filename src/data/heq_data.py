import requests
import json
import random
from datasets import Dataset, DatasetDict
import logging
from data import *

class HeQDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, data_files_url_base_path: str = 'https://raw.githubusercontent.com/NNLP-IL/Hebrew-Question-Answering-Dataset/main/data') -> None:
        self.data_files_url_base_path = data_files_url_base_path

    def build_dataset(self, splits=['train', 'val'], random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building HeQ dataset")

        datasets = {}
        for split in splits:
            url = f"{self.data_files_url_base_path}/{split}.json"
            print(f"url = {url}")
            data = self._load_json_from_github(url=url)
            datasets[split] = self._transform_data(data, random_seed=random_seed)

        return DatasetDict(datasets)
    
    def build_eval_dataset(self, split='test', random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building HeQ evaluation dataset")

        data = self._load_json_from_github(f"self.data_files_url_base_path/{split}.json")
        tasks_datasets = self._transform_eval_data(data, random_seed=random_seed)

        return tasks_datasets
        
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name == DatasetName.HEQ.value

    def _load_json_from_github(self, url):
        logger = logging.getLogger('default')
        logger.info(f"Loading json file from {url}")

        response = requests.get(url)
        if response.status_code == 200:
            content = json.loads(response.content)
            return content['data']
        else:
            print(f"Failed to fetch data from {url}")
            return None
        
    def _transform_data(self, data, random_seed: int = 42):
        random.seed(random_seed)

        transformed_heq_data = []
        for sublist in list(map(self._transform_heq_entry, data)):
            transformed_heq_data += sublist

        def transform_entry(entry):
            return {
                'anchor_text': f"{TASK_TOKENS['TASK_QUESTION_DOC']} {QUERY_TOKEN} {entry['question']}",
                'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}"
            }

        transformed_dataset_data = list(map(transform_entry, transformed_heq_data))
        dataset =  Dataset.from_list(transformed_dataset_data)
        return dataset
    
    def _transform_eval_data(self, data, random_seed: int = 42):
        random.seed(random_seed)

        transformed_heq_data = list(map(self._transform_heq_entry, data))

        def transform_query_document_entry(entry):
            return {
                'query': f"{QUERY_TOKEN} {entry['question']}",
                'document': f"{DOCUMENT_TOKEN} {entry['context']}"
            }
        
        def transform_title_document_entry(entry):
            return {
                'query': f"{QUERY_TOKEN} {entry['title']}",
                'document': f"{DOCUMENT_TOKEN} {entry['context']}"
            }

        tasks_datasets = {}

        transformed_dataset_query_doc_data = list(map(transform_query_document_entry), transformed_heq_data)
        tasks_datasets[TASK_TOKENS['TASK_QUERY_DOC']] =  Dataset.from_list(transformed_dataset_query_doc_data)

        transformed_dataset_query_doc_data = list(map(transform_title_document_entry), transformed_heq_data)
        tasks_datasets[TASK_TOKENS['TASK_TITLE_DOC']] =  Dataset.from_list(transformed_dataset_query_doc_data)

        return tasks_datasets

    def _transform_heq_entry(self, entry):
        items = []
        for paragraph in entry['paragraphs']:
            item = {
                'title': entry['title'],
                'question': random.choice([x['question'] for x in paragraph['qas']]),
                'context': paragraph['context']
            }
            items.append(item)
        return items
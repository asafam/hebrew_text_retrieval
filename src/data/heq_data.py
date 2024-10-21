import requests
import json
import random
from datasets import Dataset, DatasetDict
import logging
from data import *

class HeQDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, data_files_url_base_path: str = 'https://raw.githubusercontent.com/NNLP-IL/Hebrew-Question-Answering-Dataset/main/data', **kwargs) -> None:
        self.data_files_url_base_path = data_files_url_base_path

    def build_dataset(self, splits=['train', 'validation'], random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building HeQ dataset")

        split_file_mapping = {
            'validation': 'val',
        }

        datasets = {}
        for split in splits:
            file_split = split_file_mapping.get(split, split)
            url = f"{self.data_files_url_base_path}/{file_split}.json"
            print(f"url = {url}")
            data = self._load_json_from_github(url=url)
            datasets[split] = self._transform_data(data, random_seed=random_seed)

        return DatasetDict(datasets)
    
    def build_eval_dataset(self, split='test', random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building HeQ evaluation dataset")

        split_file_mapping = {
            'validation': 'val',
        }
        file_split = split_file_mapping.get(split, split)
        data = self._load_json_from_github(f"{self.data_files_url_base_path}/{file_split}.json")
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
                'anchor_text': f"{TASK_TOKENS[TASK.QUESTION_PASSAGE]} {QUERY_TOKEN} {entry['question']}",
                'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}",
                'question_id': entry['question_id'],
                'question': entry['question'],
                'answer': entry['answer'],
                'context': entry['context'],
            }

        transformed_dataset_data = list(map(transform_entry, transformed_heq_data))
        dataset =  Dataset.from_list(transformed_dataset_data)
        return dataset
    
    def _transform_eval_data(self, data, random_seed: int = 42):
        random.seed(random_seed)

        transformed_heq_data = []
        for sublist in list(map(self._transform_heq_entry, data)):
            transformed_heq_data += sublist

        def transform_question_document_entry(entry):
            return {
                'anchor_text': f"{QUERY_TOKEN} {entry['question']}",
                'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}"
            }
        
        def transform_title_document_entry(entry):
            return {
                'anchor_text': f"{QUERY_TOKEN} {entry['title']}",
                'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}"
            }

        tasks_datasets = {}

        transformed_dataset_query_doc_data = list(map(transform_question_document_entry, transformed_heq_data))
        tasks_datasets['TASK_QUESTION_DOC'] =  Dataset.from_list(transformed_dataset_query_doc_data)

        transformed_dataset_query_doc_data = list(map(transform_title_document_entry, transformed_heq_data))
        tasks_datasets['TASK_TITLE_DOC'] =  Dataset.from_list(transformed_dataset_query_doc_data)

        return tasks_datasets

    def _transform_heq_entry(self, entry):
        items = []
        for paragraph in entry['paragraphs']:
            rand_idx = random.randint(0, len(paragraph['qas']) - 1)
            item = {
                'title': entry['title'],
                'question_id': paragraph['qas'][rand_idx]['id'],
                'question': paragraph['qas'][rand_idx]['question'],
                'answer': paragraph['qas'][rand_idx]['answers'][0]['text'],
                'context': paragraph['context']
            }
            items.append(item)
        return items
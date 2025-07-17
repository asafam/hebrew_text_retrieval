from typing import Optional
import requests
import json
import random
from datasets import Dataset, DatasetDict
import logging
from data import *
from enum import Enum

class HeQTaskName(Enum):
    QUESTION_DOC = 'TASK_QUESTION_DOC'
    TITLE_DOC = 'TASK_TITLE_DOC'

class HeQDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, 
                 task: Optional[HeQTaskName] = None,
                 data_files_url_base_path: str = 'https://raw.githubusercontent.com/NNLP-IL/Hebrew-Question-Answering-Dataset/refs/heads/main/data/data%20v1.1',
                 **kwargs) -> None:
        self.task = task
        self.data_files_url_base_path = data_files_url_base_path
        super().__init__(**kwargs)

    def build_dataset(self, splits=['train', 'test', 'validation'], should_sample: bool = True, filter_empty_answers: bool = False, random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building HeQ dataset")

        split_file_mapping = {
            'validation': 'val',
        }

        datasets = {}
        for split in splits:
            file_split = split_file_mapping.get(split, split)
            url = f"{self.data_files_url_base_path}/{file_split}%20v1.1.json"
            print(f"url = {url}")
            data = self._load_json_from_github(url=url)
            datasets[split] = self._transform_data(data, should_sample=should_sample, filter_empty_answers=filter_empty_answers, random_seed=random_seed)

        return DatasetDict(datasets)
    
    def build_eval_dataset(self, split='test', should_sample: bool = True, filter_empty_answers: bool = False, random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building HeQ evaluation dataset")

        split_file_mapping = {
            'validation': 'val',
        }
        file_split = split_file_mapping.get(split, split)
        data = self._load_json_from_github(f"{self.data_files_url_base_path}/{file_split}.json")
        tasks_datasets = self._transform_eval_data(data, should_sample=should_sample, filter_empty_answers=filter_empty_answers, random_seed=random_seed)

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
            raise ValueError(f"Failed to fetch data from {url}")
        
    def _transform_data(self, data, should_sample: bool = True, filter_empty_answers: bool = True, random_seed: int = 42):
        random.seed(random_seed)

        transformed_heq_data = []
        for sublist in [self._transform_heq_entry(entry, index, should_sample) for index, entry in enumerate(data)]:
            transformed_heq_data += sublist

        def transform_entry(entry):
            transformed_entry = {
                'anchor_text': entry['question'],
                'positive_text': entry['context'],
                'index': entry['index'],
                'paragraph_index': entry['paragraph_index'],
                'question_id': entry['question_id'],
                'question': entry['question'],
                'answer': entry['answer'],
                'context': entry['context'],
            }

            if self.decorate_with_task_tokens:
                transformed_entry['anchor_text'] = f"{TASK_TOKENS[TASK.QUESTION_PASSAGE]} {QUERY_TOKEN} {transformed_entry['anchor_text']}"
                transformed_entry['positive_text'] = f"{DOCUMENT_TOKEN} {transformed_entry['positive_text']}"

            return transformed_entry

        transformed_dataset_data = list(map(transform_entry, transformed_heq_data))
        dataset =  Dataset.from_list(transformed_dataset_data)
        if filter_empty_answers:
            dataset = dataset.filter(lambda example: example['answer'] is not None)
        return dataset
    
    def _transform_eval_data(self, data, should_sample: bool = True, filter_empty_answers: bool = True, random_seed: int = 42):
        random.seed(random_seed)

        transformed_heq_data = []
        for sublist in [self._transform_heq_entry(entry, index, should_sample) for index, entry in enumerate(data)]:
            transformed_heq_data += sublist
        
        if filter_empty_answers:
            transformed_heq_data = [example for example in transformed_heq_data if example['answer'] is not None]
        tasks_datasets = {}

        transformed_dataset_query_doc_data = list(map(self._transform_question_document_entry, transformed_heq_data))
        tasks_datasets['TASK_QUESTION_DOC'] =  Dataset.from_list(transformed_dataset_query_doc_data)

        transformed_dataset_query_doc_data = list(map(self._transform_title_document_entry, transformed_heq_data))
        tasks_datasets['TASK_TITLE_DOC'] =  Dataset.from_list(transformed_dataset_query_doc_data)

        if self.task == HeQTaskName.QUESTION_DOC:
            return tasks_datasets['TASK_QUESTION_DOC']
        elif self.task == HeQTaskName.TITLE_DOC:
            return tasks_datasets['TASK_TITLE_DOC']
        else:
            return tasks_datasets
    

    def _transform_question_document_entry(self, entry):
            return {
                'anchor_text': f"{QUERY_TOKEN} {entry['question']}" if self.decorate_with_task_tokens else entry['question'],
                'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}" if self.decorate_with_task_tokens else entry['context'],
            }
        
    def _transform_title_document_entry(self, entry):
        return {
            'anchor_text': f"{QUERY_TOKEN} {entry['title']}" if self.decorate_with_task_tokens else entry['title'],
            'positive_text': f"{DOCUMENT_TOKEN} {entry['context']}" if self.decorate_with_task_tokens else entry['context'],
        }

    def _transform_heq_entry(self, entry, index, should_sample: bool = True):
        items = []
        for j, paragraph in enumerate(entry['paragraphs']):
            if should_sample:
                rand_idx = random.randint(0, len(paragraph['qas']) - 1)
                try:
                    item = {
                        'index': index,
                        'paragraph_index': j,
                        'title': entry['title'],
                        'question_id': paragraph['qas'][rand_idx]['id'],
                        'question': paragraph['qas'][rand_idx]['question'],
                        'answer': paragraph['qas'][rand_idx]['answers'][0]['text'] if len(paragraph['qas'][rand_idx]['answers']) else None,
                        'context': paragraph['context']
                    }
                    items.append(item)
                except IndexError:
                    print(f"IndexError for entry {index}, paragraph {j}, question index {rand_idx}. Skipping this question.")
                    continue
            else:
                for k, qa in enumerate(paragraph['qas']):
                    item = {
                        'index': index,
                        'paragraph_index': j,
                        'title': entry['title'],
                        'question_id': qa['id'],
                        'question': qa['question'],
                        'answer': qa['answers'][0]['text'],
                        'context': paragraph['context']
                    }
                    items.append(item)
        return items
    

class HeQTranslatedDatasetBuilder(BaseDatasetBuilder):
    def __init__(self,
                 queries_base_path: str,
                 query_field: str = 'question_hebrew',
                 document_field: str = 'context_hebrew',
                 **kwargs):
        super().__init__(**kwargs)
        self.queries_base_path = queries_base_path
        self.query_field = query_field
        self.document_field = document_field

    def build_dataset(self, splits=['train', 'test', 'validation'], random_state: int = 42, **kwargs):
        datasets = {}
        for split in splits:
            # Validate the split
            if split not in ['train', 'test', 'validation']:
                raise ValueError(f"Unsupported split: {split}. Supported splits are: 'train', 'test', 'validation'.")
            
            # Construct the path to the queries file
            queries_file = os.path.join(self.queries_base_path, 
                                        "validation" if split == "test" else split, # Handle 'test' as 'validation'
                                        'queries.jsonl')
            print(f"Loading {split} queries from: {queries_file}")
            queries = Dataset.from_json(queries_file)

            datasets[split] = Dataset.from_dict({
                self.query_field: [q[self.query_field] for q in queries],
                self.document_field: [q[self.document_field] for q in queries],
            })
                

        return DatasetDict(datasets)
    
    def build_eval_dataset(self, split='test', **kwargs):
        return self.build_dataset(splits=[split], **kwargs)
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.lower() == 'heq_translated'
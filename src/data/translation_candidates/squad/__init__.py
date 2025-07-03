import os
from datasets import load_dataset
import tiktoken
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import random
import importlib
import inspect
from translation.utils import split_document_by_segments
from data.translation_candidates import TranslationCandidatesDataBuilder

class SQuADDataBuilder(TranslationCandidatesDataBuilder):
    def build_data(self, 
                   dataset_name: str, 
                   model_name_or_path: str, 
                   tokenizer_name_or_path: str,
                   n: int = 0, 
                   split: str = 'test',
                   id_field: str = 'id',
                   query_field: str = 'question',
                   context_field: str = 'context',
                   max_tokens: int = 2048, 
                   random_state: int = 42):
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.map(self._compute_guid)

        # Documents
        guid_to_example = {}
        for example in dataset:
            guid = example["guid"]
            if guid not in guid_to_example:
                guid_to_example[guid] = example
        unique_documents = [{
            id_field: example[id_field], 
            "guid": guid, 
            context_field: example[context_field]
        } for guid, example in guid_to_example.items()]

        # Sample documents
        if 0 < n < len(unique_documents):
            random.seed(random_state)
            documents = random.sample(unique_documents, n)
        else:
            documents = unique_documents
        
        # Break down documents to segments
        segmented_documents = []
        for document in documents:
            segmented_documents.extend(self._split_document_by_segments(document, model_name_or_path, max_tokens=max_tokens, context_field=context_field))
        documents = segmented_documents

        # Queries
        id_to_example = {example[id_field]: example for example in dataset}
        if 0 < n < len(dataset):
            queries = [{
                id_field: example[id_field], 
                query_field: example[query_field], 
                "context_id": example[context_field],
                "context_text": example[context_field],
            } for _, example in guid_to_example.items()]
        else:
            queries = [{
                id_field: example[id_field], 
                query_field: example[query_field], 
                "context_id": example[context_field],
                "context_text": example[context_field],
            } for example in dataset]
        
        return queries, documents
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.lower() in ["rajpurkar/squad_v2"]
    
    def _compute_guid(self, example):
        context = example["context"]
        guid = hash(context)
        example["guid"] = guid
        return example

    def _get_document_tokens(self, document: dict):
        return self.tokenizer(document['text'])["input_ids"]


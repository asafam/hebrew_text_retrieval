import os
from collections import defaultdict
from datasets import load_dataset
import random
from translation.utils import split_document_by_segments
from data.translation_candidates import TranslationCandidatesDataBuilder
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from translation.utils import hash_text
from data.heq import HeQDatasetBuilder


class HeQDataBuilder(TranslationCandidatesDataBuilder):
    def build_data(self, 
                   dataset_name: str, 
                   model_name_or_path: str, 
                   tokenizer_name_or_path: str,
                   n: int = 0, 
                   split: str = 'test',
                   id_field: str = 'question_id',
                   query_field: str = 'question',
                   context_field: str = 'context',
                   max_tokens: int = 2048, 
                   random_state: int = 42):
        print(f"Building data for dataset: {dataset_name}, split: {split}, n: {n}, max_tokens: {max_tokens}")
        dataset_builder = HeQDatasetBuilder()
        datasets = dataset_builder.build_dataset(filter_empty_answers=True, splits=[split])
        dataset = datasets[split]
        dataset = dataset.map(lambda example: self._hash_text(example, context_field, hash_field="context_hash"),)

        context_hash_to_example = defaultdict(list)
        for example in dataset:
            context_hash = example["context_hash"]
            context_hash_to_example[context_hash].append(example)

        print(f"Processing queries and documentgs from dataset: {dataset_name}\n")
        queries = []
        documents = []
        random.seed(random_state)
        for context_hash, examples in context_hash_to_example.items():
            # Queries
            if 0 < n < len(dataset):
                # Sample queries
                example = random.choice(examples)
                queries.append({
                    **example,
                    "id": example[id_field], 
                    "text": example[query_field], 
                    "context_id": context_hash,
                    "context_text": example[context_field],
                })
            else:
                # Use all queries
                document_queries = [{
                    **example,
                    "id": example[id_field], 
                    "text": example[query_field], 
                    "context_id": context_hash,
                    "context_text": example[context_field],
                } for example in examples]
                queries.extend(document_queries)

            # Document
            example0 = examples[0]
            documents.append({
                **example0,
                "id": context_hash,
                "text": example0[context_field]
            })  # Use the first example as the document
        
        print(f"Queries count: {len(queries)}")
        print(f"Documents (pre-segmentation) count: {len(documents)}")
        
        # Break down documents to segments
        segmented_documents = []
        for document in tqdm(documents, desc="Segmenting documents"):
            segmented_documents.extend(split_document_by_segments(document, model_name_or_path, max_tokens=max_tokens))
        documents = segmented_documents
        print(f"Documents (post-segmentation) count: {len(documents)}")
        
        return queries, documents
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.lower() == "nnlp-il/heq"

    def _get_document_tokens(self, document: dict):
        return self.tokenizer(document['text'])["input_ids"]
    
    def _hash_text(self, example, text_field, hash_field="hash"):
        example[hash_field] = hash_text(example[text_field])
        return example

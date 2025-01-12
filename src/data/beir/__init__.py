
import os
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import random
import importlib
import inspect


class BaseBeIRDataBuilder():
    def build_data(self, dataset_name: str, tokenizer, n: int = 0, split: str = 'test',max_tokens: int = 256, random_seed: int = 42):
        corpus_dataset = load_dataset(dataset_name, 'corpus')
        queries_dataset = load_dataset(dataset_name, 'queries')
        qrels_dataset = load_dataset(f'{dataset_name}-qrels')

        if n > 0:
            random.seed(random_seed)
            sampled_indexes = random.sample(range(len(qrels_dataset[split]), n))
        else:
            sampled_indexes = range(len(qrels_dataset[split]))  # Use all data
        
        qrels = [qrels_dataset[split][i] for i in sampled_indexes]
        corpus_ids = [str(x['corpus-id']) for x in qrels]
        query_ids = [str(x['query-id']) for x in qrels]
        # Filter the documents and queries
        documents0 = corpus_dataset['corpus'].filter(lambda x: str(x["_id"]) in corpus_ids)
        documents = []
        for document in documents0:
            documents.extend(self._split_document_by_sentences(document, tokenizer, max_tokens=max_tokens))
        
        queries0 = queries_dataset['queries'].filter(lambda x: str(x["_id"]) in query_ids)
        queries = []
        for qrel in qrels:
            query = next((q for q in queries0 if str(q['_id']) == str(qrel['query-id'])), None)
            context = next((d for d in documents0 if str(d['_id']) == str(qrel['corpus-id'])), None)
            if query is not None and context is not None:
                queries.append({
                    'id': str(qrel['query-id']),
                    'segment_id': query['segment_id'],
                    'text': query['text'],
                    'context': {
                        'id': str(qrel['corpus-id']),
                        'text': context['text']
                    }
                })
        return queries, documents
    
    def is_match(self, dataset_name: str) -> bool:
        return True

    def _split_document_by_segments(self, document: dict, max_tokens: int = 256):
        # Split document into sentences
        sentences = sent_tokenize(document['text'])

        # Split into segments based on max tokens
        segments = []
        current_segment = ""
        current_segment_tokens = 0

        for sentence in sentences:
            # Tokenize the sentence and count tokens
            sentence_tokens = len(self.tokenizer(sentence)["input_ids"])
            
            # Check if adding this sentence would exceed max tokens
            if current_segment_tokens + sentence_tokens <= max_tokens:
                current_segment += " " + sentence
                current_segment_tokens += sentence_tokens
            else:
                # Add the current segment to the list
                segments.append(current_segment.strip())
                # Start a new segment with the current sentence
                current_segment = sentence
                current_segment_tokens = sentence_tokens

        # Add the last segment
        if current_segment:
            segments.append(current_segment.strip())

        # Print segments
        document_segments = []
        for idx, segment in enumerate(segments):
            document_segments.append({
                '_id': str(document['_id']),
                'segments_id': idx,
                'text': segment
            })
        return document_segments

    def _get_document_tokens(self, document: dict):
        return self.tokenizer(document['text'])["input_ids"]


def build_data(dataset_name: str, **kwargs):
    builder = _get_builder(dataset_name)
    dataset = builder.build_data(dataset_name=dataset_name, **kwargs)
    return dataset


def _get_builder(dataset_name: str) -> BaseBeIRDataBuilder:
    folder_path = os.path.dirname(os.path.abspath(__file__))
    print(folder_path)
    # Get all Python files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.py') and f != '__init__.py']
    print(files)

    # List to hold instantiated classes
    builders = []

    # Loop through the files and load classes that inherit from BeIRDatasetLoader or extend it
    for file in files:
        module_path = os.path.join(folder_path, file)

        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(file[:-3], module_path)  # Remove .py extension
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Iterate over all members of the module to find classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class is BeIRDatasetLoader or extends it
            if issubclass(obj, BaseBeIRDataBuilder) and obj != BaseBeIRDataBuilder:
                builder = obj()
                builders.append(builder)

    # Filter classes by whether they match the prompt_type
    matching_builders = [l for l in builders if l.is_match(dataset_name)]

    if not matching_builders:
        return None

    builder = matching_builders[0]
    return builder

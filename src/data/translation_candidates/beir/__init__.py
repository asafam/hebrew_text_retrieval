import os
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import random
import importlib
import inspect
from translation.utils import count_tokens
from data.translation_candidates import TranslationCandidatesDataBuilder


BeIR = {
    'Misc': ['BeIR/msmarco'], 
    'Fact checking': ['BeIR/fever', 'BeIR/climate-fever', 'BeIR/scifact'],
    'Citation-Prediction': ['BeIR/scidocs'],
    'Duplicate question retrieval': ['BeIR/quora'], # CQADupStack
    'Argument retrieval': ['BeIR/arguana'], # Touche-2020
    'News retrieval': [], # TREC-NEWS, Robust04
    'Question answering': ['BeIR/nq', 'BeIR/hotpotqa'], # FiQA-2018
    'Tweet retrieval': [], # Signal-1M
    'Bio-medical IR': ['BeIR/trec-covid', 'BeIR/nfcorpus'], # BioASQ
    'Entity retrieval': ['BeIR/dbpedia-entity'],
}


class HuggingFaceBeIRDataBuilder(TranslationCandidatesDataBuilder):
    def build_data(self, 
                   dataset_name: str, 
                   model_name: str, 
                   n: int = 0, 
                   split: str = 'test',
                   max_tokens: int = 2048, 
                   random_seed: int = 42):
        corpus_dataset = load_dataset(dataset_name, 'corpus')
        queries_dataset = load_dataset(dataset_name, 'queries')
        qrels_dataset = load_dataset(f'{dataset_name}-qrels')

        documents_ids_to_index = {str(doc['_id']): idx for idx, doc in enumerate(corpus_dataset['corpus'])}

        # Unify splits of qrels
        qrels = []
        for split in qrels_dataset.keys():
            qrels.extend(qrels_dataset[split])

        # Sample documents
        all_documents = list(range(len(corpus_dataset['corpus'])))
        documents0 = []
        if 0 < n < len(all_documents):
            random.seed(random_seed)
            sampled_documents_indexes = random.sample(all_documents, n)
            for i in sampled_documents_indexes:
                documents0.append(corpus_dataset['corpus'][i])
        else:
            documents0 = corpus_dataset['corpus'] # Use all data
        
        # Break down documents to segments
        documents = []
        for document in documents0:
            documents.extend(self._split_document_by_segments(document, model_name, max_tokens=max_tokens))

        # Sample queries
        all_queries = list(range(len(queries_dataset['queries'])))
        queries0 = []
        if 0 < n < len(all_queries):
            random.seed(random_seed)
            sampled_queries_indexes = random.sample(all_queries, n)
            for i in sampled_queries_indexes:
                queries0.append(queries_dataset['queries'][i])
        else:
            queries0 = queries_dataset['queries'] # Use all data
        
        queries = []
        for query in queries0:
            # Identify the qrel containing the query
            qrel = next((q for q in qrels if str(q['query-id']) == str(query['_id'])), None)
            context = corpus_dataset['corpus'][documents_ids_to_index[str(qrel['corpus-id'])]] if qrel is not None else None
            if query is not None and context is not None:
                queries.append({
                    **query,
                    '_id': str(query['_id']),
                    'text': query['text'].strip(),
                    'context_id': context['_id'] if context is not None else None,
                    'context_text': context['text'] if context is not None else None
                })
        return queries, documents
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.startswith('BeIR/')

    def _split_document_by_segments(self, document: dict, model_name: str, max_tokens: int = 256):
        # Split document into sentences
        sentences = sent_tokenize(document['text'])

        # Split into segments based on max tokens
        segments = []
        current_segment = ""
        current_segment_tokens = 0

        for sentence in sentences:
            # Tokenize the sentence and count tokens
            sentence_tokens = count_tokens(sentence, model_name)
            
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
                **document,
                '_id': str(document['_id']),
                'text': document['text'],
                'segment_id': idx,
                'segment_text': segment
            })
        return document_segments

    def _get_document_tokens(self, document: dict):
        return self.tokenizer(document['text'])["input_ids"]

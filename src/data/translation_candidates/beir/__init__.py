import os
import pandas as pd
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


_CQADUPSTACK_SUBSETS = [
    'android', 'english', 'gaming', 'gis', 'mathematica', 'physics',
    'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress',
]


class _Records:
    """Minimal list wrapper that mimics HuggingFace Dataset subscript/iter/len."""
    def __init__(self, records: list):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]


class _DatasetDict:
    """Minimal wrapper that mimics HuggingFace DatasetDict."""
    def __init__(self, splits: dict):
        self._splits = splits

    def __getitem__(self, key):
        return self._splits[key]

    def keys(self):
        return self._splits.keys()


def _df_to_records(df: pd.DataFrame) -> list:
    """Normalize dtypes and convert DataFrame to list of dicts."""
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('').astype(str)
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('').astype(str)
    return df.to_dict('records')


def _load_parquet_fastparquet(repo_id: str, paths: list) -> list:
    """Download Parquet files from HuggingFace and read with fastparquet."""
    from huggingface_hub import hf_hub_download
    dfs = []
    for path in sorted(paths):
        local = hf_hub_download(repo_id=repo_id, filename=path, repo_type='dataset')
        dfs.append(pd.read_parquet(local, engine='fastparquet'))
    if not dfs:
        return []
    combined = pd.concat(dfs, ignore_index=True)
    return _df_to_records(combined)


def _load_qrels_tsv(repo_id: str, paths: list) -> list:
    """Download TSV qrel files from HuggingFace and return list of dicts."""
    from huggingface_hub import hf_hub_download
    records = []
    for path in paths:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=path, repo_type='dataset')
            df = pd.read_csv(local, sep='\t', header=0,
                             dtype={'query-id': str, 'corpus-id': str, 'score': int})
            records.extend(df.to_dict('records'))
        except Exception:
            pass
    return records


def _load_via_fastparquet(dataset_name: str):
    """
    Return (corpus_dataset, queries_dataset, qrels_dataset) for datasets that
    fail with PyArrow 19 due to Repetition level histogram mismatches.
    """
    from huggingface_hub import list_repo_files

    if dataset_name == 'BeIR/cqadupstack':
        all_files = list(list_repo_files('BeIR/cqadupstack', repo_type='dataset'))
        corpus_records, query_records, qrel_records = [], [], []

        for subset in _CQADUPSTACK_SUBSETS:
            corpus_paths = [f for f in all_files
                            if f.startswith(f'{subset}/corpus/') and f.endswith('.parquet')]
            corpus_records.extend(_load_parquet_fastparquet('BeIR/cqadupstack', corpus_paths))

            query_paths = [f for f in all_files
                           if f.startswith(f'{subset}/queries/') and f.endswith('.parquet')]
            query_records.extend(_load_parquet_fastparquet('BeIR/cqadupstack', query_paths))

            qrel_records.extend(_load_qrels_tsv(
                'BeIR/cqadupstack-qrels', [f'{subset}/test.tsv']))

        corpus = _DatasetDict({'corpus': _Records(corpus_records)})
        queries = _DatasetDict({'queries': _Records(query_records)})
        qrels = _DatasetDict({'test': _Records(qrel_records)})

    else:
        all_files = list(list_repo_files(dataset_name, repo_type='dataset'))
        corpus_paths = [f for f in all_files
                        if f.startswith('corpus/') and f.endswith('.parquet')]
        query_paths = [f for f in all_files
                       if f.startswith('queries/') and f.endswith('.parquet')]

        corpus_records = _load_parquet_fastparquet(dataset_name, corpus_paths)
        query_records = _load_parquet_fastparquet(dataset_name, query_paths)

        corpus = _DatasetDict({'corpus': _Records(corpus_records)})
        queries = _DatasetDict({'queries': _Records(query_records)})
        # qrels are TSV files — load_dataset works for these
        qrels = load_dataset(f'{dataset_name}-qrels')

    return corpus, queries, qrels


class HuggingFaceBeIRDataBuilder(TranslationCandidatesDataBuilder):
    def _load_beir_dataset(self, dataset_name: str):
        """
        Load corpus, queries, and qrels for a BeIR dataset.
        Tries the standard HuggingFace loader first; if it fails (e.g. PyArrow
        Repetition level histogram mismatch on some versions), falls back to
        reading Parquet files directly with fastparquet.
        """
        try:
            corpus  = load_dataset(dataset_name, 'corpus')
            queries = load_dataset(dataset_name, 'queries')
            qrels   = load_dataset(f'{dataset_name}-qrels')
            return corpus, queries, qrels
        except Exception as e:
            print(f"  [warn] load_dataset failed for {dataset_name} ({e}); retrying with fastparquet")
            return _load_via_fastparquet(dataset_name)

    def build_data(self,
                   dataset_name: str,
                   model_name: str = None,
                   n: int = 0,
                   split: str = 'test',
                   max_tokens: int = 2048,
                   random_seed: int = 42,
                   **kwargs):
        model_name = model_name or kwargs.get('model_name_or_path', '')
        random_seed = random_seed if random_seed != 42 else kwargs.get('random_state', random_seed)

        corpus_dataset, queries_dataset, qrels_dataset = self._load_beir_dataset(dataset_name)

        documents_ids_to_index = {str(doc['_id']): idx for idx, doc in enumerate(corpus_dataset['corpus'])}

        # Resolve which qrel splits to include. "all" = union of every split.
        # Other values must match a qrel split (with dev <-> validation aliasing).
        available_splits = list(qrels_dataset.keys())
        if split == 'all':
            target_splits = available_splits
        else:
            aliases = {'dev': 'validation', 'validation': 'dev'}
            if split in available_splits:
                target_splits = [split]
            elif aliases.get(split) in available_splits:
                target_splits = [aliases[split]]
            else:
                raise ValueError(
                    f"Split '{split}' not in qrels {available_splits} for {dataset_name}. "
                    f"Use one of {available_splits} or 'all'."
                )

        # Build qid -> (qrel, split). First occurrence wins when "all" sees a
        # qid in multiple splits.
        qid_to_qrel_split = {}
        for s in target_splits:
            for q in qrels_dataset[s]:
                qid = str(q['query-id'])
                if qid not in qid_to_qrel_split:
                    qid_to_qrel_split[qid] = (q, s)

        # Sample documents
        all_documents = list(range(len(corpus_dataset['corpus'])))
        documents0 = []
        if 0 < n < len(all_documents):
            random.seed(random_seed)
            sampled_documents_indexes = random.sample(all_documents, n)
            for i in sampled_documents_indexes:
                documents0.append(corpus_dataset['corpus'][i])
        else:
            documents0 = corpus_dataset['corpus']

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
            queries0 = queries_dataset['queries']

        queries = []
        for query in queries0:
            qid = str(query['_id'])
            entry = qid_to_qrel_split.get(qid)
            if entry is None:
                continue
            qrel, qrel_split = entry
            corpus_idx = documents_ids_to_index.get(str(qrel['corpus-id']))
            context = corpus_dataset['corpus'][corpus_idx] if corpus_idx is not None else None
            if context is None:
                continue
            queries.append({
                **query,
                'context_id': context['_id'],
                'context_text': context['text'],
                'split': qrel_split,
            })
        return queries, documents

    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.startswith('BeIR/')

    def _split_document_by_segments(self, document: dict, model_name: str, max_tokens: int = 256):
        sentences = sent_tokenize(document['text'])

        segments = []
        current_segment = ""
        current_segment_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence, model_name)

            if current_segment_tokens + sentence_tokens <= max_tokens:
                current_segment += " " + sentence
                current_segment_tokens += sentence_tokens
            else:
                segments.append(current_segment.strip())
                current_segment = sentence
                current_segment_tokens = sentence_tokens

        if current_segment:
            segments.append(current_segment.strip())

        document_segments = []
        for idx, segment in enumerate(segments):
            document_segments.append({
                **document,
                'segment_id': idx,
                'segment_text': segment,
            })
        return document_segments

    def _get_document_tokens(self, document: dict):
        return self.tokenizer(document['text'])["input_ids"]

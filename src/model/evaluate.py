from typing import Optional
import numpy as np
import logging
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
import os
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import pickle
import faiss

def create_index(doc_embeddings):
    logger = logging.getLogger('default')

    # Create a FAISS index for exact search (using cosine similarity with inner product)
    logger.info(f"Index {doc_embeddings.shape[0]} documents")
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])  # Inner product for cosine similarity
    index.add(doc_embeddings)  # Add the document embeddings to the index
    logger.info(f"Total documents indexed: {index.ntotal}")

    return index

def evaluate(query_embeddings, documents, index, k: int = 10):
    logger = logging.getLogger('default')

    # Retrieve top-k documents for each query
    distances, indices = index.search(query_embeddings, k)

    logger.info(f"Evaluating model with k={k}")

    # For each query, compute the evaluation metrics
    precision_scores = []
    mrr_scores = []
    ndcg_scores = []

    for i, (relevant_index, retrieved_indices) in enumerate(zip(range(len(documents)), indices)):
        precision = precision_at_k(relevant_index, retrieved_indices, k)
        mrr = mean_reciprocal_rank(relevant_index, retrieved_indices)
        ndcg = ndcg_at_k(relevant_index, retrieved_indices, k)

        precision_scores.append(precision)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

    # Compute average metrics for the dataset
    avg_precision = np.mean(precision_scores)
    avg_mrr = np.mean(mrr_scores)
    avg_ndcg = np.mean(ndcg_scores)

    logger.info(f"Average Precision@{k}: {avg_precision}")
    logger.info(f"Average MRR: {avg_mrr}")
    logger.info(f"Average NDCG@{k}: {avg_ndcg}")

    return dict(
        precision=avg_precision,
        mrr=avg_mrr,
        ndcg=avg_ndcg
    )



def precision_at_k(relevant_index, retrieved_indices, k):
    """Calculate Precision@k."""
    top_k_retrieved = retrieved_indices[:k]
    relevant_in_top_k = 1 if relevant_index in top_k_retrieved else 0
    return relevant_in_top_k / k


def mean_reciprocal_rank(relevant_index, retrieved_indices):
    """Calculate MRR."""
    for rank, doc_id in enumerate(retrieved_indices, start=1):
        if doc_id == relevant_index:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevant_index, retrieved_indices, k):
    """Calculate DCG@k."""
    dcg = 0.0
    for i in range(min(k, len(retrieved_indices))):
        if retrieved_indices[i] == relevant_index:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(relevant_index, retrieved_indices, k):
    """Calculate NDCG@k."""
    ideal_dcg = dcg_at_k(relevant_index, [relevant_index], k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevant_index, retrieved_indices, k) / ideal_dcg


def encode_texts(texts, tokenizer, model, device, batch_size=128):
    model.eval()
    # Create a DataLoader to batch the inputs
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    # Process each batch
    for batch in tqdm(dataloader, desc="Encoding batches"):
        # Tokenize the texts
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

        # Get the embeddings
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
        batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

        # Move embeddings to CPU and convert to NumPy
        all_embeddings.append(batch_embeddings.cpu().numpy())

    # Concatenate the embeddings from all batches
    embeddings = np.vstack(all_embeddings).astype('float32')
    return embeddings


def get_embeddings(texts, tokenizer, model, device, embedding_file_path: Optional[str] = None, batch_size: int = 128):
    logger = logging.getLogger('default')

    if embedding_file_path and os.path.exists(embedding_file_path):
        logger.info(f"Loading embeddings from {embedding_file_path}")
        with open(embedding_file_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        logger.info(f"Encode {len(texts)} texts to their embeddings")
        embeddings = encode_texts(texts, tokenizer, model, device, batch_size=batch_size)

        # Create the folder path if it does not exist
        if embedding_file_path:
            folder_path = os.path.dirname(embedding_file_path)
            os.makedirs(folder_path, exist_ok=True)

            logger.info(f"Save embeddings to {embedding_file_path}")
            with open(embedding_file_path, 'wb') as f:
                pickle.dump(embeddings, f)

    return embeddings
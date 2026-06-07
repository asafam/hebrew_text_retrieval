"""
Hard-negative patching for long-context retrieval evaluation.

Each document is concatenated with K semantically similar (hard negative) documents
found via FAISS nearest-neighbour search on the model's own document embeddings.
The result is a "needle-in-a-haystack" corpus where each entry is longer but contains
only one passage that is genuinely relevant to its paired query.
"""

import random
from typing import List, Literal

import faiss
import numpy as np
import torch
import torch.nn.functional as F

SEPARATOR = "\n\n---\n\n"


def find_hard_negatives(embeddings: torch.Tensor, k: int) -> List[List[int]]:
    """
    For each document embedding, return the indices of its k nearest neighbours
    (excluding itself). Used to select hard negatives for patching.

    Args:
        embeddings: shape (N, D), L2-normalised or unnormalised (will be normalised here)
        k: number of hard negatives per document

    Returns:
        List of N lists, each containing k document indices (the hard negatives)
    """
    emb_np = F.normalize(embeddings.float(), p=2, dim=-1).cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    # k+1 because the first result is always the document itself
    _, indices = index.search(emb_np, k + 1)
    result = []
    for i, row in enumerate(indices):
        neighbors = [int(j) for j in row if j != i][:k]
        result.append(neighbors)
    return result


def patch_documents(
    documents: List[str],
    hard_neg_indices: List[List[int]],
    separator: str = SEPARATOR,
    positive_position: Literal["first", "last", "random"] = "random",
    seed: int = 42,
) -> tuple[List[str], List[int]]:
    """
    Concatenate each document with its hard negatives to produce longer documents.

    Args:
        documents: original document texts, length N
        hard_neg_indices: per-document list of hard-negative indices (from find_hard_negatives)
        separator: string placed between concatenated passages
        positive_position: where to insert the genuine positive within the patch
        seed: random seed for reproducible "random" positioning

    Returns:
        patched_texts: list of N patched document strings
        positive_positions: list of N integers — the 0-based position of the genuine
                            positive within each patched document's parts list
    """
    rng = random.Random(seed)
    patched_texts = []
    positive_positions = []

    for i, doc in enumerate(documents):
        hard_negs = [documents[j] for j in hard_neg_indices[i]]
        k = len(hard_negs)

        if positive_position == "first":
            pos = 0
        elif positive_position == "last":
            pos = k
        else:
            pos = rng.randint(0, k)

        parts = hard_negs[:pos] + [doc] + hard_negs[pos:]
        patched_texts.append(separator.join(parts))
        positive_positions.append(pos)

    return patched_texts, positive_positions


def build_patched_corpus(
    documents: List[str],
    embeddings: torch.Tensor,
    k: int,
    separator: str = SEPARATOR,
    positive_position: Literal["first", "last", "random"] = "random",
    seed: int = 42,
) -> tuple[List[str], List[int]]:
    """
    High-level helper: given a list of document texts and their embeddings, build
    a patched corpus where each document is surrounded by k hard negatives.

    Returns:
        patched_texts: one patched string per original document
        positive_positions: position (0-based) of the genuine positive in each patch
    """
    if k == 0:
        return documents, [0] * len(documents)
    hard_neg_indices = find_hard_negatives(embeddings, k)
    return patch_documents(documents, hard_neg_indices,
                           separator=separator,
                           positive_position=positive_position,
                           seed=seed)

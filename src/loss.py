from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import pickle


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        """
        Parameters:
        - temperature: Scaling factor applied to the logits before applying the softmax function.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query_embeddings, positive_embeddings, negative_embeddings = None):
        """
        Compute InfoNCE loss with optional negative samples.
        
        If a negative sample is provided, it will be used. Otherwise, in-batch positives are treated as negatives.

        Args:
        query_embeddings (torch.Tensor): Embeddings of the queries, shape (batch_size, embedding_dim).
        positive_embeddings (torch.Tensor): Embeddings of the positive documents, shape (batch_size, embedding_dim).
        negative_embeddings (torch.Tensor, optional): Embeddings of the negative documents, shape (batch_size, embedding_dim) or None.
        temperature (float): Temperature scaling parameter.
        
        Returns:
        torch.Tensor: Computed InfoNCE loss.
        """
        batch_size = query_embeddings.size(0)

        # Normalize the embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        if negative_embeddings is not None:
            # Case 1: Explicit negative samples provided
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

            # Calculate similarity between query and positive embeddings
            positive_logits = torch.sum(query_embeddings * positive_embeddings, dim=-1, keepdim=True)

            # Calculate similarity between query and negative embeddings
            negative_logits = torch.sum(query_embeddings * negative_embeddings, dim=-1, keepdim=True)

            # Concatenate the positive and negative logits
            logits = torch.cat([positive_logits, negative_logits], dim=1)
        
        else:
            # Case 2: No explicit negatives provided, use in-batch negatives
            # Compute similarities between query and all positives in the batch (including itself)
            logits = torch.matmul(query_embeddings, positive_embeddings.T)

        # Apply temperature scaling
        logits = logits / self.temperature

        # Labels: For each query, the positive sample is always at index 0 in the concatenated logits
        if negative_embeddings is not None:
            # If we have explicit negatives, we use binary labels (positive at index 0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        else:
            # If we use in-batch negatives, the correct positive is the diagonal of the matrix
            labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)

        # Compute the InfoNCE loss using cross-entropy
        loss = F.cross_entropy(logits, labels)
        
        return loss

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

    def forward(self, anchor, positive, negatives):
        """
        Compute the InfoNCE loss.

        Parameters:
        - anchor: Tensor of shape (batch_size, embedding_dim) - anchor samples
        - positive: Tensor of shape (batch_size, embedding_dim) - positive samples corresponding to each anchor
        - negatives: Tensor of shape (batch_size, num_negatives, embedding_dim) - negative samples

        Returns:
        - loss: Computed InfoNCE loss
        """
        batch_size = anchor.size(0)
        num_negatives = negatives.size(1)

        # Normalize embeddings to unit vectors
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Calculate the positive logits (similarity between anchor and positive)
        positive_logits = torch.sum(anchor * positive, dim=-1, keepdim=True)  # Shape: (batch_size, 1)

        # Calculate the negative logits (similarity between anchor and negatives)
        negative_logits = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, num_negatives)

        # Concatenate positive and negative logits
        logits = torch.cat([positive_logits, negative_logits], dim=1)  # Shape: (batch_size, 1 + num_negatives)

        # Apply temperature scaling
        logits = logits / self.temperature

        # Create labels - 0 for the positive samples, as it is the first in the concatenated logits
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Compute the InfoNCE loss using cross-entropy
        loss = F.cross_entropy(logits, labels)

        return loss
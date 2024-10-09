from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from datasets import DatasetDict, Dataset
from tqdm import tqdm
from utils import *


def train(
    model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    device,
    epochs,
    start_epoch=0,
    checkpoint_dir='checkpoints',
    clip_value = None,
    always_save_checkpoint = True
):
    logger = logging.getLogger('default')
    logger.info("Start training...")

    model.train()
    best_val_loss = float('inf')  # Initialize best validation loss to infinity

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(start_epoch, epochs):
        total_train_loss = 0.0

        # Track progress in the training loop using tqdm
        train_progress = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False, total=len(train_dataloader))

        for batch_idx, batch in train_progress:

            if len(batch) == 4: # in case we have only (anchor, positive)
                anchor_ids, anchor_mask, positive_ids, positive_mask = [x.to(device) for x in batch]
                negative_ids, negative_mask = None, None
            elif len(batch) == 6: # in case we have only (anchor, positive, negative)
                anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = [x.to(device) for x in batch]

            # Forward pass to get the embeddings
            anchor_outputs = model(input_ids=anchor_ids, attention_mask=anchor_mask)
            anchor_embeds = anchor_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
            positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            negative_embeds = None
            if negative_ids or negative_mask:
                negative_outputs = model(input_ids=negative_ids, attention_mask=negative_mask)
                negative_embeds = negative_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            # Set negatives as the other positives in the batch
            batch_size = positive_embeds.size(0)
            negatives_mask = torch.eye(batch_size, dtype=torch.bool).to(device)  # Identity matrix to mask out positives
            positive_embeds_reshaped = positive_embeds.unsqueeze(0)  # Shape: (1, batch_size, embed_dim)

            # Use the mask to select negatives (all non-diagonal elements are negatives)
            negatives_embeds = positive_embeds_reshaped.masked_select(~negatives_mask.unsqueeze(-1)).view(batch_size, batch_size - 1, -1)
            if negative_embeds:
                # Pre-allocate a tensor for negatives and anchor embeddings (shape: batch_size, batch_size, embed_dim)
                negatives_embeds_with_negative = torch.zeros(batch_size, batch_size, positive_embeds.size(1))
                negatives_embeds_with_negative[:, :-1] = negatives_embeds  # Place all negatives (batch_size, batch_size - 1, embed_dim)
                negatives_embeds_with_negative[:, -1] = negative_embeds  # In-place assignment of negative embeddings at the last index
                negatives_embeds = negatives_embeds_with_negative

            # Compute the InfoNCE loss
            loss = criterion(anchor_embeds, positive_embeds, negatives_embeds)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            total_train_loss += loss.item()

            # Update tqdm progress bar with the current batch number and average loss
            train_progress.set_postfix({
                "Batch": batch_idx + 1,
                "Train Loss": total_train_loss / (batch_idx + 1)
            })

            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch + 1} / {epochs}, Batch {batch_idx + 1} / {len(train_dataloader)}, Train Loss: {loss.item()}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} / {epochs}, Train Loss: {avg_train_loss}")

        # Compute validation loss after each epoch
        avg_val_loss = validate(model, val_dataloader, criterion, device, epoch, epochs)
        logger.info(f"Epoch {epoch + 1} / {epochs}, Validation Loss: {avg_val_loss}")

        # Save checkpoint after each epoch
        if loss < best_val_loss or always_save_checkpoint:
            best_val_loss = loss if loss < best_val_loss else best_val_loss
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, loss)


def validate(model, val_dataloader, criterion, device, epoch, epochs):
    model.eval()
    total_val_loss = 0.0

    # Track progress in the validation loop using tqdm
    val_progress = tqdm(enumerate(val_dataloader), desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False, total=len(val_dataloader))

    with torch.no_grad():
        for batch_idx, batch in val_progress:

            if len(batch) == 4: # in case we have only (anchor, positive)
                anchor_ids, anchor_mask, positive_ids, positive_mask = [x.to(device) for x in batch]
                negative_ids, negative_mask = None, None
            elif len(batch) == 6: # in case we have only (anchor, positive, negative)
                anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = [x.to(device) for x in batch]

            # Forward pass to get the embeddings
            anchor_outputs = model(input_ids=anchor_ids, attention_mask=anchor_mask)
            anchor_embeds = anchor_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
            positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            negative_embeds = None
            if negative_ids or negative_mask:
                negative_outputs = model(input_ids=negative_ids, attention_mask=negative_mask)
                negative_embeds = negative_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            # Set negatives as the other positives in the batch
            batch_size = positive_embeds.size(0)
            negatives_mask = torch.eye(batch_size, dtype=torch.bool).to(device)  # Identity matrix to mask out positives
            positive_embeds_reshaped = positive_embeds.unsqueeze(0)  # Shape: (1, batch_size, embed_dim)

            # Use the mask to select negatives (all non-diagonal elements are negatives)
            negatives_embeds = positive_embeds_reshaped.masked_select(~negatives_mask.unsqueeze(-1)).view(batch_size, batch_size - 1, -1)
            if negative_embeds:
                # Pre-allocate a tensor for negatives and anchor embeddings (shape: batch_size, batch_size, embed_dim)
                negatives_embeds_with_negative = torch.zeros(batch_size, batch_size, positive_embeds.size(1))
                negatives_embeds_with_negative[:, :-1] = negatives_embeds  # Place all negatives (batch_size, batch_size - 1, embed_dim)
                negatives_embeds_with_negative[:, -1] = negative_embeds  # In-place assignment of negative embeddings at the last index
                negatives_embeds = negatives_embeds_with_negative

            # Compute the validation loss
            val_loss = criterion(anchor_embeds, positive_embeds, negatives_embeds)
            total_val_loss += val_loss.item()

            # Update tqdm progress bar with the current batch number and average loss
            val_progress.set_postfix({
                "Batch": batch_idx + 1,
                "Val Loss": total_val_loss / (batch_idx + 1)
            })

    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss
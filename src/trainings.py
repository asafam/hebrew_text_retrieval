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
            # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: Start of Batch")


            if len(batch) == 4: # in case we have only (query, positive)
                query_ids, query_mask, positive_ids, positive_mask = [x.to(device) for x in batch]
                negative_ids, negative_mask = None, None
            elif len(batch) == 6: # in case we have only (query, positive, negative)
                query_ids, query_mask, positive_ids, positive_mask, negative_ids, negative_mask = [x.to(device) for x in batch]

            # Forward pass to get the embeddings
            query_outputs = model(input_ids=query_ids, attention_mask=query_mask)
            query_embeds = query_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
            # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: After Query Forward Pass")

            positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
            positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
            # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: After Positive Forward Pass")

            negative_embeds = None
            if negative_ids or negative_mask:
                negative_outputs = model(input_ids=negative_ids, attention_mask=negative_mask)
                negative_embeds = negative_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
                # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: After Negative Forward Pass")

            # Compute the InfoNCE loss
            loss = criterion(query_embeds, positive_embeds, negative_embeds)
            # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: After Loss Calculation")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: After Backward Pass")

            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            # print_memory_usage(f"Epoch {epoch+1}, Batch {batch_idx+1}: After Optimizer Step")

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
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, loss.item())


def validate(model, val_dataloader, criterion, device, epoch, epochs):
    model.eval()
    total_val_loss = 0.0

    # Track progress in the validation loop using tqdm
    val_progress = tqdm(enumerate(val_dataloader), desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False, total=len(val_dataloader))

    with torch.no_grad():
        for batch_idx, batch in val_progress:

            if len(batch) == 4: # in case we have only (query, positive)
                query_ids, query_mask, positive_ids, positive_mask = [x.to(device) for x in batch]
                negative_ids, negative_mask = None, None
            elif len(batch) == 6: # in case we have only (query, positive, negative)
                query_ids, query_mask, positive_ids, positive_mask, negative_ids, negative_mask = [x.to(device) for x in batch]

            # Forward pass to get the embeddings
            query_outputs = model(input_ids=query_ids, attention_mask=query_mask)
            query_embeds = query_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
            positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            negative_embeds = None
            if negative_ids or negative_mask:
                negative_outputs = model(input_ids=negative_ids, attention_mask=negative_mask)
                negative_embeds = negative_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

            # Compute the validation loss
            val_loss = criterion(query_embeds, positive_embeds, negative_embeds)
            total_val_loss += val_loss.item()

            # Update tqdm progress bar with the current batch number and average loss
            val_progress.set_postfix({
                "Batch": batch_idx + 1,
                "Val Loss": total_val_loss / (batch_idx + 1)
            })

    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss


def print_memory_usage(stage=""):
    if torch.cuda.is_available():
        print(f"{stage} - Allocated memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"{stage} - Reserved memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        print(f"{stage} - Max allocated memory: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB\n")
    else:
        print(f"{stage} - CUDA is not available.")

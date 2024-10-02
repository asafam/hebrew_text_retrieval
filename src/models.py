from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import pickle


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
    clip_value = None
):
    logger = logging.getLogger('default')
    logger.info("Start training")

    model.to(device)
    model.train()
    best_val_loss = float('inf')  # Initialize best validation loss to infinity

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(start_epoch, epochs):
        total_train_loss = 0.0
        model.to(device)
        model.train()

        # Track progress in the training loop using tqdm
        train_progress = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False, total=len(train_dataloader))

        for batch_idx, batch in train_progress:

            if len(batch) == 4: # in case we have only (anchor, positive)
                anchor_ids, anchor_mask, positive_ids, positive_mask = [x.to(device) for x in batch]

                # Forward pass to get the embeddings
                anchor_outputs = model(input_ids=anchor_ids, attention_mask=anchor_mask)
                anchor_embeds = anchor_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
                positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                # Set negatives as the other positives in the batch
                # Create a matrix where the negatives are shifted versions of positives
                batch_size = positive_embeds.size(0)
    #             negatives_embeds = torch.stack([positive_embeds[i:] + positive_embeds[:i] for i in range(1, batch_size)], dim=0)
                # Create the negatives for each index `i` by excluding the positive embedding at index `i`
                negatives_embeds_list = []

                for i in range(batch_size):
                    # Exclude the current index `i` using slicing
                    negatives_embeds = torch.cat([positive_embeds[:i], positive_embeds[i+1:]], dim=0)

                    # Append the result to the list
                    negatives_embeds_list.append(negatives_embeds)

                # Stack the negatives for each sample in the batch
                # Each entry in the batch now has (batch_size - 1) negative embeddings
                negatives_embeds = torch.stack(negatives_embeds_list)

            elif len(batch) == 6: # in case we have only (anchor, positive, negative)
                anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = [x.to(device) for x in batch]

                # Forward pass to get the embeddings
                anchor_outputs = model(input_ids=anchor_ids, attention_mask=anchor_mask)
                anchor_embeds = anchor_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
                positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                negative_outputs = model(input_ids=negative_ids, attention_mask=negative_mask)
                negative_embeds = negative_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                # Set negatives as the other positives in the batch
                # Create a matrix where the negatives are shifted versions of positives
                batch_size = positive_embeds.size(0)
    #             negatives_embeds = torch.stack([positive_embeds[i:] + positive_embeds[:i] for i in range(1, batch_size)], dim=0)
                # Create the negatives for each index `i` by excluding the positive embedding at index `i`
                negatives_embeds_list = []

                for i in range(batch_size):
                    # Exclude the current index `i` using slicing
                    negatives_embeds = torch.cat([positive_embeds[:i], negative_embeds, positive_embeds[i+1:]], dim=0)

                    # Append the result to the list
                    negatives_embeds_list.append(negatives_embeds)

                # Stack the negatives for each sample in the batch
                # Each entry in the batch now has (batch_size - 1) negative embeddings
                negatives_embeds = torch.stack(negatives_embeds_list)

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

        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

        # Compute validation loss after each epoch
        avg_val_loss = validate(model, val_dataloader, criterion, device, epoch, epochs)
        logger.info(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        # Save checkpoint after each epoch
        if loss < best_val_loss:
            best_val_loss = loss
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)


def validate(model, val_dataloader, criterion, device, epoch, epochs):
    model.eval()
    total_val_loss = 0.0

    # Track progress in the validation loop using tqdm
    val_progress = tqdm(enumerate(val_dataloader), desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False, total=len(val_dataloader))

    with torch.no_grad():
        for batch_idx, batch in val_progress:

            if len(batch) == 4:
                anchor_ids, anchor_mask, positive_ids, positive_mask = [x.to(device) for x in batch]

                # Forward pass to get the embeddings
                anchor_outputs = model(input_ids=anchor_ids, attention_mask=anchor_mask)
                anchor_embeds = anchor_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
                positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                # Set negatives as the other positives in the batch
                # Create a matrix where the negatives are shifted versions of positives
                batch_size = positive_embeds.size(0)
    #             negatives_embeds = torch.stack([positive_embeds[i:] + positive_embeds[:i] for i in range(1, batch_size)], dim=0)
                # Create the negatives for each index `i` by excluding the positive embedding at index `i`
                negatives_embeds_list = []

                for i in range(batch_size):
                    # Exclude the current index `i` using slicing
                    negatives_embeds = torch.cat([positive_embeds[:i], positive_embeds[i+1:]], dim=0)

                    # Append the result to the list
                    negatives_embeds_list.append(negatives_embeds)

                # Stack the negatives for each sample in the batch
                # Each entry in the batch now has (batch_size - 1) negative embeddings
                negatives_embeds = torch.stack(negatives_embeds_list)

            elif len(batch) == 6:
                anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = [x.to(device) for x in batch]

                # Forward pass to get the embeddings
                anchor_outputs = model(input_ids=anchor_ids, attention_mask=anchor_mask)
                anchor_embeds = anchor_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                positive_outputs = model(input_ids=positive_ids, attention_mask=positive_mask)
                positive_embeds = positive_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                negative_outputs = model(input_ids=negative_ids, attention_mask=negative_mask)
                negative_embeds = negative_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

                # Set negatives as the other positives in the batch
                # Create a matrix where the negatives are shifted versions of positives
                batch_size = positive_embeds.size(0)
    #             negatives_embeds = torch.stack([positive_embeds[i:] + positive_embeds[:i] for i in range(1, batch_size)], dim=0)
                # Create the negatives for each index `i` by excluding the positive embedding at index `i`
                negatives_embeds_list = []

                for i in range(batch_size):
                    # Exclude the current index `i` using slicing
                    negatives_embeds = torch.cat([positive_embeds[:i], negative_embeds, positive_embeds[i+1:]], dim=0)

                    # Append the result to the list
                    negatives_embeds_list.append(negatives_embeds)

                # Stack the negatives for each sample in the batch
                # Each entry in the batch now has (batch_size - 1) negative embeddings
                negatives_embeds = torch.stack(negatives_embeds_list)

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
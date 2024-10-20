import os
import sys

project_dir = '/home/nlp/achimoa/projects/hebrew_text_encoder'

from typing import Optional
import argparse
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import torch
from torch.optim import AdamW
from data import *
from loss import *
from trainings import *
from utils import *


def main(
    model_name: str,
    task_name: str,
    dataset_name: str,
    batch_size: int,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-4,
    clip_value: Optional[float] = 1.0,
    infonce_temperature: float = 0.07,
    epochs: int = 10,
    checkpoint_dir: Optional[str] = None,
    source_checkpoint_dir: Optional[str] = None,
    source_checkpoint_epoch: Optional[int] = None,
    cuda_visible_devices: str = "0",
):
    # create the logger
    model_name_slug = model_name.replace('/', '_').replace('-', '_')
    task_name_slug = task_name.replace('/', '_').replace('-', '_')
    log_file = f"./logs/{model_name_slug}/train_{task_name_slug}.log"
    logger = setup_logger(log_file)
    
    # Print the arguments
    logger.info(f"Arguments:")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Task: {task_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Target checkpoint path: {checkpoint_dir}")
    logger.info(f"Source checkpoint path: {source_checkpoint_dir}")
    logger.info(f"Source checkpoint epoch: {source_checkpoint_epoch or 'None'}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Clip value: {clip_value}")
    logger.info(f"InfoNCE temperature: {infonce_temperature}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    # Add special tokens to the tokenizer
    new_tokens = [QUERY_TOKEN, DOCUMENT_TOKEN, *TASK_TOKENS.values()]
    additional_special_tokens = [token for token in new_tokens if token not in tokenizer.get_vocab()]
    logger.debug(f"Adding special tokens: {additional_special_tokens}")
    special_tokens = {
        "additional_special_tokens": additional_special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize the InfoNCE loss and the optimizer
    logger.info("Initialize the InfoNCE loss and the optimizer")
    criterion = InfoNCELoss(temperature=infonce_temperature)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load the latest checkpoint if available and resume training
    checkpoint_dir = checkpoint_dir or f"checkpoints/{model_name_slug}/checkpoints_{task_name_slug}"
    source_checkpoint_dir = source_checkpoint_dir or checkpoint_dir
    logger.info(f"Loading checkpoint")
    start_epoch = load_checkpoint(
        model, optimizer, checkpoint_dir=source_checkpoint_dir, device=device, epoch=source_checkpoint_epoch
    )
    

    # Iterate over datasets and train the model
    start_datetime = datetime.now()

    logger.info(f"Load dataset: {dataset_name}")
    dataset = build_dataset(dataset_name, splits=['train', 'validation'])
    
    # Create DataLoaders
    dataloaders = {}
    for split in ['train', 'validation']:
        # Tokenize the train dataset
        logger.info(f"Tokenizing {split} dataset")
        anchor_inputs = tokenizer(dataset[split]['anchor_text'], return_tensors='pt', padding=True, truncation=True)
        positive_inputs = tokenizer(dataset[split]['positive_text'], return_tensors='pt', padding=True, truncation=True)

        # Create DataLoader for training
        logger.info(f"Creating {split} dataloader")
        tensor_dataset = TensorDataset(anchor_inputs['input_ids'], anchor_inputs['attention_mask'],
                                        positive_inputs['input_ids'], positive_inputs['attention_mask'])
        dataloaders[split] = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=(split != 'validation'))

    # Train the model for this dataset
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['validation'],
        device=device,
        epochs=epochs,
        start_epoch=start_epoch,
        checkpoint_dir=checkpoint_dir,
        clip_value=clip_value
    )

    end_datetime = datetime.now()
    logger.info(f"Total training of {model_name} on {dataset_name} elapsed time is {(end_datetime - start_datetime).total_seconds()} seconds")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Model training script with dataset, model, checkpoint, and epoch arguments.")
    
    # Adding arguments
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model to use. For example: 'intfloat/multilingual-e5-large'.")
    parser.add_argument('--task_name', type=str, required=True, help="The name of the task to use: 'query_passage', 'title_passage', 'question_passage', etc.")
    parser.add_argument('--dataset_name', type=str, required=True, help="The name of the dataset to use: 'wiki40b' or 'synthesized_query_document'.")
    parser.add_argument('--epochs', type=int, default=10, help="The number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="The batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="The learning rate for training.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="The weight decay for training.")
    parser.add_argument('--clip_value', type=float, default=1.0, help="The gradient clipping value.")
    parser.add_argument('--infonce_temperature', type=float, default=0.07, help="The temperature for the InfoNCE loss.")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Path to the target checkpoint file for model serialization.")
    parser.add_argument('--source_checkpoint_dir', type=str, default=None, help="Path to the source checkpoint file to initialize the model from.")
    parser.add_argument('--source_checkpoint_epoch', type=int, default=None, help="The epoch number of the source checkpoint to initialize the model from.")
    parser.add_argument('--cuda_visible_devices', type=str, default="0", help="The CUDA_VISIBLE_DEVICES environment variable")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(**vars(args))
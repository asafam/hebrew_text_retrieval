from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import pickle


def setup_logger(file_path: str):
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    # Create or retrieve the logger
    logger = logging.getLogger('default')

    # Remove all existing handlers
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Stream Handler (for console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler (for file output)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_handler = logging.FileHandler(file_path, delay=False)  # Log file name (you can specify the path)
    file_handler.setLevel(logging.DEBUG) # Set the log level for file handler
    file_handler.setFormatter(formatter) # Use the same formatter
    logger.addHandler(file_handler)

    return logger


def load_checkpoint(model, optimizer, checkpoint_dir, device, epoch=None):
    logger = logging.getLogger('default')
    checkpoint_id = None
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            checkpoint_id = sorted(checkpoint_files, key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))[-1] if epoch is None else f"checkpoint_epoch_{epoch}.pth" # Get the latest checkpoint

    if checkpoint_id:
        logger.info(f"Loading checkpoint {checkpoint_id}")
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_id)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']  # return the epoch to resume from

    logger.info("No checkpoint found. Starting from scratch.")
    return 0  # Start from the first epoch if no checkpoint found


# Save model and optimizer state
def save_checkpoint(model, optimizer, epoch, checkpoint_dir, loss = None):
    logger = logging.getLogger('default')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")


def tokenize_inputs_and_create_dataloader(tokenizer, dataset, shuffle, batch_size, PAD_MASK = 0, replace_negative_text = False):
    logger = logging.getLogger('default')
    # Tokenize the dataset
    logger.info(f"Tokenizing dataset")
    anchor_inputs = tokenizer(dataset['anchor_text'], return_tensors='pt', padding=True, truncation=True)
    positive_inputs = tokenizer(dataset['positive_text'], return_tensors='pt', padding=True, truncation=True)

    if 'negative_text' in dataset.column_names:
        if not replace_negative_text:
            negative_inputs = tokenizer(dataset['negative_text'], return_tensors='pt', padding=True, truncation=True)
        else:
            # Handling cases where some elements are None
            negative_texts = [text or dataset['positive_text'][i] for i, text in enumerate(dataset['negative_text'])]
            negative_inputs = tokenizer(negative_texts, return_tensors='pt', padding=True, truncation=True)
            for i, text in enumerate(dataset['negative_text']):
                if text is None:
                    # Set attention_mask to 0 for the corresponding index
                    negative_inputs['attention_mask'][i] = PAD_MASK

    # Create DataLoader
    logger.info(f"Creating dataloader")
    if 'negative_text' in dataset.column_names:
        tensor_dataset = TensorDataset(anchor_inputs['input_ids'], anchor_inputs['attention_mask'],
                                        positive_inputs['input_ids'], positive_inputs['attention_mask'],
                                        negative_inputs['input_ids'], negative_inputs['attention_mask'])
    else:
        tensor_dataset = TensorDataset(anchor_inputs['input_ids'], anchor_inputs['attention_mask'],
                                        positive_inputs['input_ids'], positive_inputs['attention_mask'])
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
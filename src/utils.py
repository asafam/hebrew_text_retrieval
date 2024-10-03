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


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    logger = logging.getLogger('default')

    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]  # Get the latest checkpoint

    if latest_checkpoint:
        logger.info(f"Loading checkpoint {latest_checkpoint}")
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']  # return the epoch to resume from

    logger.info("No checkpoint found. Starting from scratch.")
    return 0  # Start from the first epoch if no checkpoint found


# Save model and optimizer state
def save_checkpoint(model, optimizer, epoch, checkpoint_dir, loss):
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
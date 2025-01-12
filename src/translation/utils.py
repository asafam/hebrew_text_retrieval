from typing import *
import torch
import os
import psutil


def prepare_data_for_translation(data: List[dict], tokenizer, text_key='text'):
    


def batch_texts_by_length(data: List[dict], tokenizer, text_key='text', max_sentence_length: int = 0, batch_size: int = 512):
    # Tokenize and compute text lengths
    data_with_lengths = [{**item, "text_length": len(tokenizer(item[text_key])["input_ids"])} for item in data]

    # Sort texts by length
    sorted_data = sorted(data_with_lengths, key=lambda x: x["text_length"])

    # Group sorted texts into batches
    batches = [sorted_data[i:i + batch_size] for i in range(0, len(sorted_data), batch_size)]
    return batches


def cache_prefix(model, tokenizer, prefix, batch_size, device):
    # Tokenize the prefix and generate its cache
    prefix_inputs = tokenizer([prefix] * batch_size, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**prefix_inputs, use_cache=True)  # Run once to build the cache

    # Extract past_key_values from the model output
    past_key_values = output.past_key_values

    return past_key_values


def get_gpu_memory_usage():
    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Peak usage
    return current_memory, max_memory


def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Resident Set Size in MB

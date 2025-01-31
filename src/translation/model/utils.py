from typing import *
import torch
import os
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


STOP_TOKEN = "<STOP>"
PAD_TOKEN = "<PAD>"


def get_model_and_tokenizer(model_name: str, 
                            device: str = "cuda", 
                            pad_token: str = PAD_TOKEN,
                            special_tokens: List[str] = [],
                            extra_tokens: List[str] = [STOP_TOKEN]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model {model_name} on device {device}")
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        
        tokenizer.add_special_tokens({'pad_token': pad_token})
        
        if special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        for extra_token in extra_tokens:
            tokenizer.add_tokens(extra_token)
            
        model.resize_token_embeddings(len(tokenizer)) # Resize the model's embeddings to account for the new token
    
    return model, tokenizer


def get_stopping_criteria(tokenizer, stop_token: str = STOP_TOKEN) -> StoppingCriteria:
    # Define a stopping criteria that stops when the model generates the stop token
    class StopOnToken(StoppingCriteria):
        def __init__(self, stop_token_id):
            self.stop_token_id = stop_token_id

        def __call__(self, input_ids, scores, **kwargs):
            return input_ids[0, -1] == self.stop_token_id
        
    # Get the stop token ID
    stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
    assert stop_token_id is not None, "Stop token is not properly tokenized"
    
    # create stop token strategy
    stop_on_token = StopOnToken(stop_token_id)
    
    # Wrap in a StoppingCriteriaList
    stopping_criteria = StoppingCriteriaList([stop_on_token])
    
    return stopping_criteria


def batch_texts_by_length(data: List[dict], tokenizer, text_key='dynamic_prompt', batch_size: int = 512):
    # Tokenize and compute text lengths
    data_with_lengths = [{**item, 'text_length': len(tokenizer(item[text_key])["input_ids"])} for item in data]

    # Sort texts by length
    sorted_data = sorted(data_with_lengths, key=lambda x: x['text_length'])

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


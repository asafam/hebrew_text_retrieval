from typing import List 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
from translation.utils import *


def translate(model, tokenizer, texts: List[str], past_key_values, max_new_tokens, device):
    start_datetime = datetime.now()

    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Run batch inference using cached prefix
    with torch.no_grad():
        model_start_datetime = datetime.now()
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            past_key_values=past_key_values,  # Use cached prefix for all queries
            pad_token_id=tokenizer.pad_token_id  # Set pad token during generation
        )
        model_start_datetime = datetime.now()

    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    result = dict(
        decoded_outputs=decoded_outputs,
        inputs=inputs,
        input_tokens=[input_ids.size(0) for input_ids in inputs["input_ids"]],
        outputs=outputs,
        output_tokens=[output.size(0) for output in outputs],
        model_time=(model_start_datetime - start_datetime).total_seconds(),
        translation_time=(datetime.now() - start_datetime).total_seconds(),
    )

    # Clear CUDA memory
    del inputs
    del outputs
    del decoded_outputs

    return result


def translation_pipeline(model_name: str, 
                         prefix: str, 
                         data: List[dict], 
                         batch_size: int = 512, 
                         ids: list[str] = ['id'],
                         max_new_tokens=None,
                         verbose: bool = False,
                         device = "cuda"):
    # Create the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Cache the prefix
    past_key_values = cache_prefix(model, prefix, batch_size)

    # Create the batches
    original_order = [tuple(item[field] for field in ids) for item in data] # Store the original order
    batches = batch_texts_by_length(data, batch_size=batch_size)

    # Translate a batch of texts
    translations = []
    resources_usage = []
    for batch_idx, batch in tqdm(enumerate(batches), desc="Batches", total=len(batches)):
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Measure memory before
        gpu_memory_before, _ = get_gpu_memory_usage()
        cpu_memory_before = get_cpu_memory_usage()

        # Adjust past_key_values to match the batch size
        current_batch_size = len(batch)
        if current_batch_size < batch_size:
            adjusted_past_key_values = [
                tuple(past_layer[:current_batch_size] for past_layer in layer)
                for layer in past_key_values
            ]
        else:
            adjusted_past_key_values = past_key_values

        # Estimate the max new tokens value
        max_new_tokens = max([x['text_length'] for x in batch]) * 1.25 if max_new_tokens is None else max_new_tokens

        # Translate batch
        batch_texts = [prefix + '\n\n' + x['text'] for x in batch]
        results = translate(model=model, 
                            texts=batch_texts, 
                            past_key_values=adjusted_past_key_values, 
                            max_new_tokens=max_new_tokens)

        # Measure memory after
        gpu_memory_after, gpu_peak_memory = get_gpu_memory_usage()
        cpu_memory_after = get_cpu_memory_usage()
        resources_usage.append({
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "gpu_memory_before": gpu_memory_before,
            "gpu_memory_after": gpu_memory_after,
            "gpu_peak_memory": gpu_peak_memory,
            "cpu_memory_before": cpu_memory_before,
            "cpu_memory_after": cpu_memory_after
        })

        # Print memory usage for the batch
        if verbose:
            print(f"Batch {batch_idx + 1} for batch size {batch_size}:")
            print(f"  GPU memory used: {gpu_memory_after - gpu_memory_before:.2f} MB")
            print(f"  GPU peak memory: {gpu_peak_memory:.2f} MB")
            print(f"  CPU memory used: {cpu_memory_after - cpu_memory_before:.2f} MB")

        # Collect results
        for i, _ in enumerate(results['decoded_outputs']):
            translation = {
                **batch[i],
                "translation": results['decoded_outputs'][i],
                # "inputs": results['inputs'][i],
                "input_tokens": results['input_tokens'][i],
                # "outputs": results['outputs'][i],
                "output_tokens": results['output_tokens'][i],
                "translation_time": results['translation_time'],
                "model_time": results['model_time'],
                "model_name": model_name,
                "prefix": prefix,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
            }
            translations.append(translation)

    # Sort translations by original order
    translations = sorted(translations, key=lambda x: original_order.index(tuple(x[field] for field in ids)))

    # Clear CUDA memory
    del model  # Delete the model
    del tokenizer  # Delete the tokenizer
    torch.cuda.empty_cache()  # Clear cached memory
    import gc
    gc.collect()  # Run garbage collection

    return translations, resources_usage
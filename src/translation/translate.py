from typing import List 
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
from tqdm import tqdm
from translation.utils import *


def translate(model, 
              tokenizer, 
              texts: List[str], 
              past_key_values, 
              max_new_tokens: int,
              device: str = "cuda"):
    start_datetime = datetime.now()

    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Define stopping criteria
    stopping_criteria = get_stopping_criteria(tokenizer)

    # Run batch inference using cached prefix
    with torch.no_grad():
        model_start_datetime = datetime.now()
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
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


def run_translation_pipeline(source_file_path: str,
                             prompt_file_name: str,
                             model_name: str,
                             batch_size: int,
                             max_new_tokens=None,
                             verbose: bool = False,
                             use_cached_prefix: bool = True,
                             device = "cuda",
                             force: bool = False,
                             **kwargs):
    # Determine the output file path
    translation_output_file_path = os.path.join(os.path.dirname(source_file_path), model_name, os.path.basename(source_file_path))

    # Load the data
    file_path = translation_output_file_path if os.path.exists(translation_output_file_path) else source_file_path
    df = pd.read_csv(file_path, encoding='utf-8')
    filtered_df = df[df['translation'].isnull()] if not force else df

    # Check if the file has been fully translated
    if not force and os.path.exists(translation_output_file_path) and filtered_df.empty:
        print(f"Skipping translation of {translation_output_file_path} as have been translated.")
        return

    # Create the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name)

    # Load the prompt yaml file
    with open(prompt_file_name, 'r') as file:
        prompt_data = yaml.safe_load(file)
    prompt = prompt_data['query']
    meta_fields = {
        'english_key': kwargs.get('english_key', 'אנגלית'),
        'hebrew_key': kwargs.get('hebrew_key', 'עברית'),
        'context_key': kwargs.get('context_key', 'רקע'),
    }
    prompt_prefix = prompt['prompt_prefix'].format_map(SafeDict(meta_fields))
    prompt_template = prompt['prompt_template'].format_map(SafeDict(meta_fields))

    # Cache the prefix
    past_key_values = cache_prefix(model, tokenizer, prompt_prefix, batch_size, device)

    # Create the batches
    data = [{
            **item, 
            'dynamic_prompt': prompt_template.format(**item)
        } for item in filtered_df.to_dict(orient='records')]
    batches = batch_texts_by_length(data, tokenizer, batch_size=batch_size)

    # Get the prefix length for the new tokens estimation
    prefix_length = len(tokenizer(prompt_prefix)["input_ids"])

    # Translate a batch of texts
    resources_usage = []
    for batch_idx, batch in tqdm(enumerate(batches), desc="Batches", total=len(batches)):
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Measure memory before
        gpu_memory_before, _ = get_gpu_memory_usage()
        cpu_memory_before = get_cpu_memory_usage()

        # Estimate the max new tokens value
        max_new_tokens = max_new_tokens or (prefix_length + (max([x['text_length'] for x in batch]) * 1.25))

        # Prepare the batch for translation
        batch_texts = [(prompt_prefix + '\n' + x['dynamic_prompt']).strip() for x in batch]
        translation_args = dict(
            model=model, 
            tokenizer=tokenizer,
            texts=batch_texts, 
            max_new_tokens=max_new_tokens,
            device=device
        )

        if use_cached_prefix:
            # Adjust past_key_values to match the batch size
            current_batch_size = len(batch)
            if current_batch_size < batch_size:
                adjusted_past_key_values = [
                    tuple(past_layer[:current_batch_size] for past_layer in layer)
                    for layer in past_key_values
                ]
            else:
                adjusted_past_key_values = past_key_values
            
            # Add past_key_values to the translation arguments
            translation_args['past_key_values'] = adjusted_past_key_values
        
        # Translate batch
        results = translate(**translation_args)

        # Collect results
        batch_translations = []
        for i, item in enumerate(batch):
            translation = {
                **item,
                **meta_fields,
                "translation": results['decoded_outputs'][i],
                "input_tokens": results['input_tokens'][i],
                "output_tokens": results['output_tokens'][i],
                "translation_time": results['translation_time'],
                "model_time": results['model_time'],
                "model_name": model_name,
                "prompt_prefix": prompt_prefix,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "prompt_file_name": prompt_file_name,
                "use_cached_prefix": use_cached_prefix,
                "device": device,
                "timestamp": timestamp
            }
            batch_translations.append(translation)
        
        # Save the results
        batch_df = pd.DataFrame(batch_translations)
        df = pd.merge(df, batch_df, on='id', how='outer')
        df.to_csv(translation_output_file_path, encoding='utf-8', index=False)

        # Measure memory after
        gpu_memory_after, gpu_peak_memory = get_gpu_memory_usage()
        cpu_memory_after = get_cpu_memory_usage()
        timestamp = datetime.now()
        resources_usage.append({
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "gpu_memory_before": gpu_memory_before,
            "gpu_memory_after": gpu_memory_after,
            "gpu_peak_memory": gpu_peak_memory,
            "cpu_memory_before": cpu_memory_before,
            "cpu_memory_after": cpu_memory_after,
            "timestamp": timestamp,
        })

        # Print memory usage for the batch
        if verbose:
            print(f"Batch {batch_idx + 1} for batch size {batch_size}:")
            print(f"  GPU memory used: {gpu_memory_after - gpu_memory_before:.2f} MB")
            print(f"  GPU peak memory: {gpu_peak_memory:.2f} MB")
            print(f"  CPU memory used: {cpu_memory_after - cpu_memory_before:.2f} MB")

    # Save the resources usage
    resources_usage_df = pd.DataFrame(resources_usage)
    resources_usage_df.to_csv(translation_output_file_path.replace('.csv', '_resources_usage.csv'), encoding='utf-8', index=False)

    # Clear CUDA memory
    del model  # Delete the model
    del tokenizer  # Delete the tokenizer
    torch.cuda.empty_cache()  # Clear cached memory
    import gc
    gc.collect()  # Run garbage collection

    return df, resources_usage

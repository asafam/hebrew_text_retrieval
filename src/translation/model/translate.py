from typing import List 
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
from tqdm import tqdm
from translation.model.utils import *
from translation.utils import *


def translate(model, 
              tokenizer, 
              texts: List[str], 
              past_key_values,
              max_new_tokens: int,
              use_stop_token: bool = True,
              device: str = "cuda") -> dict:
    start_datetime = datetime.now()

    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Run batch inference using cached prefix
    with torch.no_grad():
        # Define generate method arguments
        args = dict(
            max_new_tokens=max_new_tokens,
            use_cache=True,
            past_key_values=past_key_values,  # Use cached prefix for all queries
            pad_token_id=tokenizer.pad_token_id  # Set pad token during generation
        )
        # Add stopping criteria if needed
        if use_stop_token:
            args['stopping_criteria'] = get_stopping_criteria(tokenizer)
        
        # Generate outputs
        model_start_datetime = datetime.now()
        outputs = model.generate(inputs["input_ids"], **args)
        model_end_datetime = datetime.now()

    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    result = dict(
        decoded_outputs=decoded_outputs,
        inputs=inputs,
        input_tokens=[input_ids.size(0) for input_ids in inputs["input_ids"]],
        outputs=outputs,
        output_tokens=[output.size(0) for output in outputs],
        model_time=(model_end_datetime - model_start_datetime).total_seconds(),
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
                             use_stop_token: bool = True,
                             device = "cuda",
                             limit: int = 0,
                             force: bool = False,
                             max_new_tokens_factor: float = 1.75,
                             **kwargs):
    # Determine the output file path
    model_name_slug = model_name.replace('/', '_')
    translation_output_file_path = os.path.join(os.path.dirname(source_file_path), model_name_slug, os.path.basename(source_file_path))

    # Load the data
    file_path = translation_output_file_path if os.path.exists(translation_output_file_path) else source_file_path
    df = pd.read_csv(file_path, encoding='utf-8')
    if limit > 0:
        print(f"Limiting the number of texts to {limit}.")
        df = df.head(limit)
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
    prompt_type = 'query' if source_file_path.endswith('queries.csv') else 'document'
    prompt = prompt_data[prompt_type]
    prompt_meta_fields = {
        'english_key': kwargs.get('english_key', 'אנגלית'),
        'hebrew_key': kwargs.get('hebrew_key', 'עברית'),
        'context_key': kwargs.get('context_key', 'הקשר'),
    }
    prompt_prefix = prompt['prompt_prefix'].format_map(SafeDict(prompt_meta_fields))
    prompt_template = prompt['prompt_template'].format_map(SafeDict(prompt_meta_fields))

    # Cache the prefix
    past_key_values = cache_prefix(model, tokenizer, prompt_prefix, batch_size, device)

    # Create the batches
    id_columns = ['id', 'segment_id'] if 'segment_id' in filtered_df.columns else ['id', 'context_id']
    def limit_context(x):
        tokens = tokenizer.tokenize(x)
        new_x = tokenizer.convert_tokens_to_string(tokens[:64])
        return new_x
    if 'context_text' in filtered_df.columns:
        filtered_df['context_text'] = filtered_df['context_text'].apply(limit_context)
    data = [{
            **{id: item[id] for id in id_columns},
            'dynamic_prompt': prompt_template.format(**item)
        } for item in filtered_df.to_dict(orient='records')]
    batches = batch_texts_by_length(data, tokenizer, batch_size=batch_size)

    # Translate a batch of texts
    resources_usage = []
    translation_datetime = datetime.now()
    for batch_idx, batch in tqdm(enumerate(batches), desc="Batches", total=len(batches)):
        batch_datetime = datetime.now()
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Measure memory before
        gpu_memory_before, _ = get_gpu_memory_usage()
        cpu_memory_before = get_cpu_memory_usage()

        # Estimate the max new tokens value
        max_new_tokens = max_new_tokens or (max([x['text_length'] for x in batch]) * max_new_tokens_factor)

        # Prepare the batch for translation
        batch_texts = [(prompt_prefix + '\n' + x['dynamic_prompt']).strip() for x in batch]
        translation_args = dict(
            model=model, 
            tokenizer=tokenizer,
            texts=batch_texts, 
            max_new_tokens=max_new_tokens,
            use_stop_token=use_stop_token,
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
        for i, batch_item in enumerate(batch):
            translation = {
                **batch_item,
                **prompt_meta_fields,
                "translation": results['decoded_outputs'][i],
                "input_tokens": results['input_tokens'][i],
                "output_tokens": results['output_tokens'][i],
                "translation_time": results['translation_time'],
                "model_time": results['model_time'],
                "model_name": model_name,
                "prompt_prefix": prompt_prefix,
                "prompt": batch_texts[i],
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "prompt_file_name": prompt_file_name,
                "use_cached_prefix": use_cached_prefix,
                "device": device,
                "batch_datetime": batch_datetime,
                "translation_datetime": translation_datetime,
            }
            batch_translations.append(translation)
        
        # Save the results
        batch_df = pd.DataFrame(batch_translations)
        df = pd.merge(df, batch_df, on=id_columns, how='left')
        # Overwrite columns in left_df with values from right_df if available
        for col in batch_df.columns:
            if (col + '_y') in df.columns and col not in id_columns:  # Skip key columns
                df[col] = df[col + '_y'].combine_first(df[col + '_x'])
                df.drop(columns=[col + '_x', col + '_y'], inplace=True)
        os.makedirs(os.path.dirname(translation_output_file_path), exist_ok=True)
        df.to_csv(translation_output_file_path, encoding='utf-8', index=False)

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
            "cpu_memory_after": cpu_memory_after,
            "batch_datetime": batch_datetime,
            "translation_datetime": translation_datetime,
        })

        # Print memory usage for the batch
        if verbose:
            print(f"Batch {batch_idx + 1} for batch size {batch_size}:")
            print(f"  GPU memory used: {gpu_memory_after - gpu_memory_before:.2f} MB")
            print(f"  GPU peak memory: {gpu_peak_memory:.2f} MB")
            print(f"  CPU memory used: {cpu_memory_after - cpu_memory_before:.2f} MB")

    # Save the resources usage
    resources_usage_df = pd.DataFrame(resources_usage)
    resources_usage_output_file_path = translation_output_file_path.replace('.csv', '_resources_usage.csv')
    os.makedirs(os.path.dirname(resources_usage_output_file_path), exist_ok=True)
    resources_usage_df.to_csv(resources_usage_output_file_path, encoding='utf-8', index=False)

    # Clear CUDA memory
    del model  # Delete the model
    del tokenizer  # Delete the tokenizer
    torch.cuda.empty_cache()  # Clear cached memory
    import gc
    gc.collect()  # Run garbage collection

    return df, resources_usage

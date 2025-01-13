from typing import List 
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
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


def run_translation_pipeline(model_name: str,
                             prompt_prefix: str, 
                             data: List[dict], 
                             batch_size: int, 
                             ids: list[str] = ['id'],
                             max_new_tokens=None,
                             verbose: bool = False,
                             cache_prefix: bool = True,
                             device = "cuda"):
    # Create the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name)

    # Cache the prefix
    past_key_values = cache_prefix(model, prompt_prefix, batch_size)

    # Create the batches
    original_order = [tuple(item[field] for field in ids) for item in data] # Store the original order
    batches = batch_texts_by_length(data, batch_size=batch_size)

    # Get the prefix length for the new tokens estimation
    prefix_length = len(tokenizer(prompt_prefix)["input_ids"])

    # Translate a batch of texts
    translations = []
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
        batch_texts = [prompt_prefix + '\n' + x['text'] for x in batch]
        translation_args = dict(
            model=model, 
            texts=batch_texts, 
            max_new_tokens=max_new_tokens,
            device=device
        )

        if cache_prefix:
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
                "input_tokens": results['input_tokens'][i],
                "output_tokens": results['output_tokens'][i],
                "translation_time": results['translation_time'],
                "model_time": results['model_time'],
                "model_name": model_name,
                "prompt_prefix": prompt_prefix,
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


def translate_queries(data_file_path: str,
                      translation_output_file_path: str,
                      prompt_file_name: str,
                      model_name: str,
                      batch_size: int,
                      max_new_tokens: int,
                      cache_prefix: bool,
                      device: str,
                      force: bool,) -> List[dict]:
    if not force and os.path.exists(translation_output_file_path):
        print(f"Skipping translation of {data_file_path} as the output file already exists.")
        return

    # Load the prompt yaml file
    with open(prompt_file_name, 'r') as file:
        data = yaml.safe_load(file)
    prompt = data['query']

    # Load the data
    queries = pd.read_csv(data_file_path, encoding='utf-8')
    
    # Cast data to list of dictionaries
    prompt_prefix = prompt['prompt_prefix']
    data = [{
            'id': query['id'],
            'text': prompt['prompt_template'].format(query=query['text'], context=query['context']).strip()
        } for query in queries.to_list()]

    # Run the translation pipeline
    translations, resources_usage = run_translation_pipeline(
        model_name=model_name,
        prompt_prefix=prompt_prefix,
        data=data,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        cache_prefix=cache_prefix,
        device=device
    )

    # Save the translations
    translated_queries = []
    for query, translation in zip(queries, translations):
        assert query['id'] == translation['id'], "The ids do not match" # Check that the ids match
        translated_query = {
            **query,
            **translation
        }
        translated_query['task'] = 'query_translation'
        translated_query['timestamp'] = datetime.now()
        translated_query['model'] = model_name
        translated_query['prompt_file_name'] = prompt_file_name
        translated_query['batch_size'] = batch_size
        translated_query['max_new_tokens'] = max_new_tokens
        translated_query['cache_prefix'] = cache_prefix
        translated_query['device'] = device
        translated_queries.append(translated_query)
    
    # Save the results
    translated_queries = pd.DataFrame(translated_queries)
    translated_queries.to_csv(translation_output_file_path, encoding='utf-8', index=False)

    # Save the resources usage
    resources_usage = pd.DataFrame(resources_usage)
    resources_usage.to_csv(translation_output_file_path.replace('.csv', '_resources_usage.csv'), encoding='utf-8', index=False)

    return translated_queries


def translate_documents(data_file_path: str,
                        translation_output_file_path: str,
                        prompt_file_name: str,
                        model_name: str,
                        batch_size: int,
                        max_new_tokens: int,
                        cache_prefix: bool,
                        device: str,
                        force: bool,) -> List[dict]:
    if not force and os.path.exists(translation_output_file_path):
        print(f"Skipping translation of {data_file_path} as the output file already exists.")
        return

    # Load the prompt yaml file
    with open(prompt_file_name, 'r') as file:
        data = yaml.safe_load(file)
    prompt = data['document']

    # Load the data
    documents = pd.read_csv(data_file_path, encoding='utf-8')
    
    # Cast data to list of dictionaries
    prompt_prefix = prompt['prompt_prefix']
    data = [{
            'id': document['id'],
            'text': prompt['prompt_template'].format(document=document['text']).strip()
        } for document in documents.to_list()]

    # Run the translation pipeline
    translations, resources_usage = run_translation_pipeline(
        model_name=model_name,
        prompt_prefix=prompt_prefix,
        data=data,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        cache_prefix=cache_prefix,
        device=device
    )

    # Save the translations
    translated_documents = []
    for document, translation in zip(documents, translations):
        assert document['id'] == translation['id'], "The ids do not match" # Check that the ids match
        translated_document = {
            **document,
            **translation
        }
        translated_document['task'] = 'query_translation'
        translated_document['timestamp'] = datetime.now()
        translated_document['model'] = model_name
        translated_document['prompt_file_name'] = prompt_file_name
        translated_document['batch_size'] = batch_size
        translated_document['max_new_tokens'] = max_new_tokens
        translated_document['cache_prefix'] = cache_prefix
        translated_document['device'] = device
        translated_documents.append(translated_document)
    
    # Save the results
    translated_documents = pd.DataFrame(translated_documents)
    translated_documents.to_csv(translation_output_file_path, encoding='utf-8', index=False)

    # Save the resources usage
    resources_usage = pd.DataFrame(resources_usage)
    resources_usage.to_csv(translation_output_file_path.replace('.csv', '_resources_usage.csv'), encoding='utf-8', index=False)

    return translated_documents


def main():
    parser = argparse.ArgumentParser(description="Translate queries and documents using a specified model.")

    parser.add_argument('--data_file_paths', type=str, nargs='+', required=True, help="Paths to the input files containing queries.")
    parser.add_argument('--translation_output_file_path', type=str, required=True, help="Path to save the translated queries.")
    parser.add_argument('--prompt_file_name', type=str, required=True, help="File name for the translation prompt.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the translation model.")
    parser.add_argument('--batch_size', type=int, default=32, help="Number of queries to translate in each batch.")
    parser.add_argument('--max_new_tokens', type=int, default=None, help="Maximum number of tokens to generate per query.")
    parser.add_argument('--cache_prefix', action='store_true', help="Use cached results for translation.")
    parser.add_argument('--force', action='store_true', help="Force re-translation if output file exists.")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for data_file_path in args.data_file_paths:
        args = dict(
            data_file_path=args.data_file_path,
            translation_output_file_path=args.translation_output_file_path,
            prompt_file_name=args.prompt_file_name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            cache_prefix=args.cache_prefix,
            device=device,
            force=args.force
        )
        if 'queries' in data_file_path:
            translate_queries(**args)
        elif 'documents' in data_file_path:
            translate_documents(**args)
            

if __name__ == "__main__":
    main()
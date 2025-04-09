import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from llms.router import get_llm
from tqdm import tqdm
import os
import re
import time
from translation.api.utils import *

    

def translate(system_prompt: str,
              user_prompt: str,
              model_name: str,
              temperature=0.7,
              response_format=Translation,
              fail_on_error=True):
    start_datetime = datetime.now()

    # Define the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Run chat completion inference
    try:
        model_start_datetime = datetime.now()
        client = get_llm(model_name)
        result = client.completions(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )

        model_time = (datetime.now() - model_start_datetime).total_seconds()

        translation = result['completion']
        input_tokens = result['input_tokens']
        output_tokens = result['output_tokens']

        end_datetime = datetime.now()
        translation_time = (end_datetime - start_datetime).total_seconds()

        return {
            'translation': str(translation),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model_name': model_name,
            'model_time': model_time,
            'translation_time': translation_time,
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"Error: {e}")
        if not fail_on_error:
            raise e

    return {}


def run_translation_pipeline(source_file_path: str,
                             output_dir: str,
                             prompt_file_name: str,
                             model_name: str,
                             limit: int = 0,
                             force: bool = False,
                             parallel: bool = False,
                             sleep_time: int = 0,
                             **kwargs):
    # Determine the output file path
    output_file_path = get_output_file(source_file_path, output_dir, **kwargs).replace('.csv', '_translated.csv')
    print(f"Translation output file path: {output_file_path}")

    # Load the data
    file_path = output_file_path if os.path.exists(output_file_path) else source_file_path
    df = load_data(file_path, limit, force=force, ignore_populated_column='translation')

    # Get the ID columns
    id_columns = ['_id']
    if 'segment_id' in df.columns:
        id_columns.append('segment_id')

    # Get the batch data
    prompt_type = 'query' if re.fullmatch(r'.*queries.*.csv', os.path.basename(source_file_path)) else 'document'
    batch_data = get_prompts(prompt_file_name, prompt_type, df, id_columns, **kwargs)
    
    # Define the response format
    if kwargs.get('response_format') == 'UnifiedTranslation':
        response_format = UnifiedTranslation
    elif kwargs.get('response_format') == 'UnifiedSingleSentenceTranslation':
        response_format = UnifiedSingleSentenceTranslation
    else:
        response_format = Translation

    # Translate the batch data
    if parallel:
        df = translate_parallel(df, batch_data, model_name, response_format, prompt_file_name, id_columns, output_file_path, 
                                   num_workers=kwargs.get('num_workers', 0), 
                                   sleep_time=sleep_time)
    else:
        df = translate_serial(df, batch_data, model_name, response_format, prompt_file_name, id_columns, output_file_path, 
                                    sleep_time=sleep_time)

    return df


def translate_serial(source_df, batch_data, model_name, response_format, prompt_file_name, id_columns, output_file_path, sleep_time: int = 0):
    # Translate a batch of texts
    translation_datetime = datetime.now()
    for i, item in tqdm(enumerate(batch_data), desc="Rows", total=len(batch_data)):
        batch_datetime = datetime.now()
        
        # Translate batch item
        results = translate(system_prompt=item['system_prompt'], 
                            user_prompt=item['user_prompt'],
                            model_name=model_name,
                            response_format=response_format)

        translation = {
            **item,
            **results,
            "batch_idx": i,
            "batch_size": 1,
            "prompt_file_name": prompt_file_name,
            "batch_datetime": batch_datetime,
            "translation_datetime": translation_datetime,
        }
        
        # Save the results
        translated_df = pd.DataFrame([translation])
        df = pd.merge(source_df, translated_df, on=id_columns, how='left')
        # Overwrite columns in left_df with values from right_df if available
        for col in translated_df.columns:
            if (col + '_y') in df.columns and col not in id_columns:
                df[col] = df[col + '_y'].combine_first(df[col + '_x'])
                df.drop(columns=[col + '_x', col + '_y'], inplace=True)
                
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        translated_df.to_csv(output_file_path, encoding='utf-8', index=False)

        # Sleep between requests
        if sleep_time > 0:
            time.sleep(sleep_time)

    return translated_df


def translate_parallel(source_df, batch_data, model_name, response_format, prompt_file_name, id_columns, output_file_path, num_workers: int = 0, sleep_time: int = 0):
    """
    Translates a batch of data in parallel and saves results to a CSV file.

    Args:
        batch_data (list of dicts): List of batch items containing translation prompts.
        model_name (str): Name of the translation model.
        response_format (str): Format of the response.
        prompt_file_name (str): Filename associated with the prompts.
        id_columns (list): List of columns to use as identifiers.
        output_file_path (str): Path to save the translated results.
    
    Returns:
        pd.DataFrame: DataFrame containing the merged translation results.
    """
    # Determine the number of workers
    if num_workers == 0:
        num_workers = max(1, os.cpu_count() - 1)  # Default to CPU count - 1

    print(f"Translating {len(batch_data)} items in parallel using {num_workers} workers...")

    # Prepare arguments for parallel processing
    task_args = [(i, item, model_name, response_format, prompt_file_name, sleep_time) for i, item in enumerate(batch_data)]

    # Store translation results
    translated_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch translation tasks
        futures = {executor.submit(translate_batch_item, args): args[0] for args in task_args}

        # Collect results asynchronously
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Translations"):
            try:
                translated_results.append(future.result())
            except Exception as e:
                print(f"Error processing item {futures[future]}: {e}")

    # Convert results to DataFrame and merge
    if translated_results:
        translated_df = pd.DataFrame(translated_results)
        df = pd.merge(source_df, translated_df, on=id_columns, how='left')

        # Overwrite existing columns with new values
        for col in translated_df.columns:
            if (col + '_y') in df.columns and col not in id_columns:
                df[col] = df[col + '_y'].combine_first(df[col + '_x'])
                df.drop(columns=[col + '_x', col + '_y'], inplace=True)

        # Save results to CSV
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, encoding='utf-8', index=False)
        print(f"Translation results saved to {output_file_path}")

    return df


def translate_batch_item(args, **kwargs):
    """
    Translates a single batch item and returns the translation result.
    
    Args:
        args (tuple): Contains (index, item, model_name, response_format, prompt_file_name, id_columns)
    
    Returns:
        dict: Translation results with metadata
    """
    index, item, model_name, response_format, prompt_file_name, sleep_time = args
    batch_datetime = datetime.now()

    # Perform translation
    results = translate(system_prompt=item['system_prompt'], 
                        user_prompt=item['user_prompt'],
                        model_name=model_name,
                        response_format=response_format)

    # Construct translation dictionary
    translated_item = {
        **item,
        **results,
        "batch_idx": index,
        "batch_size": 1,
        "prompt_file_name": prompt_file_name,
        "batch_datetime": batch_datetime,
        "translation_datetime": datetime.now(),
    }

    if sleep_time > 0:
        time.sleep(kwargs['sleep_time'])

    return translated_item
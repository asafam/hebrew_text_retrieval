import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
from tqdm import tqdm
import os
import time
import re
from translation.api.utils import *

# Suppresses the per-row tqdm bars when set (e.g. for the ladder pipeline, which
# renders QA results in its own rich Table instead of progress bars).
_QUIET = os.environ.get("EVAL_TRANSLATIONS_QUIET", "").lower() in ("1", "true", "yes")


def evaluate_translation(
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature=0.7,
        response_format=TranslationCritique,
        fail_on_error=True,
        critique_key='critique',
        score_key='score'
    ):
    start_datetime = datetime.now()

    from llms.router import get_llm

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        model_start_datetime = datetime.now()
        llm = get_llm(model_name)
        result = llm.completions(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        model_time = (datetime.now() - model_start_datetime).total_seconds()
        completion_parsed = result["completion"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]

        end_datetime = datetime.now()
        translation_time = (end_datetime - start_datetime).total_seconds()

        return {
            critique_key: completion_parsed.critique,
            score_key: completion_parsed.score,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model_name': model_name,
            'model_time': model_time,
            'translation_time': translation_time,
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"Error: {e}")
        if fail_on_error:
            raise e

    return {}


def run_evaluate_translations(
        source_file_path: str,
        output_dir: str,
        gold_file_path: str,
        prompt_file_name: str,
        model_name: str,
        limit: int = 0,
        force: bool = False,
        parallel: bool = False,
        **kwargs
    ):
    # Determine the output file path
    translations_evaluation_output_file_path = get_output_file(source_file_path, output_dir, **kwargs).replace('.csv', '_evaluated.csv')

    # Load the data
    file_path = translations_evaluation_output_file_path if os.path.exists(translations_evaluation_output_file_path) else source_file_path
    critique_key = kwargs.get('critique_key', 'critique')
    score_key = kwargs.get('score_key', 'score')
    df = load_data(file_path=file_path, gold_file_path=gold_file_path, limit=limit, force=force, ignore_populated_column=critique_key, **kwargs)
    text_key = kwargs.get('text_key', 'text')
    translation_key = kwargs.get('translation_key', 'translation')
    gold_key = kwargs.get('gold_key')
    df['evaluated_translation'] = df.apply(lambda x: EvaluatedTranslation(x[text_key], x[translation_key], x[gold_key] if gold_key else None), axis=1)

    # Get the ID columns
    id_columns = ['_id']
    if 'segment_id' in df.columns:
        id_columns.append('segment_id')

    # Get the batch data. Caller may pass prompt_type explicitly (preferred); the
    # filename regex is only a fallback for legacy call sites that don't.
    prompt_type = kwargs.pop("prompt_type", None) or (
        "query" if re.fullmatch(r".*queries.*.csv", os.path.basename(source_file_path)) else "document"
    )
    batch_data = get_prompts(prompt_file_name, prompt_type, df, id_columns, **kwargs)
    
    # Define the response format
    response_format = TranslationCritique

    if parallel:
        df = evaluate_translation_parallel(
            df, batch_data, model_name, response_format, critique_key, score_key,
            prompt_file_name, translations_evaluation_output_file_path, id_columns,
            num_workers=kwargs.get("num_workers", 0),
            sleep_time=kwargs.get("sleep_time", 0),
        )
    else:
        df = evaluate_translations_serial(df, batch_data, model_name, response_format, critique_key, score_key, prompt_file_name, translations_evaluation_output_file_path, id_columns, sleep_time=kwargs.get('sleep_time', 0))
    
    return df


def evaluate_translations_serial(source_df, batch_data, model_name, response_format, critique_key, score_key, prompt_file_name, output_file_path, id_columns, sleep_time: int = 0):
    evaluation_datetime = datetime.now()
    evaluations = []

    _TRANSIENT_CODES = (429, 503, 504)

    for i, item in tqdm(enumerate(batch_data), desc="Rows", total=len(batch_data), disable=_QUIET):
        if sleep_time > 0 and i > 0:
            time.sleep(sleep_time)
        results = {}
        for attempt in range(3):
            try:
                results = evaluate_translation(
                    system_prompt=item['system_prompt'],
                    user_prompt=item['user_prompt'],
                    model_name=model_name,
                    response_format=response_format,
                    critique_key=critique_key,
                    score_key=score_key,
                    fail_on_error=True,
                )
                break
            except Exception as e:
                code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                is_transient = code in _TRANSIENT_CODES or any(
                    kw in str(e) for kw in ("DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "timeout")
                )
                if is_transient and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"\nTransient error (attempt {attempt+1}/3), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"\nSkipping row {i} after error: {e}")
                    break
        evaluations.append({
            **item,
            **results,
            "batch_idx": i,
            "batch_size": 1,
            "prompt_file_name": prompt_file_name,
            "batch_datetime": datetime.now(),
            "evaluation": evaluation_datetime,
        })

    evaluated_df = pd.DataFrame(evaluations)
    df = pd.merge(source_df, evaluated_df, on=id_columns, how='left')
    for col in evaluated_df.columns:
        if (col + '_y') in df.columns and col not in id_columns:
            df[col] = df[col + '_y'].combine_first(df[col + '_x'])
            df.drop(columns=[col + '_x', col + '_y'], inplace=True)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, encoding='utf-8', index=False)
    return df

def evaluate_translation_parallel(source_df, batch_data, model_name, response_format, critique_key, score_key, prompt_file_name, output_file_path, id_columns, num_workers: int = 0, sleep_time: int = 0):
    """
    Translates a batch of data in parallel and saves results to a CSV file.

    Args:
        batch_data (list of dicts): List of batch items containing translation prompts.
        model_name (str): Name of the translation model.
        response_format (str): Format of the response.
        prompt_file_name (str): Filename associated with the prompts.
        id_columns (list): List of columns to use as identifiers.
        output_file_path (str): Path to save the evaluated results.
    
    Returns:
        pd.DataFrame: DataFrame containing the merged translation results.
    """
    # Determine the number of workers
    if num_workers == 0:
        num_workers = max(1, os.cpu_count() - 1)  # Default to CPU count - 1

    print(f"Translating {len(batch_data)} items in parallel using {num_workers} workers...")

    # Prepare arguments for parallel processing
    task_args = [(i, item, model_name, response_format, critique_key, score_key, prompt_file_name, sleep_time) for i, item in enumerate(batch_data)]

    # Store translation results
    evaluated_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch translation tasks
        futures = {executor.submit(evaluate_translation_batch_item, args): args[0] for args in task_args}

        # Collect results asynchronously
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Translations", disable=_QUIET):
            try:
                evaluated_results.append(future.result())
            except Exception as e:
                print(f"Error processing item {futures[future]}: {e}")

    # Convert results to DataFrame and merge
    if not evaluated_results:
        # All workers failed — return source unchanged rather than crashing with
        # an UnboundLocalError, so the caller can detect the empty result.
        print(f"[eval] WARNING: no results produced for {output_file_path} "
              f"(all {len(batch_data)} items failed).")
        return source_df

    evaluated_df = pd.DataFrame(evaluated_results)
    df = pd.merge(source_df, evaluated_df, on=id_columns, how='left')

    # Overwrite existing columns with new values
    for col in evaluated_df.columns:
        if (col + '_y') in df.columns and col not in id_columns:
            df[col] = df[col + '_y'].combine_first(df[col + '_x'])
            df.drop(columns=[col + '_x', col + '_y'], inplace=True)

    # Save results to CSV
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, encoding='utf-8', index=False)

    return df


def evaluate_translation_batch_item(args, **kwargs):
    """
    Translates a single batch item and returns the translation result.
    
    Args:
        args (tuple): Contains (index, item, model_name, response_format, prompt_file_name, id_columns)
    
    Returns:
        dict: Translation results with metadata
    """
    index, item, model_name, response_format, critique_key, score_key, prompt_file_name, sleep_time = args
    batch_datetime = datetime.now()

    # Perform translation
    results = evaluate_translation(
        system_prompt=item['system_prompt'], 
        user_prompt=item['user_prompt'],
        model_name=model_name,
        response_format=response_format,
        critique_key=critique_key,
        score_key=score_key
    )

    # Construct translation dictionary
    evaluated_item = {
        **item,
        **results,
        "batch_idx": index,
        "batch_size": 1,
        "prompt_file_name": prompt_file_name,
        "batch_datetime": batch_datetime,
        "translation_datetime": datetime.now(),
    }

    if sleep_time > 0:
        time.sleep(sleep_time)

    return evaluated_item
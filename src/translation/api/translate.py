from typing import List 
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
from tqdm import tqdm
import os
from translation.utils import *


def translate(system_prompt: str,
              user_prompt: str,
              model_name: str,
              temperature=0.7):
    start_datetime = datetime.now()

    # Load the OpenAI client
    client = OpenAI(
        organization=os.environ['OPENAI_API_ORG'],
        api_key=os.environ['OPENAI_API_KEY'],
        project=os.environ['OPENAI_PROJECT']
    )    

    # Define the response format
    class Translation(BaseModel):
        hebrew: str
    response_format = Translation

    # Define the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Run chat completion inference
    model_start_datetime = datetime.now()
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format=response_format
    )
    model_time = (datetime.now() - model_start_datetime).total_seconds()

    translation = completion.choices[0].message.parsed.hebrew
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    end_datetime = datetime.now()
    translation_time = (end_datetime - start_datetime).total_seconds()

    return {
        'translation': translation,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model_name': model_name,
        'model_time': model_time,
        'translation_time': translation_time,
        'timestamp': datetime.now()
    }


def run_translation_pipeline(source_file_path: str,
                             prompt_file_name: str,
                             model_name: str,
                             limit: int = 0,
                             force: bool = False,
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

    # Load the prompt yaml file
    with open(prompt_file_name, 'r') as file:
        prompt_data = yaml.safe_load(file)
    prompt_type = 'query' if source_file_path.endswith('queries.csv') else 'document'
    prompt = prompt_data[prompt_type]
    prompt_meta_fields = {
        'english_key': kwargs.get('english_key', 'English'),
        'hebrew_key': kwargs.get('hebrew_key', 'Hebrew'),
        'context_key': kwargs.get('context_key', 'Context'),
    }
    system_prompt = prompt['system_prompt']
    user_prompt_prefix = prompt['user_prompt_prefix'].format_map(SafeDict(prompt_meta_fields))
    user_prompt_template = prompt['user_prompt_template'].format_map(SafeDict(prompt_meta_fields))

    # Create the batches
    id_columns = ['id', 'segment_id'] if 'segment_id' in filtered_df.columns else ['id']
    data = [{
            **{id: item[id] for id in id_columns},
            'dynamic_prompt': user_prompt_template.format(**item)
        } for item in filtered_df.to_dict(orient='records')]

    # Translate a batch of texts
    translation_datetime = datetime.now()
    for i, item in tqdm(enumerate(data), desc="Rows", total=len(data)):
        batch_datetime = datetime.now()

        # Prepare the batch for translation
        user_prompt = (user_prompt_prefix + '\n' + item['dynamic_prompt']).strip()
        
        # Translate batch
        results = translate(system_prompt=system_prompt, 
                            user_prompt=user_prompt,
                            model_name=model_name)

        translation = {
            **item,
            **prompt_meta_fields,
            **results,
            "prompt_prefix": user_prompt_prefix,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "batch_idx": i,
            "batch_size": 1,
            "prompt_file_name": prompt_file_name,
            "batch_datetime": batch_datetime,
            "translation_datetime": translation_datetime,
        }
        
        # Save the results
        batch_df = pd.DataFrame([translation])
        df = pd.merge(df, batch_df, on=id_columns, how='left')
        # Overwrite columns in left_df with values from right_df if available
        for col in batch_df.columns:
            if (col + '_y') in df.columns and col not in id_columns:  # Skip key columns
                df[col] = df[col + '_y'].combine_first(df[col + '_x'])
                df.drop(columns=[col + '_x', col + '_y'], inplace=True)
        os.makedirs(os.path.dirname(translation_output_file_path), exist_ok=True)
        df.to_csv(translation_output_file_path, encoding='utf-8', index=False)

    return df

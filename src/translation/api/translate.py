from typing import List 
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import yaml
from tqdm import tqdm
import os
import re
from translation.api.utils import *


class Translation(BaseModel):
    text: str
    translation: str

    def __str__(self):
        return self.translation
    
    def __repr__(self):
        return self.translation
    

def translate(system_prompt: str,
              user_prompt: str,
              model_name: str,
              temperature=0.7,
              response_format=Translation):
    start_datetime = datetime.now()

    # Load the OpenAI client
    client = OpenAI(
        organization=os.environ['OPENAI_API_ORG'],
        api_key=os.environ['OPENAI_API_KEY'],
        project=os.environ['OPENAI_PROJECT']
    )

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

    translation = completion.choices[0].message.parsed
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

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


def run_translation_pipeline(source_file_path: str,
                             prompt_file_name: str,
                             model_name: str,
                             limit: int = 0,
                             force: bool = False,
                             **kwargs):
    # Determine the output file path
    model_name_slug = model_name.replace('/', '_')
    file_path = os.path.dirname(source_file_path)
    file_name = os.path.basename(source_file_path)
    if kwargs.get('version') is not None:
        file_name = f"{file_name.split('.')[0]}_{kwargs['version']}.{file_name.split('.')[1]}"
    translation_output_file_path = os.path.join(file_path, model_name_slug, file_name)
    print(f"Translation output file path: {translation_output_file_path}")

    # Load the data
    file_path = translation_output_file_path if os.path.exists(translation_output_file_path) else source_file_path
    df = load_data(file_path, limit, force)

    # Get the ID columns
    id_columns = ['_id']
    if 'segment_id' in df.columns:
        id_columns.append('segment_id')

    # Get the batch data
    prompt_type = 'query' if re.fullmatch(r'.*queries.*.csv', os.path.basename(source_file_path)) else 'document'
    batch_data = get_prompts(prompt_file_name, prompt_type, df, id_columns, **kwargs)
    
    # Define the response format
    response_format = kwargs.get('response_format', Translation)

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

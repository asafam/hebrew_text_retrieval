from typing import List
from openai import OpenAI
import pandas as pd
import os
import yaml


class SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep unresolved placeholders
    

def load_data(file_path: str, limit: int, force: bool):
    df = pd.read_csv(file_path, encoding='utf-8')
    if limit > 0:
        print(f"Limiting the number of texts to {limit}.")
        df = df.head(limit)
    filtered_df = df[df['translation'].isnull()] if 'translation' in df.columns and not force else df

    # Check if the file has been fully translated
    if not force and filtered_df.empty:
        print(f"Skipping translation of {file_path} as have been translated.")
        return
    
    return filtered_df
    

def load_prompts(prompt_file_name: str, prompt_type: str, **kwargs):
    with open(prompt_file_name, 'r') as file:
        prompt_data = yaml.safe_load(file)
    
    prompt = prompt_data[prompt_type]
    prompt_meta_fields = get_prompt_meta_fields(**kwargs)
    system_prompt = prompt['system_prompt']
    user_prompt_prefix = prompt['user_prompt_prefix'].format_map(SafeDict(prompt_meta_fields))
    user_prompt_template = prompt['user_prompt_template'].format_map(SafeDict(prompt_meta_fields))

    return system_prompt, user_prompt_prefix, user_prompt_template


def get_openai_client():
    client = OpenAI(
        organization=os.environ['OPENAI_API_ORG'],
        api_key=os.environ['OPENAI_API_KEY'],
        project=os.environ['OPENAI_PROJECT']
    )
    return client


def get_prompts(prompt_file_name: str, prompt_type: str, df: pd.DataFrame, id_columns: List[str], **kwargs):
    system_prompt, user_prompt_prefix, user_prompt_template = load_prompts(prompt_file_name, prompt_type, **kwargs)
    
    # Create the candidates for translation
    prompt_meta_fields = get_prompt_meta_fields(**kwargs)

    prompts = [
        {
            **{id: record[id] for id in id_columns},
            **prompt_meta_fields,
            'system_prompt': system_prompt,
            'user_prompt': (user_prompt_prefix + '\n' + user_prompt_template.format(**record)).strip(),
            'prompt_prefix': user_prompt_prefix,
            'prompt_file_name': prompt_file_name,
        } for i, record in enumerate(df.to_dict(orient='records'))
    ]
    return prompts


def get_prompt_meta_fields(**kwargs):
    prompt_meta_fields = {
        'english_key': kwargs.get('english_key', 'English'),
        'hebrew_key': kwargs.get('hebrew_key', 'Hebrew'),
        'context_key': kwargs.get('context_key', 'Context'),
        'hebrew_key_query': kwargs.get('hebrew_key_query', 'Hebrew Query'),
        'hebrew_key_document': kwargs.get('hebrew_key_document', 'Hebrew Document'),
    }
    return prompt_meta_fields


def get_translation_output_file(source_file_path, model_name, **kwargs):
    model_name_slug = model_name.replace('/', '_')
    file_path = os.path.dirname(source_file_path)
    file_name = os.path.basename(source_file_path)
    if kwargs.get('version') is not None:
        file_name = f"{file_name.split('.')[0]}_{kwargs['version']}.{file_name.split('.')[1]}"
        print(file_name)
    translation_output_file_path = os.path.join(file_path, model_name_slug, file_name)
    return translation_output_file_path
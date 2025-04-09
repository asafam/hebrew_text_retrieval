from typing import List, Optional
from openai import OpenAI
import pandas as pd
import os
import yaml
from pydantic import BaseModel, Field


class EvaluatedTranslation():
    english: str
    hebrew: str
    gold: str

    def __init__(self, english: str, hebrew: str, gold: str):
        self.english = english
        self.hebrew = hebrew
        self.gold = gold

    def __str__(self):
        return f"English: {self.english}\nHebrew: {self.hebrew}\nGold: {self.gold}"


class TranslationCritique(BaseModel):
    critique: str
    score: int


class Translation(BaseModel):
    text: str = Field(description="The text to be translated.")
    translation: str = Field(description="The translated text.")

    def __str__(self):
        return self.translation
    
    def __repr__(self):
        return self.translation
    

class UnifiedTranslation(BaseModel):
    text1: str
    text2: str
    translation1: str
    translation2: str

    def get_document_translation(self):
        return self.translation1
    
    def get_query_translation(self):
        return self.translation2

    def __str__(self):
        return "<hebrew_document>" + self.get_document_translation() + "</hebrew_document><hebrew_query>" + self.get_query_translation() + "</hebrew_query>"
    
    def __repr__(self):
        return "<hebrew_document>" + self.get_document_translation() + "</hebrew_document><hebrew_query>" + self.get_query_translation() + "</hebrew_query>"


class UnifiedSingleSentenceTranslation(UnifiedTranslation):
    text: str
    translation: str

    def get_document_translation(self):
        return self.translation.split('###')[0]
    
    def get_query_translation(self):
        return self.translation.split('###')[1]


class SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep unresolved placeholders
    

def load_data(file_path: str, 
              limit: int, 
              force: bool, 
              ignore_populated_column: str,
              gold_file_path: Optional[str] = None, 
              **kwargs):
    df = pd.read_csv(file_path, encoding='utf-8')
    if limit > 0:
        print(f"Limiting the number of texts to {limit}.")
        df = df.head(limit)
    filtered_df = df[df[ignore_populated_column].isnull()] if ignore_populated_column in df.columns and not force else df

    # Check if the file has been fully translated
    if not force and filtered_df.empty:
        print(f"Skipping {file_path} as have been processed.")
        return
    
    if gold_file_path:
        gold_df = pd.read_csv(gold_file_path, encoding='utf-8')
        id_columns = kwargs.get('id_columns', ['_id', 'context_id'])
        translation_column = kwargs.get('translation_key', 'translation')
        gold_translation_column = kwargs.get('gold_key', 'gold_translation')
        golf_df_selected = gold_df[id_columns + [translation_column]]
        golf_df_selected = golf_df_selected.rename(columns={translation_column: gold_translation_column})
        filtered_df = filtered_df.merge(golf_df_selected, on=id_columns, how='left')
    
    return filtered_df
    

def load_prompts(prompt_file_name: str, prompt_type: Optional[str], **kwargs):
    with open(prompt_file_name, 'r') as file:
        prompt_data = yaml.safe_load(file)
    
    prompt = prompt_data[prompt_type] if prompt_type else prompt_data
    prompt_meta_fields = get_prompt_meta_fields(**kwargs)
    system_prompt = prompt['system_prompt']
    user_prompt_prefix = prompt.get('user_prompt_prefix', "").format_map(SafeDict(prompt_meta_fields))
    user_prompt_template = prompt.get('user_prompt_template').format_map(SafeDict(prompt_meta_fields))

    return system_prompt, user_prompt_prefix, user_prompt_template


def get_openai_client():
    client = OpenAI(
        organization=os.environ['OPENAI_API_ORG'],
        api_key=os.environ['OPENAI_API_KEY'],
        project=os.environ['OPENAI_PROJECT']
    )
    return client


def get_prompts(prompt_file_name: str, prompt_type: Optional[str], df: pd.DataFrame, id_columns: List[str], **kwargs):
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


def get_output_file(source_file_path, output_dir, **kwargs):
    """
    Return the output file path for the translation.
    """
    file_name = os.path.basename(source_file_path)
    output_file_path = os.path.join(output_dir, file_name)
    return output_file_path
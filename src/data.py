from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import logging
import pickle
from pathlib import Path
from tqdm import tqdm
import random

def transform_dataset_wiki40b(splits=['train', 'validation', 'test']):
    logger = logging.getLogger('default')
    logger.info("Transforming Wiki40B dataset")

    # Load the Wiki40B dataset
    logger.info("Loading Wiki40B dataset")
    dataset = load_dataset("wiki40b", "he")
    decoded_dataset = dataset.map(lambda x: {'text': decode_text(x['text'])})

    def transform_entry(entry):
        # Process the 'text' using parse_wiki_article
        article = parse_wiki_article(entry['text'])

        # Extract anchor_text and positive_text based on the parsed output
        anchor_text = article['title']
        if 'sections' in article and len(article['sections']) > 0:
            anchor_text += " " + article['sections'][0]['section']
            positive_text = article['sections'][0]['paragraphs'][0]
        else:
            positive_text = article['abstract'][0]

        # Return the transformed data
        return {
            'anchor_text': 'query : ' + anchor_text,
            'positive_text': 'document: ' + positive_text,
        }

    # Apply the transformation to the train, validation, and test splits
    transformed_dataset = {}
    for split in splits:
        # Transform each subset of the dataset using map (this processes each 'text' entry)
        logger.info(f"Transforming {split} split")
        transformed_split = decoded_dataset[split].map(transform_entry)
        transformed_dataset[split] = transformed_split

    # Return the transformed dataset as a DatasetDict
    logger.info("Done transforming Wiki40B dataset")
    return DatasetDict(transformed_dataset)


def decode_text(text):
    decoded_text = bytes(text, "utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
    return decoded_text


def parse_wiki_article(text):
    lines = text.strip().split('\n')

    PARAGRAPH_DIVIDER = '_NEWLINE_'

    # Initialize variables
    article_dict = {'title': '', 'abstract': '', 'sections': []}
    current_section = None
    abstract_parsed = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "_START_ARTICLE_":
            # The next line is the title
            article_dict['title'] = lines[i + 1].strip()
            i += 2  # Move to the next relevant line
        elif line == "_START_PARAGRAPH_":
            # If the abstract has not been parsed and the current section is None, this is the abstract
            paragraph = lines[i + 1].strip()
            if not abstract_parsed and not current_section:
                article_dict['abstract'] = paragraph.split(PARAGRAPH_DIVIDER)
                abstract_parsed = True
            elif current_section:
                current_section['paragraphs'] = paragraph.split(PARAGRAPH_DIVIDER)
            i += 2
        elif line == "_START_SECTION_":
            # The next line is the section name
            section_name = lines[i + 1].strip()
            current_section = {'section': section_name, 'paragraphs': ''}
            article_dict['sections'].append(current_section)
            i += 2
        else:
            i += 1  # Move to the next line if none of the cases match

    return article_dict


def transform_dataset_synthesized(data_folder_path, test_size=0.2):
    logger = logging.getLogger('default')
    logger.info("Transforming synthesized dataset")

    logger.info("Loading synthesize query document dataset from {data_folder_path}")
    data = _load_synthesized_data_files(data_folder_path=data_folder_path)

    def transform_entry(entry):
        # Return the transformed data
        return {
            'anchor_text': 'query: ' + entry['user_query'],
            'positive_text': 'document: ' + entry['positive_document'],
            'negative_text': 'document: ' + entry['hard_negative_document'],
        }

    # Apply the transformation to each entry
    transformed_data = list(map(transform_entry, data))

    # Convert the list of dictionaries to a Hugging Face Dataset
    dataset = Dataset.from_list(transformed_data)

    # Split the dataset into train and test sets
    train_test_dataset = dataset.train_test_split(test_size=test_size)
    # Further split the test set into validation and test
    test_validation_dataset = train_test_dataset['test'].train_test_split(test_size=0.5)  # 50% of 'test' for 'validation'
    train_validation_dataset = DatasetDict({
        'train': train_test_dataset['train'],  # Keep the 'train' split
        'test': test_validation_dataset['test'],  # The remaining 'test' set
        'validation': test_validation_dataset['train'],  # The remaining 'test' set
    })

    logger.info("Done transforming synthesized dataset")
    return train_validation_dataset


def transform_dataset(dataset_name, **kwargs):
    if dataset_name == 'wiki40b':
        return transform_dataset_wiki40b(**kwargs)
    elif dataset_name == 'synthesized_query_document':
        return transform_dataset_synthesized(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    

def unify_datasets(datasets, columns = ['anchor_text', 'positive_text', 'negative_text']):
    # Create an empty DatasetDict to store the concatenated splits
    unified_dataset = DatasetDict()

    # Iterate over datasets and train
    for dataset in datasets:   
        for split in dataset.keys():
            dataset_split = dataset[split].select_columns(columns)
            if split not in dataset:
                unified_dataset[split] = dataset_split
            else:
                unified_dataset[split] = concatenate_datasets([dataset[split], dataset_split])
    
    return unified_dataset
    

def _load_synthesized_data_files(data_folder_path):
    logger = logging.getLogger('default')
    data = []

    # Method to use pathlib to find all .pkl files
    data_folder = Path(data_folder_path)
    files = [file_path for file_path in data_folder.glob('*.pkl')]

    # Sort the files by name (optional, if you want a specific order)
    files.sort(key=lambda x: x.name)

    # Load and concatenate data from each file
    for file_path in tqdm(files, desc="Loading data files"):
        logger.debug(f"Loading data from {file_path.name}")
        with file_path.open('rb') as f:
            file_data = pickle.load(f)
            if isinstance(data, list):
                data.extend(file_data)
            else:
                logger.debug(f"Warning: {file_path.name} does not contain a list.")

    data = [item for item in data if item['success']]
    
    return data


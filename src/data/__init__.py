from enum import Enum
import os
import importlib.util
import inspect
import logging

# Define special tokens
DOCUMENT_TOKEN = '[DOCUMENT]'
QUERY_TOKEN = '[QUERY]'
TASK_TOKENS = dict(
    TASK_QUERY_DOC='[TASK_QUERY_DOC]',
    TASK_TITLE_DOC='[TASK_TITLE_DOC]',
    TASK_QUESTION_DOC='[TASK_QUESTION_DOC]',
)


class DatasetName(Enum):
    WIKI40B = 'wiki40b'
    SYNTHESIZED_QUERY_DOCUMENT = 'synthesized_query_document'
    HEQ = 'heq'


class BaseDatasetBuilder():
    def build_dataset(self, **kwargs):
        raise NotImplementedError()
    
    def build_eval_dataset(self, split='test', **kwargs):
        raise NotImplementedError()
    
    def is_match(self, dataset_name: str) -> bool:
        raise NotImplementedError()
    
    
def build_dataset(dataset_name: str, **kwargs):
    logger = logging.getLogger('default')

    builder = _get_builder(dataset_name)
    dataset = builder.build_dataset(**kwargs)
    return dataset


def build_eval_dataset(dataset_name: str, **kwargs):
    logger = logging.getLogger('default')

    builder = _get_builder(dataset_name)
    dataset = builder.build_eval_dataset(**kwargs)
    return dataset


def _get_builder(dataset_name: str) -> BaseDatasetBuilder:
    logger = logging.getLogger('default')
    
    folder_path = os.path.dirname(os.path.abspath(__file__))
    # Get all Python files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.py') and f != '__init__.py']

    # List to hold instantiated classes
    builders = []

    # Loop through the files and load classes that are BasePromptBuilder or extend it
    for file in files:
        module_path = os.path.join(folder_path, file)

        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(file[:-3], module_path)  # Remove .py extension
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Iterate over all members of the module to find classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class is BasePromptBuilder or extends it
            if issubclass(obj, BaseDatasetBuilder) and name != 'BaseDatasetBuilder':
                try:
                    # Instantiate the class with the given prompt_type
                    builder = obj()
                    builders.append(builder)
                except Exception as e:
                    logger.error(f"Failed to instantiate {name}: {e}")

    # Filter classes by whether they match the prompt_type
    matching_builders = [b for b in builders if b.is_match(dataset_name)]

    if not matching_builders:
        return None

    builder = matching_builders[0]
    return builder


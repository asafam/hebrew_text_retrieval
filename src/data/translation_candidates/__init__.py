import os
import importlib
import inspect
import glob

from data.datasets.utils import hash_text

class TranslationCandidatesDataBuilder:
    def build_data(self, 
                   dataset_name: str, 
                   model_name_or_path: str, 
                   tokenizer_name_or_path: str,
                   n: int = 0, 
                   split: str = 'test',
                   id_field: str = 'id',
                   query_field: str = 'query',
                   context_field: str = 'text',
                   max_tokens: int = 2048, 
                   random_state: int = 42):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def is_match(self, dataset_name: str) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def hash_text(self, example, text_field, hash_field="hash"):
        example[hash_field] = hash_text(example[text_field])
        return example
    

def build_data(dataset_name: str, **kwargs):
    builder = _get_builder(dataset_name)
    dataset = builder.build_data(dataset_name=dataset_name, **kwargs)
    return dataset


def _get_builder(dataset_name: str) -> TranslationCandidatesDataBuilder:
    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Get all Python files in the folder
    print(f"Looking for Builder Python files in {folder_path}")
    files = [f for f in glob.glob(f"{folder_path}/**/*.py")]

    # List to hold instantiated classes
    builders = []

    # Loop through the files and load classes that inherit from BeIRDatasetLoader or extend it
    for file in files:
        module_path = os.path.join(folder_path, file)

        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(file[:-3], module_path)  # Remove .py extension
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Iterate over all members of the module to find classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class is BeIRDatasetLoader or extends it
            if issubclass(obj, TranslationCandidatesDataBuilder) and obj != TranslationCandidatesDataBuilder:
                builder = obj()
                builders.append(builder)

    # Filter classes by whether they match the prompt_type
    matching_builders = [l for l in builders if l.is_match(dataset_name)]

    if not matching_builders:
        return None

    builder = matching_builders[0]
    return builder
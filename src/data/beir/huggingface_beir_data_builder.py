
import yaml
from data.beir import BaseBeIRDataBuilder


class HuggingFaceBeIRDataBuilder(BaseBeIRDataBuilder):
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.startswith('BeIR/')
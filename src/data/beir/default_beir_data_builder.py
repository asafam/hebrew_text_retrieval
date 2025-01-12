
import yaml
from data.beir import BaseBeIRDataBuilder


class DefaultBeIRDataBuilder(BaseBeIRDataBuilder):
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name in ['BeIR/msmarco']
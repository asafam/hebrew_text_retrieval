import os
from datasets import Dataset, DatasetDict
from data import BaseDatasetBuilder

class SquadV2DatasetBuilder(BaseDatasetBuilder):
    def __init__(self,
                 queries_base_path: str,
                 query_field: str = 'question',
                 document_field: str = 'context',
                 **kwargs):
        super().__init__(**kwargs)
        self.queries_base_path = queries_base_path
        self.query_field = query_field
        self.document_field = document_field
        
    def build_dataset(self, splits=['train', 'test', 'validation'], random_state: int = 42, **kwargs):
        datasets = {}
        for split in splits:
            # Validate the split
            if split not in ['train', 'test', 'validation']:
                raise ValueError(f"Unsupported split: {split}. Supported splits are: 'train', 'test', 'validation'.")
            
            # Construct the path to the queries file
            queries_file = os.path.join(self.queries_base_path, 
                                        "validation" if split == "test" else split, # Handle 'test' as 'validation'
                                        'queries.jsonl')
            print(f"Loading {split} queries from: {queries_file}")
            queries = Dataset.from_json(queries_file)

            if split in ['test', 'validation']:
                val_split = queries.train_test_split(test_size=0.5, seed=random_state)
                datasets[split] = Dataset.from_dict({
                    self.query_field: [q[self.query_field] for q in val_split["test" if split == 'test' else "train"]],
                    self.document_field: [q[self.document_field] for q in val_split["test" if split == 'test' else "train"]],
                })
            else:
                datasets[split] = Dataset.from_dict({
                    self.query_field: [q[self.query_field] for q in queries],
                    self.document_field: [q[self.document_field] for q in queries],
                })

        return DatasetDict(datasets)
    
    def build_eval_dataset(self, split='test', **kwargs):
        return self.build_dataset(splits=[split], **kwargs)
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name.lower() == 'squad_v2'
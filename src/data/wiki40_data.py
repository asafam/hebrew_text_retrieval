from datasets import Dataset, DatasetDict, load_dataset
import logging
from data import *

class Wiki40bDatasetBuilder(BaseDatasetBuilder):
    def build_dataset(self, splits=['train', 'validation', 'test'], include_task_token: bool = True):
        logger = logging.getLogger('default')
        logger.info("Building Wiki40B dataset")

        # Load the Wiki40B dataset
        logger.info("Loading Wiki40B dataset")
        dataset = load_dataset("wiki40b", "he")
        decoded_dataset = dataset.map(lambda x: {'text': self._decode_text(x['text'])})

        def transform_entry(entry):
            # Process the 'text' using parse_wiki_article
            article = self._parse_wiki_article(entry['text'])

            # Extract anchor_text and positive_text based on the parsed output
            anchor_text = article['title']
            if 'sections' in article and len(article['sections']) > 0:
                anchor_text += " " + article['sections'][0]['section']
                positive_text = article['sections'][0]['paragraphs'][0]
            else:
                positive_text = article['abstract'][0]

            # Return the transformed data
            return {
                'anchor_text': f"{TASK_TOKENS['TASK_TITLE_DOC']} {QUERY_TOKEN} {anchor_text}" if include_task_token else f"{QUERY_TOKEN} {anchor_text}",
                'positive_text': f"{DOCUMENT_TOKEN} {positive_text}",
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
    
    def build_eval_dataset(self, split='test', random_seed: int = 42):
        logger = logging.getLogger('default')
        logger.info("Building Synthesized Query Document evaluation dataset")

        tasks_datasets = {
            'TASK_TITLE_DOC': self.build_dataset(splits=[split], include_task_token=False)[split]
        }
        return tasks_datasets
    
    def is_match(self, dataset_name: str) -> bool:
        return dataset_name == DatasetName.WIKI40B.value

    def _decode_text(self, text):
        decoded_text = bytes(text, "utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
        return decoded_text

    def _parse_wiki_article(self, text):
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



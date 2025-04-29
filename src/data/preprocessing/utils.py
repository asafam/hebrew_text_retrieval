from typing import List, Dict, Optional
import os
from pathlib import Path
from streaming import MDSWriter
from datasets import Dataset
import sentencepiece as spm
from tqdm import tqdm

# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="outputs/tokenizer/HebrewModernBERT_mixed_1M_100K.model", 
#     unk_token='[UNK]',
#     pad_token='[PAD]',
#     cls_token='[CLS]',
#     sep_token='[SEP]',
#     mask_token='[MASK]'
# )
tokenizer = spm.SentencePieceProcessor(model_file="outputs/tokenizer/HebrewModernBERT_mixed_1M_100K.model")

def tokenize(text: str) -> List[int]:
    return tokenizer.encode(text)

def save_as_mds(dataset: Dataset,
                columns: Dict[str, str], 
                output_dir: str, 
                shard_size_limit: int, 
                compression: Optional[int] = None):
    print(f"ℹ️ Iterating over a dataset streams into MDS format")
    print(f"Output directory: {output_dir}")
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True) # Ensure the output_dir exists
    with MDSWriter(out=output_dir, columns=columns, size_limit=shard_size_limit, compression=compression) as writer:
        for record in tqdm(dataset, desc="Writing records to MDS..."):
            # Write the next record to MDS
            writer.write(record)

    print(f"✅ Data saved in MDS format in {output_dir}")


def save_as_txt(dataset: Dataset,
                column: str,
                output_file: str):
    print(f"ℹ️ Iterating over a dataset streams into TXT format")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True) # Ensure the output_dir exists
    with open(os.path.join(output_file, "dataset.txt"), "w", encoding="utf-8") as out_f:
        for record in dataset:
            out_f.write(record[column] + "\n")

    print(f"✅ Data saved in TXT format in {output_file}")

from typing import Union
import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import argparse
import os
from pathlib import Path

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

def main(
    corpus: str,
    vocab_size: int,
    output_dir: str = 'tokenizer',
    force: bool = False
) -> Tokenizer:
    # Check if corpus file exists
    if not os.path.exists(corpus):
        raise FileNotFoundError(f"Corpus file not found: {corpus}")
    
    if os.path.exists(output_dir) and not force:
        print(f"⚠️ {output_dir} already exists. Use a different output directory.")
        return
    
    # Create output directory if it doesn't exist
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    spm.SentencePieceTrainer.Train(
        input=corpus,  # .txt input file
        model_prefix=output_dir,  
        vocab_size=vocab_size,  
        model_type='bpe',  # Byte-Pair Encoding
        character_coverage=0.9995,  
        num_threads=4,  
        pad_id=0,  # Index for [PAD]
        unk_id=1,  # Index for [UNK]
        bos_id=-1,  # No beginning-of-sequence token
        eos_id=-1,  # No end-of-sequence token
        user_defined_symbols=['[CLS]', '[SEP]', '[MASK]']  # Special tokens
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='data/corpus.txt')
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--output_dir', type=str, default='tokenizer')
    args = parser.parse_args()
    main(
        corpus=args.corpus, 
        vocab_size=args.vocab_size,
        output_dir=args.output_dir
    )  # Train tokenizer with vocabulary
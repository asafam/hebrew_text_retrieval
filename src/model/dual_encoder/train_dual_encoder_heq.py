import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, AutoConfig, PreTrainedModel
import torch
from torch import nn
import torch.nn.functional as F
from data.heq.heq_data import HeQDatasetBuilder, HeQTaskName
from model.dual_encoder.models import InfoNCEDualEncoder

def get_dataset():
    heq_dataset_builder = HeQDatasetBuilder(task=HeQTaskName.QUESTION_DOC, decorate_with_task_tokens=False)
    heq_dataset = heq_dataset_builder.build_dataset(filter_empty_answers=True)
    return heq_dataset

def preprocess(
        example,
        tokenizer_q,
        tokenizer_d,
        query='question', 
        paragraph='context', 
        truncation=True, 
        padding="max_length", 
        max_length=1024
    ):
    q = tokenizer_q(
        example[query], truncation=truncation, padding=padding, max_length=max_length
    )
    d = tokenizer_d(
        example[paragraph], truncation=truncation, padding=padding, max_length=max_length
    )
    return {
        "q_input_ids": q['input_ids'],
        "q_attention_mask": q['attention_mask'],
        "d_input_ids": d['input_ids'],
        "d_attention_mask": d['attention_mask'],
    }

def collate_fn(batch):
    return {
        "q_input_ids": torch.tensor([item["q_input_ids"] for item in batch]),
        "q_attention_mask": torch.tensor([item["q_attention_mask"] for item in batch]),
        "d_input_ids": torch.tensor([item["d_input_ids"] for item in batch]),
        "d_attention_mask": torch.tensor([item["d_attention_mask"] for item in batch]),
    }

def main(
    query_model_name: str,
    doc_model_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    learning_rate=2e-5,
    remove_unused_columns=False,    
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    max_length=1024
):
    tokenizer_q = AutoTokenizer.from_pretrained(query_model_name)
    tokenizer_d = AutoTokenizer.from_pretrained(doc_model_name)
    
    heq_dataset = get_dataset()
    processed = heq_dataset.map(lambda sample: preprocess(sample, tokenizer_q, tokenizer_d, max_length=max_length))

    config = AutoConfig.from_pretrained(query_model_name)
    model = InfoNCEDualEncoder(config, query_model_name, doc_model_name, pooling='cls')

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        remove_unused_columns=remove_unused_columns,    
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed['train'],
        eval_dataset=processed['validation'],
        data_collator=collate_fn,
    )

    trainer.train()

if __name__ == "__main__":
    argparse.ArgumentParser(description="Train a dual encoder model with InfoNCE loss on HeQ dataset.")
    parser = argparse.ArgumentParser(description="Train a dual encoder model with InfoNCE loss on HeQ dataset.")
    parser.add_argument("--query_model_name", type=str, default="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250522_1841", help="Query model name or path")
    parser.add_argument("--doc_model_name", type=str, default="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250522_1841", help="Document model name or path (optional, defaults to query model if not provided)"),
    parser.add_argument("--output_dir", type=str, default="./outputs/models/dual_encoder/dual_encoder_infonce_heq", help="Output directory for model and logs")    
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--remove_unused_columns", action='store_true', help="Remove unused columns in the dataset")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every X steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization")
    args = parser.parse_args()
    
    main(
        query_model_name=args.query_model_name,
        doc_model_name=args.doc_model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        remove_unused_columns=args.remove_unused_columns,    
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        max_length=args.max_length
    )
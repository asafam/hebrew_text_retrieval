# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hebrew text information retrieval system combining dense passage retrieval (transformer embeddings), multi-LLM translation pipelines, and tokenizer training. The project trains retrieval models for Hebrew using contrastive learning with InfoNCE loss.

## Environment Setup

```bash
conda activate biu   # or 'htr'
export PYTHONPATH="./src:$PYTHONPATH"
```

Requires a `.env` file with API keys:
```
OPENAI_API_ORG, OPENAI_API_KEY, OPENAI_PROJECT
GEMINI_API_KEY, GEMINI_PROJECT
TOGETHER_API_KEY
ANTHROPIC_API_KEY
```

## Common Commands

**Train a retrieval model:**
```bash
python src/model/train_model.py \
    --model_name onlplab/alephbert-base \
    --task_name fact_passage \
    --dataset_name heq \
    --batch_size 32 \
    --epochs 10
```

See `scripts/model/train/` for configurations with specific hyperparameters (18+ templates covering AlephBERT, DictaBERT, E5-small across different datasets).

**Run translation pipeline:**
```bash
python src/translation/api/run_translation_pipeline.py \
    --source_file_paths <csv_file> \
    --output_dir <dir> \
    --prompt_file_name <yaml_file> \
    --model_name gpt-4o \
    --english_key "English" \
    --hebrew_key "Hebrew"
```

**Train tokenizer:**
```bash
python src/tokenizer/train_tokenizer.py
```

**Evaluate tokenizer:**
```bash
bash scripts/tokenizer/eval_tokenizer.sh
```

There is no test suite — validation is done via scripts and notebooks.

## Architecture

### Key Components

**`src/data/`** — Dataset loading with a factory pattern. `build_dataset(dataset_name)` in `__init__.py` auto-discovers builders extending `BaseDatasetBuilder`. Supported datasets: `wiki40b`, `heq`, `synthesized_query_document`, `heq_fact_passage_syn`. Builders output `(query, positive, negative)` triplets with task tokens like `[fact_passage_task]`, `[title_passage_task]`, plus `query:` / `passage:` prefixes.

**`src/model/`** — Training and evaluation:
- `train_model.py` — CLI entry point
- `trainings.py` — Training loop, gradient clipping, checkpoint management
- `evaluate.py` — Generates embeddings, builds FAISS index, computes Precision@k / MRR / NDCG@k
- `loss.py` — InfoNCE (contrastive) loss
- `utils.py` — Tokenization, checkpoint I/O, logging setup

**`src/llms/`** — LLM router pattern. `router.py:get_llm(model_name)` dispatches to `openai.py`, `google.py`, or `together_ai.py` based on regex matching on the model name.

**`src/translation/api/`** — Translation pipelines using LLM APIs. Structured outputs via Pydantic models (`data_structures/`). Supports parallel translation with `concurrent.futures`. Prompt templates in `prompts/translation/` (YAML).

**`src/tokenizer/`** — SentencePiece BPE tokenizer training and evaluation.

### Training Pipeline Flow

```
build_dataset() → tokenize_inputs_and_create_dataloader()
    → train loop: forward(query, positive, optional_negative) → InfoNCE loss → backward
    → validate every epoch → save_best_checkpoint()
```

Checkpoints saved to `checkpoints/{model_slug}/checkpoints_{task_slug}/checkpoint_epoch_{N}.pth`. Training resumes automatically from the latest checkpoint.

### Embedding Strategy

CLS token `[0]` is used as the sentence embedding. Models are fine-tuned with batch negatives (in-batch negatives from other queries serve as implicit negatives).

### Logs

Written to `./logs/{model_slug}/train_{task_slug}.log` at DEBUG level.

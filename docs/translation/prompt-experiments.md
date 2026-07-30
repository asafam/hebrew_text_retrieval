# Translations

Decision guide for translating BeIR queries and documents to Hebrew.

---

## How it works

All three pipeline scripts are **config-driven**. A config file (e.g. `config/experiments/translation_eval.sh`) declares all variables — datasets, models, prompts, limits — and each script sources it at startup.

```
config/experiments/translation_eval.sh   ← full run (12 datasets × 4 prompts × 3 models × 7 judges)
config/experiments/translation_eval_dryrun.sh     ← smoke test (2 datasets, 1 prompt, 1 model, 10 rows)
```

Pass a config as the first argument, or omit it to use the baseline default:

```bash
bash scripts/translation/run_translation_pipeline.sh                          # uses translation_eval.sh
bash scripts/translation/run_translation_pipeline.sh config/experiments/translation_eval_dryrun.sh
```

There is **one translation prompt** for everything. The `query:` vs `document:` section is selected automatically based on the source filename — files containing "queries" in the name use the query section; all others use the document section.

---

## Full experiment workflow

```
Step 1 — Build candidates
Step 2 — Translate  (all prompt strategies × all models)
Step 3 — Evaluate   (all judge models, category-specific prompts)
          └── collect_results.py + analyze_results.py run automatically after Step 3
```

### Step 1 — Build candidates

```bash
bash scripts/data/run_build_translation_candidates.sh
# or with a specific config:
bash scripts/data/run_build_translation_candidates.sh config/experiments/translation_eval.sh
```

Samples 100 queries + 100 documents per dataset across all 12 BeIR datasets.
All documents are segmented at 256 tokens. Short documents produce one segment (`segment_id=0`).

**Continuation:** already-completed datasets are skipped automatically. Re-run the same command after a crash and it picks up where it left off. Set `FORCE=true` in the config to rebuild everything.

**Parallelism:** set `WORKERS=4` in the config file — the script passes it as `--workers 4` to Python, which builds N datasets simultaneously. Datasets are independent so this is safe. When running Python directly, pass `--workers 4` explicitly.

```
outputs/translation/BeIR/candidates/
└── <dataset_slug>/
    ├── queries.csv      (_id, text, context_id, context_text, category, dataset_name)
    └── documents.csv    (_id, text, segment_id, segment_text, category, dataset_name)
```

### Step 2 — Translate

```bash
bash scripts/translation/run_translation_pipeline.sh
# or with a specific config:
bash scripts/translation/run_translation_pipeline.sh config/experiments/translation_eval.sh
```

Translates both queries and documents for every (dataset × translation model × prompt strategy).
The four prompt strategies tested cover the main ablation axes:

| Slug | Examples | Context | Notes |
|---|---|---|---|
| `fewshot` | yes | yes | Baseline |
| `zeroshot` | no | yes | Ablation: no examples |
| `fewshot_nocontext` | yes | no | Ablation: no context |
| `fewshot_searchopt` | yes | yes | Variant: search-keyword retention |

```
candidates/<dataset_slug>/<translation_model>/<prompt_slug>/
    ├── queries_translated.csv
    └── documents_translated.csv
```

### Step 3 — Evaluate + Analyze

```bash
bash scripts/translation/run_eval_translation_api_pipeline.sh
# or with a specific config:
bash scripts/translation/run_eval_translation_api_pipeline.sh config/experiments/translation_eval.sh
```

Scores each translation 0–5 with all judge models. Eval prompt is selected automatically by dataset category.
After all evaluations complete, automatically runs `collect_results.py` and `analyze_results.py`.

```
candidates/<dataset_slug>/<translation_model>/<prompt_slug>/evaluations/<judge_model>/
    ├── queries_translated_evaluated.csv
    └── documents_translated_evaluated.csv

outputs/translation/BeIR/
    ├── results_translation_eval.csv   ← all results in one file, all factors as columns
    └── analysis_translation_eval/
        ├── analysis_report.md      ← ranked tables for every factor
        ├── by_translation_model.png
        ├── by_prompt_slug.png
        ├── by_category.png
        ├── by_text_type.png
        ├── by_text_length.png
        ├── by_judge_model.png
        ├── heatmap_category_model.png
        └── heatmap_category_prompt.png
```

---

## Dry-run before launching

Use the dry-run config to verify the full pipeline end-to-end before launching the full experiment.
It runs 2 datasets, 1 prompt, 1 model, 1 judge, 10 rows per file — completes in minutes.

```bash
bash scripts/data/run_build_translation_candidates.sh       config/experiments/translation_eval_dryrun.sh
bash scripts/translation/run_translation_pipeline.sh        config/experiments/translation_eval_dryrun.sh
bash scripts/translation/run_eval_translation_api_pipeline.sh config/experiments/translation_eval_dryrun.sh
```

To customize a dry-run, edit `config/experiments/translation_eval_dryrun.sh` — reduce `DATASET_SLUGS`,
`TRANSLATION_MODELS`, or `JUDGE_MODELS` arrays, or set `LIMIT=10` to cap rows per file.

---

## Translating a single file manually

```bash
conda activate htr && source .env && export PYTHONPATH="./src:$PYTHONPATH"

# Queries (pass context_key for document context)
python src/translation/api/run_translation_pipeline.py \
    --source_file_paths path/to/queries.csv \
    --prompt_file_name prompts/translation/openai/translation_prompts_fewshot_v20250220.yaml \
    --output_dir path/to/output/ \
    --model_name gpt-4o-mini-2024-07-18 \
    --english_key text \
    --hebrew_key translation \
    --context_key context_text \
    --workers 4

# Documents (segment_text is the column to translate; no context_key)
python src/translation/api/run_translation_pipeline.py \
    --source_file_paths path/to/documents.csv \
    --prompt_file_name prompts/translation/openai/translation_prompts_fewshot_v20250220.yaml \
    --output_dir path/to/output/ \
    --model_name gpt-4o-mini-2024-07-18 \
    --english_key segment_text \
    --hebrew_key translation \
    --workers 4
```

## Evaluating a single file manually

```bash
# Queries from a QA dataset (uses qa prompt)
python src/translation/api/run_evaluate_translations.py \
    --source_file_paths path/to/queries_translated.csv \
    --prompt_file_name prompts/translation/openai/translation_evaluation_nogold_qa_v20250406.yaml \
    --output_dir path/to/output/ \
    --model_name claude-sonnet-4-6 \
    --english_key text \
    --hebrew_key translation \
    --workers 4

# Documents from a bio-medical dataset (uses technical prompt)
python src/translation/api/run_evaluate_translations.py \
    --source_file_paths path/to/documents_translated.csv \
    --prompt_file_name prompts/translation/openai/translation_evaluation_nogold_technical_v20250406.yaml \
    --output_dir path/to/output/ \
    --model_name claude-sonnet-4-6 \
    --english_key segment_text \
    --hebrew_key translation \
    --workers 4
```

---

## Reference: which prompt for which dataset?

| Dataset | Category | Translation prompt | Eval prompt |
|---|---|---|---|
| BeIR/msmarco | Misc | fewshot_v20250220 | general |
| BeIR/fever | Fact checking | fewshot_v20250220 | general |
| BeIR/climate-fever | Fact checking | fewshot_v20250220 | general |
| BeIR/scifact | Fact checking | fewshot_v20250220 | **technical** |
| BeIR/scidocs | Citation prediction | fewshot_v20250220 | **technical** |
| BeIR/quora | Duplicate questions | fewshot_v20250220 | **qa** |
| BeIR/arguana | Argument retrieval | fewshot_v20250220 | **qa** |
| BeIR/nq | Question answering | fewshot_v20250220 | **qa** |
| BeIR/hotpotqa | Question answering | fewshot_v20250220 | **qa** |
| BeIR/trec-covid | Bio-medical IR | fewshot_v20250220 | **technical** |
| BeIR/nfcorpus | Bio-medical IR | fewshot_v20250220 | **technical** |
| BeIR/dbpedia-entity | Entity retrieval | fewshot_v20250220 | general |

> Note: all datasets share the same translation prompt — category-specific variation
> is only needed for *evaluation*, not translation.


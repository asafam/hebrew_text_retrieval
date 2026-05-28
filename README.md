# Hebrew Text Retrieval — BeIR Translation Experiments

This project translates the [BeIR](https://github.com/beir-cellar/beir) information retrieval benchmark from English to Hebrew and evaluates translation quality using LLM-as-a-judge. The translated datasets are intended for training and evaluating dense retrieval models for Hebrew.

---

## Experiments

### Experiment 1 — Translation quality: single-dataset baseline (2025-02)

**Goal:** Establish baseline translation quality on a single dataset (MSMarco) across multiple LLMs and prompt strategies.

**Datasets:** `BeIR/msmarco` — 1,024 queries (with document context), 1,024 documents

**Translation models:**
- `gpt-4o-mini-2024-07-18`
- `gemini-2.0-flash-lite`
- `google_gemma-2-27b-it` (via Together AI)

**Prompt strategies tested** (all v20250220, see `prompts/translation/README.md`):
- Zero-shot vs. few-shot (4 examples: health, Python, pasta, gaming)
- With document context vs. without context (no-context as baseline)
- Search-optimised (instructs model to use terms Hebrew speakers would search with)
- Unified (translate document + query in a single call)
- Examples-first ordering

**Key finding:** Few-shot with context outperforms zero-shot and no-context variants. Context is critical for resolving short ambiguous queries.

**Outputs:** `outputs/translation/BeIR/BeIR_msmarco/`

---

### Experiment 2 — Long document segmentation (2025-02)

**Goal:** Determine optimal document segment length for translation quality.

**Datasets:** Long documents (>512 tokens) aggregated from 12 BeIR datasets — dominated by `BeIR/climate-fever` (~97%) and `BeIR/scifact` (~3%). 8,537 documents total.

**Segment lengths tested:** 256 / 512 / 1024 / 2048 tokens (using `tiktoken` GPT-4 encoding)

**Translation models:** `gpt-4o-mini-2024-07-18`, `gemini-2.0-flash-lite`, `google_gemma-2-27b-it`

**Evaluation:** Each translation scored 0–5 by `gpt-4o-2024-08-06` using the additive rubric in `translation_evaluation_nogold_v20250304.yaml`. Scores aggregated per document (mean over segments), then averaged across documents.

**Key finding:** 256–512 tokens is the sweet spot. Shorter segments translate more accurately; 2048-token unsegmented documents lose quality.

**Outputs:** `outputs/translation/BeIR/long_documents/`

---

### Experiment 4 — Full BeIR corpus translation with shard-ladder QA gating (2025-05) 🔜

**Goal:** Translate all 15 BeIR datasets end-to-end (queries + documents) using `gemini-3.1-flash-lite`, with progressive QA gating to catch quality drift before wasting compute on large corpora.

**Datasets:** All 15 BeIR datasets (3K–8.8M documents each, ordered small → large):
`nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`, `trec-covid`, `webis-touche2020`, `cqadupstack`, `quora`, `nq`, `hotpotqa`, `dbpedia-entity`, `fever`, `climate-fever`, `msmarco`

**Translation model:** `gemini-3.1-flash-lite`

**QA judge:** `gemini-3.1-pro-preview` (samples 25 rows per shard, scores 1–5)

**Approach — fixed-shard ladder:**
Each dataset is split into fixed-size shards (configured per dataset: 500 rows for `nfcorpus` → 100K rows for `msmarco`). After translating each shard, the judge scores a random sample of the accumulated translations. If the mean score falls below 3.5, that dataset stops automatically; others continue unaffected.

**Pilot result (all 15/15 datasets passed):** scores ranged 3.72–5.00 / 5 on 100-row synchronous pilot runs.

**Outputs:** `outputs/translation/runs/<timestamp>_<run_id>/`

---

### Experiment 3 — Multi-dataset balanced evaluation with multi-judge LLM-as-a-judge (2025-04) 🔜

**Goal:** Evaluate translation quality across a balanced sample of all BeIR categories using multiple LLM judges, with category-appropriate evaluation prompts.

**Motivation:** Experiments 1 and 2 covered only 1–2 domains. BeIR spans 9 categories (QA, bio-medical, argument retrieval, fact-checking, etc.) with different translation challenges. A single evaluation prompt does not capture these differences equally.

**Datasets:** All 12 BeIR datasets — 100 queries + 100 documents per dataset (~1,200 items total, balanced across all 9 categories).

| Category | Datasets |
|---|---|
| Misc | msmarco |
| Fact checking | fever, climate-fever, scifact |
| Citation prediction | scidocs |
| Duplicate questions | quora |
| Argument retrieval | arguana |
| Question answering | nq, hotpotqa |
| Bio-medical IR | trec-covid, nfcorpus |
| Entity retrieval | dbpedia-entity |

**Translation models:** `gpt-4o-mini-2024-07-18`, `gemini-2.0-flash-lite`, `google_gemma-2-27b-it`

**Judge models:**
- `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`
- `gemini-3.1-pro`, `gemini-3.1-flash`
- `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`

**Evaluation prompts** — category-specific (all v20250406):

| Prompt | Datasets | Distinguishing criterion |
|---|---|---|
| `translation_evaluation_nogold_v20250406.yaml` | msmarco, fever, climate-fever, dbpedia-entity | IR keyword preservation |
| `translation_evaluation_nogold_technical_v20250406.yaml` | scifact, scidocs, trec-covid, nfcorpus | Technical term handling (translate vs. transliterate vs. keep in English per Israeli scientific convention) |
| `translation_evaluation_nogold_qa_v20250406.yaml` | nq, hotpotqa, arguana, quora | Natural Hebrew question phrasing; rhetorical connectives; de-emphasises keyword matching |

**Outputs:** `outputs/translation/BeIR/candidates/<dataset_slug>/<translation_model>/`

---

## Running the experiments

### Prerequisites

```bash
conda activate htr
cp .env.example .env   # fill in API keys
export PYTHONPATH="./src:$PYTHONPATH"
```

Required keys in `.env`: `OPENAI_API_KEY`, `OPENAI_API_ORG`, `OPENAI_PROJECT`, `GEMINI_API_KEY`, `GEMINI_PROJECT`, `TOGETHER_API_KEY`, `ANTHROPIC_API_KEY`

### Experiment 4 — Full corpus ladder pipeline

```bash
# 1. Build sharded candidate CSVs (one shard_manifest.json per dataset)
#    Shard sizes are configured in config/translation/full_corpus.yaml
#    under datasets.shard_sizes (e.g. 500 for nfcorpus, 100000 for msmarco).
python -m translation.build_translation_candidates \
    --dataset_names BeIR/nfcorpus BeIR/scifact ...   \  # or all 15
    --model_name_or_path gpt-4o-mini-2024-07-18       \
    --output_path outputs/translation/candidates       \
    --shard-size 500   # overridden per-dataset by the YAML at runtime

# 2. Dry-run: inspect the shard plan without translating
python -m translation.api.run_beir_ladder_pipeline \
    --config config/translation/full_corpus.yaml \
    --dry-run

# 3. Run the ladder (new timestamped run folder created automatically)
python -m translation.api.run_beir_ladder_pipeline \
    --config config/translation/full_corpus.yaml \
    [--dataset BeIR/nfcorpus]   # optional: single dataset

# 4. Resume after an interruption (human decision required)
#    The pipeline exits with instructions if an in-progress run is detected.
#    Once you decide to resume:
python -m translation.api.run_beir_ladder_pipeline \
    --config config/translation/full_corpus.yaml \
    --resume
```

> **Kill safety:** if the process is killed and restarted without `--resume`, it exits
> with a message listing your options (resume the same run, or start fresh by changing
> `run_id` in the config). Resuming an interrupted run re-translates only the current
> in-flight shard; all completed shards are skipped.

### Experiment 3 — step by step

```bash
# 1. Build balanced candidate sets (100 queries + 100 docs per dataset)
bash scripts/data/run_build_translation_candidates.sh

# 2. Translate queries with each translation model
bash scripts/translation/run_translation_candidates_api_pipeline.sh

# 3. Evaluate translations with all judge models
bash scripts/translation/run_eval_translation_api_pipeline.sh
```

Set `LIMIT=10` in any script for a quick dry-run before launching the full job.

### Re-running earlier experiments

```bash
# Translate (original long-document experiment)
bash scripts/translation/run_translation_api_pipeline.sh

# Evaluate (original long-document experiment)
# Edit SOURCE_FILE_PATHS and MODEL_NAME in run_eval_translation_api_pipeline.sh,
# then run:
bash scripts/translation/run_eval_translation_api_pipeline.sh
```

---

## Output structure

```
outputs/
├── translation/
│   ├── BeIR/                              # Experiments 1–3
│   │   ├── BeIR_msmarco/                  # Experiment 1
│   │   │   ├── queries.csv
│   │   │   ├── documents.csv
│   │   │   └── <model>/
│   │   │       ├── queries_translated.csv
│   │   │       └── queries_translated_evaluated.csv
│   │   ├── long_documents/                # Experiment 2
│   │   │   ├── long_docs.csv
│   │   │   └── <model>/
│   │   │       └── long_docs_segmented_*_translated*.csv
│   │   └── candidates/                    # Experiment 3
│   │       └── <dataset_slug>/
│   │           ├── queries.csv
│   │           └── <translation_model>/
│   │               └── queries_translated.csv
│   │
│   ├── candidates/                        # Experiment 4 — sharded candidates
│   │   └── <dataset_slug>/
│   │       ├── queries_shard_000.csv
│   │       ├── queries_shard_001.csv
│   │       ├── ...
│   │       ├── documents_shard_000.csv
│   │       ├── ...
│   │       └── shard_manifest.json        # shard index (read by ladder pipeline)
│   │
│   └── runs/                              # Experiment 4 — ladder run output
│       └── <YYYYMMDD_HHMMSS>_<run_id>/
│           ├── run.log
│           ├── progress.json              # per-shard state; safe for --resume
│           ├── qa_scores.csv              # one row per shard per dataset
│           ├── plots/
│           │   ├── <dataset_slug>.png     # score vs. cumulative rows (±1σ)
│           │   └── summary.png            # heatmap: all datasets × all shards
│           └── <dataset_slug>/
│               ├── shards/
│               │   ├── queries_shard_000_translated.csv
│               │   └── documents_shard_000_translated.csv
│               ├── queries_accumulated.csv
│               └── documents_accumulated.csv
│
└── beir_translation/
    └── qa_history.csv                     # global QA log across all runs
```

---

## Evaluation rubric

All experiments use a **0–5 additive rubric**. Each point is awarded independently:

| Score | Criterion |
|---|---|
| 0 | Missing or not Hebrew |
| +1 | Core semantics recognisable |
| +1 | Structural fidelity (clause order, logical flow) |
| +1 | Lexical and morphological accuracy |
| +1 | IR keyword preservation / technical term handling / natural question phrasing *(varies by prompt)* |
| +1 | Flawless fluency |

Responses are structured as `TranslationCritique(critique: str, score: int)` via LLM tool-use.

---

## Key source files

| File | Purpose |
|---|---|
| `src/translation/build_translation_candidates.py` | Samples queries + documents from BeIR; `--shard-size N` splits into fixed-size shards + manifest |
| `src/data/translation_candidates/beir/__init__.py` | Loads BeIR datasets (incl. fastparquet fallback for fiqa, webis-touche2020, cqadupstack), pairs queries with relevant documents, segments long docs |
| `src/translation/api/run_beir_ladder_pipeline.py` | Shard-ladder orchestrator: translate → accumulate → judge → gate per shard; kill-safe with `--resume` |
| `src/translation/api/plot_ladder_scores.py` | Renders per-dataset score curves (±1σ) and summary heatmap after each shard |
| `src/translation/api/run_beir_batch_gcs.py` | GCS batch pipeline: pilot / submit / collect phases via Vertex AI |
| `src/translation/api/translate.py` | Translation pipeline (serial + parallel workers) |
| `src/translation/api/evaluate_translations.py` | LLM-as-a-judge evaluation pipeline |
| `src/llms/router.py` | Routes model name to OpenAI / Gemini / Anthropic / Together AI |
| `config/translation/full_corpus.yaml` | Single config for all 15 BeIR datasets: models, prompts, shard sizes, QA thresholds |
| `prompts/translation/README.md` | All prompt variants with dates and descriptions |

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
outputs/translation/BeIR/
├── BeIR_msmarco/                         # Experiment 1
│   ├── queries.csv
│   ├── documents.csv
│   └── <model>/
│       ├── queries_with_ambiguity_translated.csv
│       └── queries_with_ambiguity_translated_evaluated.csv
├── long_documents/                        # Experiment 2
│   ├── long_docs.csv
│   ├── long_docs_segmented_{256,512,1024}.csv
│   └── <model>/
│       ├── long_docs_segmented_*_translated.csv
│       └── long_docs_segmented_*_translated_evaluated.csv
└── candidates/                            # Experiment 3
    └── <dataset_slug>/
        ├── queries.csv
        ├── documents.csv
        └── <translation_model>/
            ├── queries_translated.csv
            └── evaluations/
                └── <judge_model>/
                    └── queries_translated_evaluated.csv
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
| `src/translation/build_translation_candidates.py` | Samples queries + documents from BeIR via HuggingFace |
| `src/data/beir/__init__.py` | Loads BeIR datasets, pairs queries with relevant documents, segments long documents |
| `src/translation/api/translate.py` | Translation pipeline (serial + parallel) |
| `src/translation/api/evaluate_translations.py` | LLM-as-a-judge evaluation pipeline |
| `src/llms/router.py` | Routes model name to OpenAI / Gemini / Anthropic / Together AI |
| `prompts/translation/README.md` | All prompt variants with dates and descriptions |

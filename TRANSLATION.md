# BeIR Full Corpus Translation

Translates all queries, documents, and titles across 12 BeIR datasets into Hebrew using the Gemini Batch API (Vertex AI). The pipeline is split into three explicit phases — **pilot → submit → collect** — so all 36 batch jobs (12 datasets × queries + documents + titles) are submitted simultaneously instead of sequentially.

---

## Prerequisites

### 1. GCP Authentication

This pipeline uses Vertex AI (not the Gemini API key). Run once:

```bash
gcloud auth application-default login
```

Verify it works:

```bash
gcloud auth application-default print-access-token
```

`GEMINI_API_KEY` must be **unset** — the pipeline will raise an error if it is set.

### 2. GCS Bucket

The bucket `beir-translation` must exist in project `iucc-tsarfaty-lab-gcp-asaf`, region `us-east1`. Create it via the GCP Console or:

```bash
gsutil mb -p iucc-tsarfaty-lab-gcp-asaf -l us-east1 gs://beir-translation
```

### 3. Candidate CSVs

The source CSVs (English text to translate) must exist under `outputs/translation/BeIR/candidates/`. Each dataset directory contains `queries.csv` and `documents.csv`. These are generated separately by the candidate-building pipeline and are a prerequisite for running any translation.

### 4. Environment

```bash
conda activate htr
export PYTHONPATH="./src:$PYTHONPATH"
```

Install dependencies:

```bash
pip install -r requirements-translation.txt
```

---

## Config

All pipeline parameters live in `config/translation/full_corpus.yaml`. Nothing is hardcoded.

```yaml
run_id: "full_corpus_zeroshot_nocontext_gemini31flashlite"

# ── Per-type model and prompt configuration ────────────────────────────────────

queries:
  model: "gemini-3.1-flash-lite"
  temperature: 0.7
  prompt:
    file: "prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20250220.yaml"
    type: "query"
    text_col: "text"              # CSV column with source text
    english_key: "Text"           # template variable: source text
    hebrew_key: "Hebrew Query"    # template variable: translation output
    context_col: "context_text"   # CSV column for context (leave blank if none)
    context_key: "Context"        # template variable: context

documents:
  model: "gemini-3.1-flash-lite"
  temperature: 0.7
  prompt:
    file: "prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20250220.yaml"
    type: "document"
    text_col: "segment_text"
    english_key: "Text"
    hebrew_key: "Hebrew Document"

titles:
  model: "gemini-3.1-flash-lite"
  temperature: 0.7
  prompt:
    file: "prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20250220.yaml"
    type: "document"
    text_col: "segment_text"
    english_key: "Text"
    hebrew_key: "Hebrew Document"

# ── Datasets ───────────────────────────────────────────────────────────────────

datasets:
  names:
    - "BeIR/nfcorpus"        # 3,633 docs
    - "BeIR/scifact"          # 5,183 docs
    - "BeIR/arguana"          # 8,674 docs
    - "BeIR/scidocs"          # 25,657 docs
    - "BeIR/trec-covid"       # 171,332 docs
    - "BeIR/quora"            # 522,931 docs
    - "BeIR/nq"               # 2,681,468 docs
    - "BeIR/hotpotqa"         # 5,233,329 docs
    - "BeIR/dbpedia-entity"   # 4,635,922 docs
    - "BeIR/fever"            # 5,416,568 docs
    - "BeIR/climate-fever"    # 5,416,593 docs
    - "BeIR/msmarco"          # 8,841,823 docs

# ── GCS / Vertex AI ───────────────────────────────────────────────────────────

gcs:
  bucket: "beir-translation"
  project: "iucc-tsarfaty-lab-gcp-asaf"
  location: "us-east1"
  input_prefix: "beir/input"
  output_prefix: "beir/output"

# ── Guardrails ────────────────────────────────────────────────────────────────

guardrails:
  max_cost_usd: 2500.0
  cost_per_1m_input_tokens: 0.50     # gemini-3.1-flash-lite
  cost_per_1m_output_tokens: 1.50    # gemini-3.1-flash-lite

# ── Pilot ─────────────────────────────────────────────────────────────────────

progression:
  pilot_n: 100        # rows translated synchronously before batch submit
  pilot_qa: true      # run LLM-as-a-judge on pilot output; block submit on failure
```

To use a different model for queries vs documents, change the `model` field in the relevant section.

---

## Running the Pipeline

### Step 1 — Pilot (smoke test)

Translates 100 rows per dataset synchronously (no batch jobs), runs LLM-as-a-judge QA on the output, and prints a score for every dataset. Takes ~20–30 minutes for all 12.

```bash
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml pilot
```

After it finishes:
- Review the QA scores printed in the terminal (PASS / FAIL per dataset)
- Manually spot-check a few rows in the translated CSVs:
  ```
  outputs/beir_translation/full_corpus/<run_id>/<dataset>/queries_translated.csv
  outputs/beir_translation/full_corpus/<run_id>/<dataset>/documents_translated.csv
  ```

Datasets that fail QA are automatically excluded from submit. Fix the prompt or model and re-run pilot before proceeding.

### Step 2 — Submit

Uploads full input JSONLs to GCS and fires all batch jobs at once. Only datasets that passed pilot QA are submitted. Omit `--yes` the first time to review the cost estimate before confirming.

```bash
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml submit --yes
```

Job names and GCS paths are saved to `progress.json` after each submission — crash-safe and re-entrant.

### Step 3 — Collect

Checks job status, downloads completed results, runs post-translation QA, and exports to BeIR JSONL. Prints a status table showing every dataset and job.

```bash
# Run once to check current status
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml collect

# Or loop until everything is done (run overnight)
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml collect --wait --poll-interval 3600
```

Example status table:

```
┌──────────────────────┬────────────────────────┬────────────────────────┬───────────┬──────┬────────┐
│ Dataset              │ Queries                │ Documents              │ Titles    │ QA   │ Export │
├──────────────────────┼────────────────────────┼────────────────────────┼───────────┼──────┼────────┤
│ BeIR/nfcorpus        │ SUCCEEDED (1h 42m)     │ SUCCEEDED (2h 03m)     │ SUCCEEDED │ PASS │ done   │
│ BeIR/trec-covid      │ SUCCEEDED (8h 11m)     │ RUNNING (6h 30m)       │ SUCCEEDED │ -    │ -      │
│ BeIR/msmarco         │ RUNNING (3h 15m)       │ PENDING                │ -         │ -    │ -      │
└──────────────────────┴────────────────────────┴────────────────────────┴───────────┴──────┴────────┘
```

---

## Quality Gates

| Gate | When | What it catches |
|---|---|---|
| Pilot QA (LLM-as-a-judge) | After pilot | Bad translations, wrong prompt or model — blocks submit |
| Cost estimate prompt | Before submit | Unexpected scale or pricing — requires manual confirmation |
| Row-count check | At collect | Ordering/alignment bugs between input and output |
| Post-collect QA | After collect | Degradation vs. baseline on full dataset translations |

---

## Retry a Single Dataset

```bash
# Re-run pilot for one dataset (e.g. after fixing its prompt)
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml pilot --dataset BeIR_fever

# Submit only that dataset
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml submit --dataset BeIR_fever --yes

# Collect only that dataset
python -m translation.api.run_beir_batch_gcs \
    --config config/translation/full_corpus.yaml collect --dataset BeIR_fever
```

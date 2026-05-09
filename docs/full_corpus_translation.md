# BeIR Full Corpus Translation

Translates all 12 BeIR datasets (queries + documents) from English to Hebrew using an LLM,
producing HuggingFace-ready JSONL files with matching IDs and original text preserved alongside translations.

---

## Quick start

```bash
# 1. Install dependencies (once)
pip install -r requirements-translation.txt

# 2. Set up API keys in .env (copy from .env.example)
cp .env.example .env   # then fill in your keys

# 3. Translate queries first (cheaper, fast validation)
bash scripts/translation/run_full_corpus.sh query

# 4. After QA passes, translate documents
bash scripts/translation/run_full_corpus.sh document
```

The pipeline is **resumable**: re-running the same command picks up where it left off.

---

## Configuration

All settings live in **`config/translation/full_corpus.yaml`**.

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `name` | `gpt-5.4-mini` | Translation model |
| `model` | `temperature` | `0.7` | Sampling temperature |
| `prompt` | `file` | `...zeroshot_nocontext...` | Prompt YAML |
| `datasets` | `names` | all 12 BeIR | Datasets to process |
| `datasets` | `num_samples` | `0` | `0` = all records; `N` = random N per dataset |
| `execution` | `mode` | `auto` | `auto` → batch API for `gpt-*`, parallel otherwise |
| `guardrails` | `max_cost_usd` | `2500` | Hard stop if estimated cost exceeds this |
| `guardrails` | `cost_per_1m_input_tokens` | `0.075` | Batch API pricing |
| `progression` | `pilot_n` | `100` | Rows to translate first before scaling |
| `progression` | `pilot_qa` | `true` | Run LLM-as-a-judge on pilot before continuing |
| `qa` | `enabled` | `true` | Run QA after each dataset |
| `qa` | `baseline_csv` | `outputs/translation/BeIR/results_translation_eval.csv` | Reference scores |
| `qa` | `judge_model` | `claude-sonnet-4-6` | Judge for QA evaluations |
| `qa` | `sample_size` | `25` | Rows sampled per dataset for QA |

---

## Run script options

```bash
# Translate only queries
bash scripts/translation/run_full_corpus.sh query

# Translate only documents
bash scripts/translation/run_full_corpus.sh document

# Translate both (default)
bash scripts/translation/run_full_corpus.sh both

# Skip cost confirmation (CI / non-interactive)
bash scripts/translation/run_full_corpus.sh query --yes

# Use a different config
bash scripts/translation/run_full_corpus.sh query --config config/translation/my_config.yaml
```

Or call Python directly:

```bash
export PYTHONPATH="src:$PYTHONPATH"
python -m translation.api.run_beir_translation_pipeline \
    --config config/translation/full_corpus.yaml \
    --text-type query \
    --yes
```

---

## Progressive translation

The pipeline translates in two passes per dataset:

1. **Pilot** (`progression.pilot_n = 100`): Translates the first 100 rows.
2. **Pilot QA**: Runs LLM-as-a-judge on the pilot. If scores fall below baseline, the pipeline **stops** and you must investigate before continuing.
3. **Full translation**: Translates remaining rows (pilot rows are preserved, not re-translated).
4. **Post-dataset QA**: Another QA check after full translation.

Set `progression.pilot_n: 0` to skip the pilot and go straight to full translation.

---

## Guardrails

Before submitting each batch job the pipeline:

1. **Estimates cost** — samples 300 rows to estimate average token counts and projects total USD cost.
2. **Checks budget cap** — aborts if estimated cost exceeds `guardrails.max_cost_usd`.
3. **Prompts for confirmation** — shows the estimate and asks `Proceed? [y/N]`. Pass `--yes` to skip.
4. **QA after each dataset** — compares LLM judge scores against the 25-sample baseline. Stops the pipeline with exit code `2` on degradation.

### QA degradation thresholds

A dataset is flagged when **either** condition holds:
- z-score: `(baseline_mean − sample_mean) / baseline_std > 1.5`
- Absolute drop: `baseline_mean − sample_mean > 0.5` points

On failure the pipeline:
- Prints the degraded datasets and 5 spot-check translation pairs
- Writes a JSON report to `outputs/beir_translation/full_corpus/qa_*.json`
- Exits with code `2`

To resume after fixing the issue:
```bash
bash scripts/translation/run_full_corpus.sh query   # re-run; picks up at the failed dataset
```

---

## Output structure

```
outputs/beir_translation/full_corpus/
  full_corpus_zeroshot_nocontext_gpt54mini/
    progress.json                    ← checkpoint; do not delete
    BeIR_nfcorpus/
      queries_translated.csv         ← translated queries with original text + _id
      documents_translated.csv       ← translated docs (segmented) with original + _id
      beir/
        corpus.jsonl                 ← HuggingFace corpus: {_id, title (HE), title_en, text (HE), text_en}
        queries.jsonl                ← HuggingFace queries: {_id, text (HE), text_en}
        qrels/test.tsv               ← relevance judgments (unchanged from original BeIR)
        metadata.json                ← run metadata
    BeIR_scifact/
      ...
  cache/
    gpt_5_4_mini__translation_prompts_zeroshot_nocontext_v20250220.jsonl
      ← shared translation cache; speeds up duplicate segments across datasets

jobs/full_corpus/
  batch_jobs.json                    ← OpenAI batch job tracking (one entry per chunk)
```

---

## Dataset sizes and cost estimate

| Dataset | Queries | Documents | Corpus size |
|---|---:|---:|---|
| NFCorpus | 323 | 3,633 | tiny |
| SciFact | 300 | 5,183 | tiny |
| ArguAna | 1,406 | 8,674 | tiny |
| SCIDOCS | 1,000 | 25,657 | small |
| TREC-COVID | 50 | 171,332 | small |
| Quora | 15,000 | 522,931 | medium |
| NQ | 3,452 | 2,681,468 | large |
| HotpotQA | 7,405 | 5,233,329 | large |
| DBPedia-Entity | 400 | 4,635,922 | large |
| FEVER | 140,085 | 5,416,568 | large (Wikipedia) |
| Climate-FEVER | 1,535 | 5,416,593 | large (same Wikipedia corpus) |
| MSMARCO | 6,980 | 8,841,823 | very large |
| **Total** | **~178K** | **~32.9M** | |

Estimated total cost at `gpt-5.4-mini` batch API rates: **~$1,200–$1,600** depending on average document length.

---

## Uploading to HuggingFace

After translation, each dataset's `beir/` directory contains HuggingFace-compatible JSONL files.

```bash
# Upload a single dataset
huggingface-cli upload <your-org>/beir-hebrew \
    outputs/beir_translation/full_corpus/full_corpus_zeroshot_nocontext_gpt54mini/BeIR_nfcorpus/beir/ \
    BeIR_nfcorpus/ \
    --repo-type dataset

# Or use the Python SDK
from huggingface_hub import upload_folder
upload_folder(
    repo_id="<your-org>/beir-hebrew",
    folder_path="outputs/beir_translation/.../BeIR_nfcorpus/beir",
    path_in_repo="BeIR_nfcorpus",
    repo_type="dataset",
)
```

---

## Resuming after interruption

The pipeline writes `progress.json` after every completed phase. Re-running the same command
resumes automatically. No data is lost or re-translated.

```bash
# Simply re-run — it continues from the last checkpoint
bash scripts/translation/run_full_corpus.sh query
```

To force a re-run of a specific phase, set `force_translation: true` or `force_candidates: true`
in the config, or delete the relevant `*_translated.csv` file.

---

## Changing model or prompt

Create a new config file based on `full_corpus.yaml` and change the `run_id` to get a separate output directory:

```yaml
# config/translation/full_corpus_claude.yaml
run_id: "full_corpus_zeroshot_nocontext_claude_haiku"
model:
  name: "claude-haiku-4-5-20251001"
...
```

```bash
bash scripts/translation/run_full_corpus.sh query --config config/translation/full_corpus_claude.yaml
```

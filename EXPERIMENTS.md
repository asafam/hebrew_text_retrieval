# EXPERIMENTS.md

Step-by-step guide for running the BeIR Hebrew translation experiments.

---

## 1. One-time setup

### Environment

```bash
conda activate htr
```

All pipeline scripts activate `htr` automatically, but your shell must be able to find `conda` (i.e., conda must be initialized in your shell profile).

### `.env` file

The project root must contain a `.env` with:

```
OPENAI_API_KEY=...
OPENAI_API_ORG=...
OPENAI_PROJECT=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
TOGETHER_API_KEY=...
```

All pipeline scripts source `.env` automatically.

### Python path

Only needed when running Python scripts directly (not via the pipeline shell scripts):

```bash
export PYTHONPATH="./src:$PYTHONPATH"
```

---

## 2. How configs work

Every pipeline script takes an optional config file as its first argument:

```
bash <script.sh> [config/experiments/<config>.sh]
```

The config file is a plain bash file that defines all variables (datasets, models, limits, etc.). The script contains only logic — nothing to edit between runs.

**Available configs:**

| Config file | What it runs |
|---|---|
| `config/experiments/translation_eval_dryrun.sh` | 2 datasets · 1 translation model · 1 judge · 10 rows — fast smoke test |
| `config/experiments/translation_eval.sh` | Full experiment: 12 datasets × 4 prompts × 3 translation models × 7 judges |
| `config/experiments/translation_eval_calibration.sh` | Calibration: same translations scored by all 3 eval prompt variants |

---

## 3. Run the translation evaluation

The three steps are **strictly sequential** — each depends on the output of the previous:

```
Step 1 (build candidates)  →  Step 2 (translate)  →  Step 3 (evaluate + analyze)
```

**Always dry-run first** to confirm the full pipeline works end-to-end before launching the real run.

### Step 0 — Dry run (smoke test)

Run all three steps in order with the dryrun config:

```bash
# 1. Build candidate files (required before translation)
bash scripts/data/run_build_translation_candidates.sh \
    config/experiments/translation_eval_dryrun.sh

# 2. Translate (requires candidates from step above)
bash scripts/translation/run_translation_pipeline.sh \
    config/experiments/translation_eval_dryrun.sh

# 3. Evaluate + analyze (requires translations from step above)
bash scripts/translation/run_eval_translation_api_pipeline.sh \
    config/experiments/translation_eval_dryrun.sh
```

Writes to `outputs/translation/BeIR/candidates_dryrun/` — separate from production outputs.

---

### Step 1 — Build candidates

```bash
bash scripts/data/run_build_translation_candidates.sh \
    config/experiments/translation_eval.sh --workers 4
```

Samples 100 queries + 100 documents per dataset. Documents are segmented at up to 2048 tokens (effectively one segment per document for all BeIR datasets). Each document row carries a `segment_id=0`.

**Continuation:** datasets with existing output files are skipped. Re-run after a crash and it resumes from where it stopped. Set `FORCE=true` in the config to force a full rebuild.

**Parallelism:** set `WORKERS=4` in the config file — the script passes it as `--workers 4` to Python, which processes N datasets in parallel. A failed dataset prints an error and continues rather than aborting the run. When running Python directly, pass `--workers 4` explicitly.

**Output:**
```
outputs/translation/BeIR/candidates/
└── BeIR_<dataset>/
    ├── queries.csv      # _id, text, context_text, category, dataset_name
    └── documents.csv    # _id, segment_text, segment_id, category, dataset_name
```

---

### Step 2 — Translate

```bash
bash scripts/translation/run_translation_pipeline.sh \
    config/experiments/translation_eval.sh --workers 4
```

Translates every combination of (dataset × translation model × prompt strategy).
Queries are translated with document context; documents use `segment_text` as input.

**Continuation:** combinations with an existing output file are skipped. Re-run after a crash to resume. Set `FORCE=true` in the config to re-translate everything.

**Parallelism:** set `WORKERS=N` in the config — the script passes it as `--workers N` to Python, running N (dataset × model × prompt) combinations simultaneously. When running Python directly, pass `--workers N` explicitly.

**Output:**
```
outputs/translation/BeIR/candidates/
└── BeIR_<dataset>/
    └── <translation_model>/
        └── <prompt_slug>/
            ├── queries_translated.csv
            └── documents_translated.csv
```

---

### Step 3 — Evaluate + Analyze

```bash
bash scripts/translation/run_eval_translation_api_pipeline.sh \
    config/experiments/translation_eval.sh
```

Scores every translation 0–5 with each judge model. The eval prompt (general / technical / QA) is chosen automatically per dataset (defined in the config's `DATASET_EVAL_PROMPT` map).

When all scoring is done, automatically runs:
- `src/translation/collect_results.py` → master CSV
- `src/translation/analyze_results.py` → charts + report

**Output:**
```
outputs/translation/BeIR/candidates/
└── BeIR_<dataset>/
    └── <translation_model>/
        └── <prompt_slug>/
            └── evaluations/
                └── <judge_model>/
                    ├── queries_translated_evaluated.csv
                    └── documents_translated_evaluated.csv

outputs/translation/BeIR/
├── results_translation_eval.csv        ← all scores, all factors as columns
└── analysis/
    ├── analysis_report.md           ← ranked tables, 10 questions answered
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

## 4. Run eval prompt calibration

Determines whether category-specialized eval prompts (general / technical / QA) score meaningfully differently from each other on the same translations.

```bash
bash scripts/translation/run_eval_prompt_calibration_pipeline.sh \
    config/experiments/translation_eval_calibration.sh
```

Uses translations already produced in Step 2 (fixed to `gpt-4o-mini-2024-07-18` + `fewshot` strategy). Runs all three eval prompts against all judge models.

**Output:**
```
outputs/translation/BeIR/candidates/
└── BeIR_<dataset>/
    └── <translation_model>/
        └── <prompt_slug>/
            └── eval_calibration/
                └── <eval_prompt_slug>/
                    └── <judge_model>/
                        ├── queries_translated_evaluated.csv
                        └── documents_translated_evaluated.csv

outputs/translation/BeIR/
├── calibration_results_translation_eval_calibration.csv
└── analysis_eval_prompts/
    ├── analysis_report.md
    └── *.png
```

---

## 5. Re-running / resuming

- **Interrupted run**: just re-run the same command. `FORCE=false` (default) skips rows that already have output.
- **Force re-score everything**: set `FORCE=true` in the config file.
- **Single file**: see `TRANSLATIONS.md` for manual per-file commands.

---

## 6. Creating a new experiment config

```bash
cp config/experiments/translation_eval.sh config/experiments/translation_eval_myvariant.sh
```

Edit `translation_eval_myvariant.sh`:
- Set a unique `EXPERIMENT_NAME` (controls output CSV filename and analysis subfolder)
- Change `DATASET_SLUGS`, `TRANSLATION_MODELS`, `JUDGE_MODELS`, or `TRANSLATION_PROMPTS` as needed
- Set `LIMIT=10` for a quick test before committing to the full run

Then run all three steps with your new config.

---

## 7. Research questions (answered automatically by `analyze_results.py`)

| # | Question |
|---|---|
| 1 | Which translation model is best overall? |
| 2 | Which prompt strategy is best overall? |
| 3 | Does quality vary by dataset category? |
| 4 | Query vs document — which is harder to translate? |
| 5 | Does text length hurt quality? (short = 1 segment, long = multiple) |
| 6 | How consistent are judge models? (calibration and discrimination) |
| 7 | Best (model × prompt) per category — heatmap |
| 8 | Best (model × prompt) combinations overall |
| 9 | Does context help more for queries than documents? |
| 10 | Which model degrades least on long documents? |

---

## 8. Factors in the master results CSV

`results_<experiment_name>.csv` has one row per scored translation with these columns:

| Column | Values |
|---|---|
| `dataset_slug` | e.g. `BeIR_msmarco` |
| `category` | Misc / Fact checking / QA / Bio-medical / etc. |
| `translation_model` | e.g. `gpt-4o-mini-2024-07-18` |
| `prompt_slug` | `fewshot` / `zeroshot` / `fewshot_nocontext` / `fewshot_searchopt` |
| `judge_model` | e.g. `claude-sonnet-4-6` |
| `text_type` | `query` or `document` |
| `text_length_bucket` | `short` (1 segment) or `long` (multiple segments) |
| `score` | 0–5 |
| `critique` | judge's free-text rationale |

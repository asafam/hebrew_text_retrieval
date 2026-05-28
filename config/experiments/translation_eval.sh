# Experiment 3 — Baseline configuration
# Multi-dataset balanced evaluation across all BeIR categories.
# Full factorial: 5 query prompts × 5 translation models × 2 judge models × 12 datasets.
#
# Usage:
#   bash scripts/data/run_build_translation_candidates.sh       config/experiments/translation_eval.sh
#   bash scripts/translation/run_translation_pipeline.sh  config/experiments/translation_eval.sh
#   bash scripts/translation/run_eval_translation_api_pipeline.sh        config/experiments/translation_eval.sh

# ── Experiment identity ────────────────────────────────────────────────────────
EXPERIMENT_NAME="translation_eval"
CANDIDATES_BASE="outputs/translation/BeIR/candidates"

# ── Build settings ─────────────────────────────────────────────────────────────
NUM_SAMPLES=25                # queries + documents per dataset
MAX_SEGMENT_TOKENS=2048       # effectively no segmentation for all BeIR datasets
TOKENIZER_MODEL="gpt-4o-mini-2024-07-18"
RANDOM_SEED=42

DATASET_SLUGS=(
    "BeIR_msmarco"
    "BeIR_fever"
    "BeIR_climate-fever"
    "BeIR_scifact"
    "BeIR_scidocs"
    "BeIR_quora"
    "BeIR_arguana"
    "BeIR_nq"
    "BeIR_hotpotqa"
    "BeIR_trec-covid"
    "BeIR_nfcorpus"
    "BeIR_dbpedia-entity"
)

# ── Translation prompts ────────────────────────────────────────────────────────
# QUERY_PROMPT_SLUGS and DOCUMENT_PROMPT_SLUGS list the strategies for each
# text type. Each slug must have a matching TRANSLATION_PROMPT_<slug> variable.
QUERY_PROMPT_SLUGS=(
    "fewshot"
    "fewshot_nocontext"
    "fewshot_searchopt"
    "zeroshot"
    "zeroshot_nocontext"
)
DOCUMENT_PROMPT_SLUGS=(
    "fewshot"
    "zeroshot"
)
QUERY_PROMPT_fewshot="prompts/translation/api/translation/query/fewshot.yaml"
QUERY_PROMPT_fewshot_nocontext="prompts/translation/api/translation/query/fewshot_nocontext.yaml"
QUERY_PROMPT_fewshot_searchopt="prompts/translation/api/translation/query/fewshot_searchopt.yaml"
QUERY_PROMPT_zeroshot="prompts/translation/api/translation/query/zeroshot.yaml"
QUERY_PROMPT_zeroshot_nocontext="prompts/translation/api/translation/query/zeroshot_nocontext.yaml"

DOCUMENT_PROMPT_fewshot="prompts/translation/api/translation/document/fewshot.yaml"
DOCUMENT_PROMPT_zeroshot="prompts/translation/api/translation/document/zeroshot.yaml"

# ── Translation models ─────────────────────────────────────────────────────────
TRANSLATION_MODELS=(
    "gpt-5.4-mini"
    "gpt-5.4-nano"
    "claude-haiku-4-5-20251001"
    "gemini-3.1-flash-lite"
    "moonshotai/kimi-k2.6"
)

# ── Evaluation prompts (LLM-as-a-judge) ───────────────────────────────────────
# Category-specialized prompts — selected automatically per dataset.
PROMPT_GENERAL="prompts/translation/api/evaluation/translation_evaluation_nogold_v20250406.yaml"
PROMPT_TECHNICAL="prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20250406.yaml"
PROMPT_QA="prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20250406.yaml"

# Dataset slug → eval prompt. Variable names use underscores (hyphens not allowed).
# The eval script sanitizes DATASET_SLUG (replaces - with _) before looking these up.
DATASET_EVAL_PROMPT_BeIR_msmarco="$PROMPT_GENERAL"
DATASET_EVAL_PROMPT_BeIR_fever="$PROMPT_GENERAL"
DATASET_EVAL_PROMPT_BeIR_climate_fever="$PROMPT_GENERAL"
DATASET_EVAL_PROMPT_BeIR_scifact="$PROMPT_TECHNICAL"
DATASET_EVAL_PROMPT_BeIR_scidocs="$PROMPT_TECHNICAL"
DATASET_EVAL_PROMPT_BeIR_quora="$PROMPT_QA"
DATASET_EVAL_PROMPT_BeIR_arguana="$PROMPT_QA"
DATASET_EVAL_PROMPT_BeIR_nq="$PROMPT_QA"
DATASET_EVAL_PROMPT_BeIR_hotpotqa="$PROMPT_QA"
DATASET_EVAL_PROMPT_BeIR_trec_covid="$PROMPT_TECHNICAL"
DATASET_EVAL_PROMPT_BeIR_nfcorpus="$PROMPT_TECHNICAL"
DATASET_EVAL_PROMPT_BeIR_dbpedia_entity="$PROMPT_GENERAL"

# ── Judge models ───────────────────────────────────────────────────────────────
JUDGE_MODELS=(
    "claude-sonnet-4-6"
    "google/gemini-3.1-pro-preview"
)

# ── Run settings ───────────────────────────────────────────────────────────────
LIMIT=25        # translate only 25 rows per file
SAMPLE=0        # fraction of rows to evaluate (0.0 = all, 0.25 = random 25%)
FORCE=false     # set to true to re-run already-completed rows
WORKERS=4       # number of parallel workers (0 or unset = serial)

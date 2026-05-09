# Experiment 3 — Dry-run configuration
# Quick smoke test: 2 datasets, 1 translation prompt, 1 translation model,
# 1 judge model, 10 rows per file.
# Use this to verify the pipeline end-to-end before launching the full experiment.
#
# Usage:
#   bash scripts/data/run_build_translation_candidates.sh       config/experiments/translation_eval_dryrun.sh
#   bash scripts/translation/run_translation_pipeline.sh  config/experiments/translation_eval_dryrun.sh
#   bash scripts/translation/run_eval_translation_api_pipeline.sh        config/experiments/translation_eval_dryrun.sh

EXPERIMENT_NAME="translation_eval_dryrun"
CANDIDATES_BASE="outputs/translation/BeIR/candidates_dryrun"

# ── Build settings ─────────────────────────────────────────────────────────────
NUM_SAMPLES=20
MAX_SEGMENT_TOKENS=256
TOKENIZER_MODEL="gpt-4o-mini-2024-07-18"
RANDOM_SEED=42

DATASET_SLUGS=(
    "BeIR_msmarco"
    "BeIR_trec-covid"
)

# ── Translation prompts ────────────────────────────────────────────────────────
PROMPT_SLUGS=("fewshot")
TRANSLATION_PROMPT_fewshot="prompts/translation/api/translation/translation_prompts_fewshot_v20250220.yaml"

# ── Translation models ─────────────────────────────────────────────────────────
TRANSLATION_MODELS=(
    "gpt-4o-mini-2024-07-18"
)

# ── Evaluation prompts ─────────────────────────────────────────────────────────
PROMPT_GENERAL="prompts/translation/api/evaluation/translation_evaluation_nogold_v20250406.yaml"
PROMPT_TECHNICAL="prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20250406.yaml"
PROMPT_QA="prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20250406.yaml"

DATASET_EVAL_PROMPT_BeIR_msmarco="$PROMPT_GENERAL"
DATASET_EVAL_PROMPT_BeIR_trec_covid="$PROMPT_TECHNICAL"

# ── Judge models ───────────────────────────────────────────────────────────────
JUDGE_MODELS=(
    "claude-haiku-4-5-20251001"
)

# ── Run settings ───────────────────────────────────────────────────────────────
LIMIT=10
FORCE=false
# WORKERS unset → serial (easier to debug)

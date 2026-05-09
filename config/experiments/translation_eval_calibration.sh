# Eval prompt calibration — configuration
# Evaluates the SAME translations with all three eval prompts to determine
# whether category-specialized prompts score differently from the general one.
#
# Fixed to one translation model and one prompt strategy (fewshot baseline).
# Uses a representative subset of datasets — one or two per eval-prompt category.
#
# Usage:
#   bash scripts/translation/run_eval_prompt_calibration_pipeline.sh config/experiments/translation_eval_calibration.sh

EXPERIMENT_NAME="translation_eval_calibration"
CANDIDATES_BASE="outputs/translation/BeIR/candidates"

# ── Fixed translation settings ─────────────────────────────────────────────────
CALIBRATION_TRANSLATION_MODEL="gpt-4o-mini-2024-07-18"
CALIBRATION_TRANSLATION_PROMPT_SLUG="fewshot"

# ── Representative datasets (one or two per eval-prompt category) ──────────────
# general  → msmarco, climate-fever
# technical → trec-covid, scidocs
# QA        → nq, arguana
CALIBRATION_DATASET_SLUGS=(
    "BeIR_msmarco"
    "BeIR_climate-fever"
    "BeIR_trec-covid"
    "BeIR_scidocs"
    "BeIR_nq"
    "BeIR_arguana"
)

# ── All three eval prompts ─────────────────────────────────────────────────────
EVAL_PROMPT_SLUGS=("general" "technical" "qa")
EVAL_PROMPT_general="prompts/translation/api/evaluation/translation_evaluation_nogold_v20250406.yaml"
EVAL_PROMPT_technical="prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20250406.yaml"
EVAL_PROMPT_qa="prompts/translation/api/evaluation/translation_evaluation_nogold_qa_v20250406.yaml"

# ── Judge models ───────────────────────────────────────────────────────────────
JUDGE_MODELS=(
    "gpt-5.4"
    "gpt-5.4-mini"
    "gpt-5.4-nano"
    "gemini-3.1-pro"
    "gemini-3.1-flash"
    "claude-sonnet-4-6"
    "claude-haiku-4-5-20251001"
)

# ── Run settings ───────────────────────────────────────────────────────────────
LIMIT=0
FORCE=false
WORKERS=4       # number of parallel workers (0 or unset = serial)

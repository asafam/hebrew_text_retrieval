#!/bin/bash -i

# Eval prompt calibration experiment.
# Evaluates the same translations with all three eval prompts to determine
# whether category-specialized prompts produce meaningfully different scores.
#
# Usage:
#   bash scripts/translation/run_eval_prompt_calibration_pipeline.sh [config_file]
#
# Default config: config/experiments/translation_eval_calibration.sh

CONFIG="${1:-config/experiments/translation_eval_calibration.sh}"
if [ ! -f "$CONFIG" ]; then echo "Config not found: $CONFIG"; exit 1; fi
echo "Using config: $CONFIG"
source "$CONFIG"

echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr

if [ -f .env ]; then set -a; source .env; set +a; else echo ".env not found!"; exit 1; fi

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=3

[ "${FORCE}" = "true" ] && FORCE_FLAG="--force" || FORCE_FLAG=""

echo ""
echo "Experiment     : $EXPERIMENT_NAME"
echo "Translation    : $CALIBRATION_TRANSLATION_MODEL / $CALIBRATION_TRANSLATION_PROMPT_SLUG"
echo "Datasets       : ${CALIBRATION_DATASET_SLUGS[*]}"
echo "Eval prompts   : ${EVAL_PROMPT_SLUGS[*]}"
echo "Judge models   : ${JUDGE_MODELS[*]}"
echo ""

for DATASET_SLUG in "${CALIBRATION_DATASET_SLUGS[@]}"; do
    TRANS_DIR="${CANDIDATES_BASE}/${DATASET_SLUG}/${CALIBRATION_TRANSLATION_MODEL}/${CALIBRATION_TRANSLATION_PROMPT_SLUG}"

    for EVAL_SLUG in "${EVAL_PROMPT_SLUGS[@]}"; do
        var="EVAL_PROMPT_${EVAL_SLUG}"
        EVAL_PROMPT_FILE="${!var}"

        for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
            EVAL_OUTPUT_DIR="${TRANS_DIR}/eval_calibration/${EVAL_SLUG}/${JUDGE_MODEL}"

            QUERY_SOURCE="${TRANS_DIR}/queries_translated.csv"
            if [ -f "$QUERY_SOURCE" ]; then
                echo "Calibration | ${DATASET_SLUG} | eval:${EVAL_SLUG} | judge:${JUDGE_MODEL} | queries"
                python src/translation/api/run_evaluate_translations.py \
                    --source_file_paths "${QUERY_SOURCE}" \
                    --prompt_file_name "$EVAL_PROMPT_FILE" \
                    --output_dir "$EVAL_OUTPUT_DIR" \
                    --model_name "$JUDGE_MODEL" \
                    --limit $LIMIT \
                    --english_key "text" \
                    --hebrew_key "translation" \
                    $FORCE_FLAG \
                    ${WORKERS:+--workers $WORKERS}
            else
                echo "Skipping ${DATASET_SLUG} queries: not found at ${QUERY_SOURCE}"
            fi

            DOC_SOURCE="${TRANS_DIR}/documents_translated.csv"
            if [ -f "$DOC_SOURCE" ]; then
                echo "Calibration | ${DATASET_SLUG} | eval:${EVAL_SLUG} | judge:${JUDGE_MODEL} | documents"
                python src/translation/api/run_evaluate_translations.py \
                    --source_file_paths "${DOC_SOURCE}" \
                    --prompt_file_name "$EVAL_PROMPT_FILE" \
                    --output_dir "$EVAL_OUTPUT_DIR" \
                    --model_name "$JUDGE_MODEL" \
                    --limit $LIMIT \
                    --english_key "segment_text" \
                    --hebrew_key "translation" \
                    $FORCE_FLAG \
                    ${WORKERS:+--workers $WORKERS}
            fi

        done
    done
done

# ── Collect and analyze ────────────────────────────────────────────────────────
MASTER_CSV="outputs/translation/BeIR/results_${EXPERIMENT_NAME}.csv"
ANALYSIS_DIR="outputs/translation/BeIR/analysis_${EXPERIMENT_NAME}"

echo ""
echo "Collecting calibration results → $MASTER_CSV"
python src/translation/collect_results.py \
    --results_dir "$CANDIDATES_BASE" \
    --output_path "$MASTER_CSV" \
    --calibration

echo ""
echo "Running calibration analysis → $ANALYSIS_DIR"
python src/translation/analyze_eval_prompts.py \
    --results_path "$MASTER_CSV" \
    --output_dir "$ANALYSIS_DIR"

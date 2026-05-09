#!/bin/bash -i

# Translates BeIR candidates for a given experiment configuration.
#
# Usage:
#   bash scripts/translation/run_translation_pipeline.sh [config_file]
#
# Default config: config/experiments/translation_eval.sh
#
# Continuation: already-translated files are skipped automatically.
#   Re-run the same command after a crash to resume.
#   Set FORCE=true in the config to re-translate everything.
#
# Workers: set WORKERS=N in the config to run N (dataset × model × prompt)
#   combinations simultaneously. Each job translates its rows serially.

CONFIG="${1:-config/experiments/translation_eval.sh}"
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
OUTER_WORKERS="${WORKERS:-1}"

echo ""
echo "Experiment        : $EXPERIMENT_NAME"
echo "Translation models: ${TRANSLATION_MODELS[*]}"
echo "Prompt strategies : ${PROMPT_SLUGS[*]}"
echo "Datasets          : ${DATASET_SLUGS[*]}"
echo "Workers           : $OUTER_WORKERS"
echo ""

# ── Job pool ──────────────────────────────────────────────────────────────────
# Tracks background PIDs; blocks when the pool is full.
_pids=()

_cleanup() {
    [ "${#_pids[@]}" -gt 0 ] && kill "${_pids[@]}" 2>/dev/null
    exit 1
}
trap _cleanup INT TERM

_wait_for_slot() {
    while [ "${#_pids[@]}" -ge "$OUTER_WORKERS" ]; do
        local alive=()
        for pid in "${_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        _pids=("${alive[@]}")
        [ "${#_pids[@]}" -ge "$OUTER_WORKERS" ] && sleep 1
    done
}

_run() {
    if [ "$OUTER_WORKERS" -gt 1 ]; then
        _wait_for_slot
        python "$@" &
        _pids+=($!)
    else
        python "$@"
    fi
}

# ── Translation loop ──────────────────────────────────────────────────────────
_total=$(( ( ${#QUERY_PROMPT_SLUGS[@]} + ${#DOCUMENT_PROMPT_SLUGS[@]} ) * ${#DATASET_SLUGS[@]} * ${#TRANSLATION_MODELS[@]} ))
_job=0

# ── Queries ────────────────────────────────────────────────────────────────────
for PROMPT_SLUG in "${QUERY_PROMPT_SLUGS[@]}"; do
    var="QUERY_PROMPT_${PROMPT_SLUG}"
    PROMPT_FILE="${!var}"

    for DATASET_SLUG in "${DATASET_SLUGS[@]}"; do
        for MODEL_NAME in "${TRANSLATION_MODELS[@]}"; do
            _job=$(( _job + 1 ))
            OUTPUT_DIR="${CANDIDATES_BASE}/${DATASET_SLUG}/${MODEL_NAME}/${PROMPT_SLUG}"

            QUERY_SOURCE="${CANDIDATES_BASE}/${DATASET_SLUG}/queries.csv"
            if [ -f "$QUERY_SOURCE" ]; then
                echo "[${_job}/${_total}] Translate queries   | ${DATASET_SLUG} | ${MODEL_NAME} | ${PROMPT_SLUG}"
                _run src/translation/api/run_translation_pipeline.py \
                    --source_file_paths "${QUERY_SOURCE}" \
                    --prompt_file_name "$PROMPT_FILE" \
                    --output_dir "$OUTPUT_DIR" \
                    --model_name "$MODEL_NAME" \
                    --limit $LIMIT \
                    --english_key "text" \
                    --hebrew_key "translation" \
                    --context_key "context_text" \
                    $FORCE_FLAG
            fi
        done
    done
done

# ── Documents ──────────────────────────────────────────────────────────────────
for PROMPT_SLUG in "${DOCUMENT_PROMPT_SLUGS[@]}"; do
    var="DOCUMENT_PROMPT_${PROMPT_SLUG}"
    PROMPT_FILE="${!var}"

    for DATASET_SLUG in "${DATASET_SLUGS[@]}"; do
        for MODEL_NAME in "${TRANSLATION_MODELS[@]}"; do
            _job=$(( _job + 1 ))
            OUTPUT_DIR="${CANDIDATES_BASE}/${DATASET_SLUG}/${MODEL_NAME}/${PROMPT_SLUG}"

            DOC_SOURCE="${CANDIDATES_BASE}/${DATASET_SLUG}/documents.csv"
            if [ -f "$DOC_SOURCE" ]; then
                echo "[${_job}/${_total}] Translate documents | ${DATASET_SLUG} | ${MODEL_NAME} | ${PROMPT_SLUG}"
                _run src/translation/api/run_translation_pipeline.py \
                    --source_file_paths "${DOC_SOURCE}" \
                    --prompt_file_name "$PROMPT_FILE" \
                    --output_dir "$OUTPUT_DIR" \
                    --model_name "$MODEL_NAME" \
                    --limit $LIMIT \
                    --english_key "segment_text" \
                    --hebrew_key "translation" \
                    $FORCE_FLAG
            fi
        done
    done
done

# Wait for all remaining background jobs
wait

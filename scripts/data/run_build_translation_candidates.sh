#!/bin/bash -i

# Builds translation candidate sets for a given experiment configuration.
#
# Usage:
#   bash scripts/data/run_build_translation_candidates.sh [config_file]
#
# Default config: config/experiments/translation_eval.sh

CONFIG="${1:-config/experiments/translation_eval.sh}"
if [ ! -f "$CONFIG" ]; then echo "Config not found: $CONFIG"; exit 1; fi
echo "Using config: $CONFIG"
source "$CONFIG"

echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr

if [ -f .env ]; then set -a; source .env; set +a; else echo ".env not found!"; exit 1; fi

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Convert DATASET_SLUGS (BeIR_msmarco) to dataset names (BeIR/msmarco)
DATASET_NAMES=()
for slug in "${DATASET_SLUGS[@]}"; do
    DATASET_NAMES+=("${slug/_//}")
done

echo ""
echo "Experiment  : $EXPERIMENT_NAME"
echo "Datasets    : ${#DATASET_NAMES[@]}"
echo "Samples     : $NUM_SAMPLES per dataset"
echo "Seg. tokens : $MAX_SEGMENT_TOKENS"
echo "Output      : $CANDIDATES_BASE"
echo ""

[ "${FORCE}" = "true" ] && FORCE_FLAG="--force" || FORCE_FLAG=""

python src/translation/build_translation_candidates.py \
    --dataset_names "${DATASET_NAMES[@]}" \
    --num_samples $NUM_SAMPLES \
    --max_document_segment_tokens $MAX_SEGMENT_TOKENS \
    --model_name "$TOKENIZER_MODEL" \
    --output_path "$CANDIDATES_BASE" \
    --random_seed $RANDOM_SEED \
    ${WORKERS:+--workers $WORKERS} \
    $FORCE_FLAG

echo ""
echo "Done. Per-dataset outputs written to ${CANDIDATES_BASE}/<dataset_slug>/"

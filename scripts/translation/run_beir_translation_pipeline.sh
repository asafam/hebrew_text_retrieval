#!/bin/bash -i
# Runs the BeIR translation pipeline from a YAML config file.
#
# Usage:
#   ./scripts/translation/run_beir_translation_pipeline.sh
#   ./scripts/translation/run_beir_translation_pipeline.sh config/translation/my_config.yaml

set -euo pipefail

# ── Conda activation ─────────────────────────────────────────────────────────
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr

# ── Environment variables ─────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo ".env file not found!"
    exit 1
fi

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# ── Arguments ─────────────────────────────────────────────────────────────────
CONFIG_FILE="${1:-config/translation/beir_translation_zeroshot_gpt4o_mini.yaml}"

echo "Running BeIR translation pipeline"
echo "Config: $CONFIG_FILE"

python src/translation/api/run_beir_translation_pipeline.py --config "$CONFIG_FILE"

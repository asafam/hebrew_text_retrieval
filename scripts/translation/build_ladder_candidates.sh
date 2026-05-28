#!/usr/bin/env bash
# Thin wrapper: sets up the conda environment and delegates to the Python
# orchestrator which reads all configuration from the YAML.
#
# Usage:
#   bash scripts/translation/build_ladder_candidates.sh
#   bash scripts/translation/build_ladder_candidates.sh --config config/translation/candidates.yaml
#   bash scripts/translation/build_ladder_candidates.sh --dataset nfcorpus
#   bash scripts/translation/build_ladder_candidates.sh --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr

if [ -f .env ]; then set -a; source .env; set +a; fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/build_ladder_candidates.py" "$@"

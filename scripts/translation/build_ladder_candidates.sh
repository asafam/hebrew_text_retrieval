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

_conda_sh=""
for _candidate in \
    "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$_candidate" ]; then _conda_sh="$_candidate"; break; fi
done
if [ -z "$_conda_sh" ]; then
    echo "ERROR: could not find conda.sh — set CONDA_PREFIX or add conda to PATH" >&2
    exit 1
fi
source "$_conda_sh"
conda activate htr

if [ -f .env ]; then set -a; source .env; set +a; fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/build_ladder_candidates.py" "$@"

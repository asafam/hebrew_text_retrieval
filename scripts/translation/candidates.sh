#!/usr/bin/env bash
# Build sharded BeIR translation candidates (step 1 of the translation pipeline).
#
# Sibling of translate.sh. Reads all settings from the candidates YAML and
# delegates to the Python orchestrator, which fans out one subprocess per
# dataset and shows a live progress table.
#
# Usage:
#   bash scripts/translation/candidates.sh                                  # all datasets in the config
#   bash scripts/translation/candidates.sh --dataset nfcorpus               # one dataset (partial match)
#   bash scripts/translation/candidates.sh --split test                     # qrel split filter (default: all)
#   bash scripts/translation/candidates.sh --force                          # rebuild existing
#   bash scripts/translation/candidates.sh --config config/translation/candidates.yaml
#
# Output: outputs/translation/candidates/<slug>/{queries,documents}_shard_NNN.csv + shard_manifest.json
# (the translate.sh run later consumes these — both phases share one candidate source).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── conda ──────────────────────────────────────────────────────────────────────
_conda_sh=""
for _candidate in \
    "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$_candidate" ]; then _conda_sh="$_candidate"; break; fi
done
if [ -z "$_conda_sh" ]; then
    echo "ERROR: could not find conda.sh — add conda to PATH" >&2
    exit 1
fi
source "$_conda_sh"
conda activate htr

if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

CONFIG="config/translation/candidates.yaml"
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

python "$SCRIPT_DIR/build_ladder_candidates.py" --config "$CONFIG" "${PASS[@]}"

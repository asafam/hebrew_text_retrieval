#!/usr/bin/env bash
# BeIR Full Corpus Translation — portable entry point.
#
# Usage:
#   bash scripts/translation/run_full_corpus.sh query            # translate queries
#   bash scripts/translation/run_full_corpus.sh document         # translate documents
#   bash scripts/translation/run_full_corpus.sh both             # translate both (default)
#   bash scripts/translation/run_full_corpus.sh query --yes      # skip cost confirmation
#   bash scripts/translation/run_full_corpus.sh query --config config/translation/full_corpus.yaml
#
# Requires:
#   - Python 3.10+ with pip install -r requirements-translation.txt
#   - .env file with OPENAI_API_KEY, OPENAI_API_ORG, OPENAI_PROJECT, ANTHROPIC_API_KEY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "WARNING: .env file not found. Ensure API keys are set in the environment."
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Parse arguments
TEXT_TYPE="${1:-both}"
CONFIG="config/translation/full_corpus.yaml"
EXTRA_ARGS=""

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --yes|-y) EXTRA_ARGS="$EXTRA_ARGS --yes"; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "BeIR Full Corpus Translation"
echo "  Config:    $CONFIG"
echo "  Text type: $TEXT_TYPE"
echo "  Root:      $PROJECT_ROOT"
echo ""

python -m translation.api.run_beir_translation_pipeline \
    --config "$CONFIG" \
    --text-type "$TEXT_TYPE" \
    $EXTRA_ARGS

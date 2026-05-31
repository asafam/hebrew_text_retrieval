#!/usr/bin/env bash
# Translate BeIR datasets to Hebrew (step 2 of the translation pipeline).
#
# Sibling of candidates.sh. One entry point for both translation phases:
#
#   --pilot     small synchronous sample per dataset + LLM-as-a-judge QA gate
#               (run_beir_batch_gcs pilot). Use this to sanity-check quality
#               before committing to the full corpus.
#   (default)   full-corpus shard-ladder: translate every shard via Vertex
#               batch, judging after each cadence step (run_beir_ladder_pipeline).
#
# Usage:
#   bash scripts/translation/translate.sh                              # full corpus, all datasets
#   bash scripts/translation/translate.sh --dataset BeIR/nfcorpus      # single dataset
#   bash scripts/translation/translate.sh --resume                     # resume an interrupted run
#   bash scripts/translation/translate.sh --dry-run                    # show the shard plan, no work
#   bash scripts/translation/translate.sh --pilot                      # pilot all datasets
#   bash scripts/translation/translate.sh --pilot --dataset BeIR/nfcorpus --yes
#
# Auth: uses Vertex AI (gcloud ADC). GEMINI_API_KEY is force-unset below —
# the batch/pilot path requires ADC, not an API key. Run once if needed:
#   gcloud auth application-default login

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
# Vertex AI ADC only — the pilot/batch paths reject a stray GEMINI_API_KEY.
unset GEMINI_API_KEY
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ── args ─────────────────────────────────────────────────────────────────────
CONFIG="config/translation/full_corpus.yaml"
PILOT=0
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot)  PILOT=1; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

# ── ADC preflight ────────────────────────────────────────────────────────────
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    echo "ERROR: Vertex AI ADC not available." >&2
    echo "Run: gcloud auth application-default login" >&2
    exit 1
fi

if [[ "$PILOT" -eq 1 ]]; then
    echo "[translate] PILOT phase — config=$CONFIG"
    exec python -m translation.api.run_beir_batch_gcs --config "$CONFIG" pilot "${PASS[@]}"
else
    echo "[translate] FULL CORPUS ladder — config=$CONFIG"
    exec python -m translation.api.run_beir_ladder_pipeline --config "$CONFIG" "${PASS[@]}"
fi

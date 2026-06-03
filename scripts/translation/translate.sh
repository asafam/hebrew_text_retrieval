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
#   bash scripts/translation/translate.sh                              # all datasets, serial
#   bash scripts/translation/translate.sh --parallel                   # all datasets, in parallel
#   bash scripts/translation/translate.sh --parallel --resume          # resume all in parallel
#   bash scripts/translation/translate.sh --dataset BeIR/nfcorpus      # single dataset
#   bash scripts/translation/translate.sh --resume                     # resume an interrupted run
#   bash scripts/translation/translate.sh --dry-run                    # show the shard plan, no work
#   bash scripts/translation/translate.sh --pilot                      # pilot all datasets
#   bash scripts/translation/translate.sh --pilot --dataset BeIR/nfcorpus --yes
#
# --parallel fans out one background process per dataset (each writes its own
# progress.<slug>.json / run.<slug>.log / qa_scores.<slug>.csv). A QA stop on
# one dataset never blocks the others.
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
PARALLEL=0
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot)    PILOT=1;    shift ;;
        --parallel) PARALLEL=1; shift ;;
        --config)   CONFIG="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

# ── ADC preflight ────────────────────────────────────────────────────────────
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    echo "ERROR: Vertex AI ADC not available." >&2
    echo "Run: gcloud auth application-default login" >&2
    exit 1
fi

# ── Parallel fan-out ─────────────────────────────────────────────────────────
if [[ "$PARALLEL" -eq 1 && "$PILOT" -eq 0 ]]; then
    # Read dataset list from config and spawn one process per dataset.
    # Each process is isolated (progress.<slug>.json etc.) so they never block
    # each other. A QA stop on one dataset leaves the others running.
    mapfile -t DATASETS < <(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for d in cfg["datasets"]["names"]:
    print(d)
PY
)
    echo "[translate] PARALLEL ladder — ${#DATASETS[@]} datasets — config=$CONFIG"
    pids=()
    logs=()
    for dataset in "${DATASETS[@]}"; do
        slug=$(python -c "d='$dataset'; print(d.replace('BeIR/','BeIR_').replace('/','_'))")
        logfile="$PROJECT_ROOT/outputs/translation/parallel_${slug}.log"
        logs+=("$logfile")
        bash "$SCRIPT_DIR/translate.sh" \
            --config "$CONFIG" --dataset "$dataset" "${PASS[@]}" \
            >"$logfile" 2>&1 &
        pids+=($!)
        echo "  [$dataset] PID=$!  log=$logfile"
    done

    echo
    echo "[translate] All ${#pids[@]} datasets launched. Waiting..."
    failed=0
    for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        dataset="${DATASETS[$i]}"
        if wait "$pid"; then
            echo "  [$dataset] DONE (PID $pid)"
        else
            echo "  [$dataset] FAILED or STOPPED (PID $pid) — check ${logs[$i]}"
            failed=$((failed + 1))
        fi
    done
    echo
    if [[ "$failed" -gt 0 ]]; then
        echo "[translate] $failed dataset(s) stopped/failed. Check logs above."
        exit 1
    else
        echo "[translate] All datasets complete."
    fi
    exit 0
fi

# ── Single dataset or serial all ─────────────────────────────────────────────
if [[ "$PILOT" -eq 1 ]]; then
    echo "[translate] PILOT phase — config=$CONFIG"
    exec python -m translation.api.run_beir_batch_gcs --config "$CONFIG" pilot "${PASS[@]}"
else
    echo "[translate] FULL CORPUS ladder — config=$CONFIG"
    exec python -m translation.api.run_beir_ladder_pipeline --config "$CONFIG" "${PASS[@]}"
fi

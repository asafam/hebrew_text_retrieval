#!/usr/bin/env bash
# Build sharded translation candidates for the ladder pipeline.
#
# Each BeIR dataset is processed in its own background job. Shard sizes are
# taken from config/translation/full_corpus.yaml (datasets.shard_sizes).
#
# Usage:
#   bash scripts/translation/build_ladder_candidates.sh            # all 15 datasets
#   bash scripts/translation/build_ladder_candidates.sh nfcorpus   # one dataset (slug suffix)
#   bash scripts/translation/build_ladder_candidates.sh --force    # rebuild existing
#
# Outputs:  outputs/translation/candidates/<dataset_slug>/
#           outputs/translation/candidates/<dataset_slug>/shard_manifest.json
#
# Logs:     outputs/translation/candidates/logs/<dataset_slug>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────

TOKENIZER_MODEL="gpt-4o-mini-2024-07-18"
OUTPUT_PATH="outputs/translation/candidates"
MAX_SEGMENT_TOKENS=512
RANDOM_STATE=42
FORCE_FLAG=""

# ── Shard sizes (must match config/translation/full_corpus.yaml) ──────────────

declare -A SHARD_SIZES
SHARD_SIZES["BeIR/nfcorpus"]=500
SHARD_SIZES["BeIR/scifact"]=1000
SHARD_SIZES["BeIR/arguana"]=1000
SHARD_SIZES["BeIR/scidocs"]=5000
SHARD_SIZES["BeIR/fiqa"]=10000
SHARD_SIZES["BeIR/trec-covid"]=10000
SHARD_SIZES["BeIR/webis-touche2020"]=25000
SHARD_SIZES["BeIR/cqadupstack"]=25000
SHARD_SIZES["BeIR/quora"]=25000
SHARD_SIZES["BeIR/nq"]=100000
SHARD_SIZES["BeIR/hotpotqa"]=100000
SHARD_SIZES["BeIR/dbpedia-entity"]=100000
SHARD_SIZES["BeIR/fever"]=100000
SHARD_SIZES["BeIR/climate-fever"]=100000
SHARD_SIZES["BeIR/msmarco"]=100000

# Ordered small → large so small datasets finish first and the pipeline can
# start while large ones are still being built.
ALL_DATASETS=(
    "BeIR/nfcorpus"
    "BeIR/scifact"
    "BeIR/arguana"
    "BeIR/scidocs"
    "BeIR/fiqa"
    "BeIR/trec-covid"
    "BeIR/webis-touche2020"
    "BeIR/cqadupstack"
    "BeIR/quora"
    "BeIR/nq"
    "BeIR/hotpotqa"
    "BeIR/dbpedia-entity"
    "BeIR/fever"
    "BeIR/climate-fever"
    "BeIR/msmarco"
)

# ── Argument parsing ──────────────────────────────────────────────────────────

FILTER=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_FLAG="--force" ;;
        --*)     echo "Unknown flag: $arg"; exit 1 ;;
        *)       FILTER="$arg" ;;   # partial slug match, e.g. "nfcorpus"
    esac
done

DATASETS=()
for d in "${ALL_DATASETS[@]}"; do
    if [ -z "$FILTER" ] || [[ "$d" == *"$FILTER"* ]]; then
        DATASETS+=("$d")
    fi
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No datasets matched filter: '$FILTER'"
    exit 1
fi

# ── Setup ─────────────────────────────────────────────────────────────────────

LOG_DIR="$OUTPUT_PATH/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "Building ladder candidates"
echo "  Datasets  : ${#DATASETS[@]}"
echo "  Output    : $OUTPUT_PATH"
echo "  Logs      : $LOG_DIR"
echo "  Force     : ${FORCE_FLAG:-(no)}"
echo ""

# ── Launch one background job per dataset ────────────────────────────────────

declare -A PIDS
declare -A LOGS

for dataset in "${DATASETS[@]}"; do
    slug="${dataset/\//_}"
    shard_size="${SHARD_SIZES[$dataset]:-10000}"
    log_file="$LOG_DIR/${slug}.log"
    LOGS["$dataset"]="$log_file"

    # Skip if manifest already exists (unless --force)
    manifest="$OUTPUT_PATH/${slug}/shard_manifest.json"
    if [ -z "$FORCE_FLAG" ] && [ -f "$manifest" ]; then
        echo "  SKIP  $dataset  (manifest exists — use --force to rebuild)"
        continue
    fi

    echo "  START $dataset  (shard_size=$shard_size)  → $log_file"

    python -m translation.build_translation_candidates \
        --dataset_names "$dataset" \
        --model_name_or_path "$TOKENIZER_MODEL" \
        --output_path "$OUTPUT_PATH" \
        --max_document_segment_tokens "$MAX_SEGMENT_TOKENS" \
        --random_state "$RANDOM_STATE" \
        --shard-size "$shard_size" \
        $FORCE_FLAG \
        > "$log_file" 2>&1 &

    PIDS["$dataset"]=$!
done

# ── Wait and report ───────────────────────────────────────────────────────────

if [ ${#PIDS[@]} -eq 0 ]; then
    echo ""
    echo "All datasets already built. Run with --force to rebuild."
    echo ""
    exit 0
fi

echo ""
echo "Waiting for ${#PIDS[@]} job(s)..."
echo ""

FAILED=()
for dataset in "${!PIDS[@]}"; do
    pid="${PIDS[$dataset]}"
    slug="${dataset/\//_}"
    manifest="$OUTPUT_PATH/${slug}/shard_manifest.json"

    if wait "$pid"; then
        shards=$(python -c "import json; m=json.load(open('$manifest')); print(len(m['types']['queries']), 'query shards,', len(m['types']['documents']), 'doc shards')" 2>/dev/null || echo "?")
        echo "  OK    $dataset  ($shards)"
    else
        echo "  FAIL  $dataset  — see ${LOGS[$dataset]}"
        FAILED+=("$dataset")
    fi
done

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED datasets (${#FAILED[@]}):"
    for d in "${FAILED[@]}"; do
        echo "  $d"
        tail -5 "${LOGS[$d]}" | sed 's/^/    /'
    done
    echo ""
    exit 1
else
    echo "All ${#PIDS[@]} dataset(s) built successfully."
    echo ""
    echo "Next step:"
    echo "  python -m translation.api.run_beir_ladder_pipeline \\"
    echo "      --config config/translation/full_corpus.yaml --dry-run"
    echo ""
fi

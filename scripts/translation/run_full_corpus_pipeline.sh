#!/bin/bash
# Full corpus translation pipeline with mid-phase QA gates.
#
# Runs Phase 1 → QA → Phase 2 → QA → Phase 3
# Halts automatically if QA detects score degradation (exit code 2).
#
# Usage:
#   bash scripts/translation/run_full_corpus_pipeline.sh
#   bash scripts/translation/run_full_corpus_pipeline.sh --start-phase 2   # resume from phase 2

set -euo pipefail

# ── Setup ─────────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr

if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

START_PHASE="${1:-1}"
if [[ "$1" == "--start-phase" ]]; then START_PHASE="$2"; fi

BASELINE_CSV="outputs/translation/BeIR/results_translation_eval.csv"
BASELINE_MODEL="gpt-5.4-mini"
BASELINE_PROMPT="zeroshot_nocontext"
JUDGE_MODEL="claude-sonnet-4-6"
SAMPLE_SIZE=25
RUNS_BASE="outputs/beir_translation/full_corpus"

run_qa() {
    local phase="$1"
    local run_dir="$2"
    local report="$RUNS_BASE/qa_p${phase}.json"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  QA Gate — Phase ${phase}"
    echo "════════════════════════════════════════════════════════"

    python src/translation/qa_phase.py \
        --phase-run-dir  "$run_dir" \
        --baseline-csv   "$BASELINE_CSV" \
        --baseline-model "$BASELINE_MODEL" \
        --baseline-prompt "$BASELINE_PROMPT" \
        --judge-model    "$JUDGE_MODEL" \
        --sample-size    "$SAMPLE_SIZE" \
        --output-report  "$report" \
        --workers 4
    EXIT=$?

    if [ $EXIT -eq 2 ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  QA FAILED — score degradation detected in Phase ${phase}  ║"
        echo "║  Review $report  ║"
        echo "║  Fix the issue before re-running from --start-phase ${phase} ║"
        echo "╚══════════════════════════════════════════════════════╝"
        exit 2
    elif [ $EXIT -eq 1 ]; then
        echo "  QA WARN: coverage issues detected but scores OK — continuing."
    else
        echo "  QA PASSED."
    fi
}

run_phase() {
    local phase="$1"
    local config="$2"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Phase ${phase}: $config"
    echo "════════════════════════════════════════════════════════"
    python src/translation/api/run_beir_translation_pipeline.py --config "$config"
}

# ── Phase 1: Small datasets ───────────────────────────────────────────────────
if [ "$START_PHASE" -le 1 ]; then
    run_phase 1 "config/translation/full_corpus_p1_small.yaml"
    run_qa 1 "$RUNS_BASE/full_corpus_p1_small"
fi

# ── Phase 2: Medium datasets ──────────────────────────────────────────────────
if [ "$START_PHASE" -le 2 ]; then
    run_phase 2 "config/translation/full_corpus_p2_medium.yaml"
    run_qa 2 "$RUNS_BASE/full_corpus_p2_medium"
fi

# ── Phase 3: Large datasets ───────────────────────────────────────────────────
if [ "$START_PHASE" -le 3 ]; then
    run_phase 3 "config/translation/full_corpus_p3_large.yaml"
    run_qa 3 "$RUNS_BASE/full_corpus_p3_large"
fi

echo ""
echo "All phases complete. Full corpus translated and QA passed."

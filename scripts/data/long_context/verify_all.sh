#!/bin/bash
# Verify every built long-context dataset against every condition.
#
# Exits non-zero if any check fails, so this is safe to gate the eval on. Each dataset x
# condition runs the full suite (gold intact at recorded offsets, corpus size constant across
# rungs, infix nesting, no qrel positive used as filler, position distribution, ...).
#
# Usage:
#   bash scripts/data/long_context/verify_all.sh
#   OUT_ROOT=/some/other/path bash scripts/data/long_context/verify_all.sh

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

OUT_ROOT="${OUT_ROOT:-data/retrieval/beir_longctx/v1}"
CONDITIONS="${CONDITIONS:-random bm25}"

fail=0
summary=()

for ds_dir in "$OUT_ROOT"/*/; do
    ds=$(basename "$ds_dir")
    if [ ! -f "$ds_dir/manifest.json" ]; then
        echo "SKIP $ds  (no manifest -- build incomplete)"
        summary+=("$ds: INCOMPLETE")
        fail=1
        continue
    fi
    for cond in $CONDITIONS; do
        echo
        echo "======================================================================"
        echo "  $ds  /  condition=$cond"
        echo "======================================================================"
        # Always print every FAIL line. An earlier version piped the failing branch through
        # `tail -30`, which silently dropped the failures when they appeared early in the
        # output -- the summary said "FAIL (2 failed)" while the saved log showed none.
        if out=$(python src/data/long_context/verify.py \
                    --dataset_dir "$ds_dir" --condition "$cond" 2>&1); then
            n=$(grep -c '\[PASS\]' <<<"$out")
            grep -E 'OK -|corpus size constant|infix nesting' <<<"$out"
            summary+=("$ds/$cond: PASS ($n checks)")
        else
            nf=$(grep -c '\[FAIL\]' <<<"$out" || true)
            echo "--- all failing checks ---"
            grep '\[FAIL\]' <<<"$out" || echo "  (none -- the verifier itself crashed)"
            echo "--- tail of output ---"
            tail -12 <<<"$out"
            summary+=("$ds/$cond: FAIL ($nf failed)")
            fail=1
        fi
    done
done

echo
echo "======================================================================"
echo "  SUMMARY"
echo "======================================================================"
for line in "${summary[@]}"; do echo "  $line"; done
echo
if [ "$fail" -eq 0 ]; then
    echo "All datasets verified. Safe to run the eval."
else
    echo "VERIFICATION FAILED -- do not run the eval until resolved."
fi
exit "$fail"

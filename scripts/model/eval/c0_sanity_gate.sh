#!/bin/bash
#SBATCH --job-name=c0_gate
#SBATCH --output=logs/slurm/c0_gate_%j.out
#SBATCH --error=logs/slurm/c0_gate_%j.err
#SBATCH --partition=L4-12h
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# c0 SANITY GATE for the long-context benchmark.
#
# c0 is the *unpadded* rung: the original translated-BeIR corpora with no welding at all. It is
# therefore identical to the existing zero-shot eval, which means it has a known right answer
# and can falsify the harness before any long-context number is interpreted.
#
# The gate re-encodes from scratch (--force_reencode) rather than reusing cached embeddings --
# reusing the cache would only prove the cache is readable, not that the pipeline reproduces.
#
# PASS CRITERIA
#   1. Ranking on every dataset: mE5-large > mE5-base ~ NDB >> HMB-base-final
#   2. Per-dataset NDCG@10 within +-0.01 of the values in results.json (NOT the values in
#      EVALUATION.md -- see below)
#
# TARGETS (from results.json, post-IDCG-fix):
#            scifact  scidocs   fiqa
#   mE5-large  0.581    0.139   0.335
#   mE5-base   0.549    0.125   0.241
#   NDB        0.501    0.093   0.288
#   HMB-final  0.312    0.031   0.055
#
# NOTE ON TARGETS: EVALUATION.md lists higher numbers for scidocs and fiqa (e.g. mE5-large fiqa
# 0.382, scidocs 0.232). Those predate the compute_metrics fix, which stopped normalising the
# ideal DCG against the retrieved slice. The discrepancy scales with positives-per-query --
# ~0.000 on scifact (1.1/query), ~0.045 on fiqa (2.6), ~0.09 on scidocs (4.9) -- which is the
# signature of that bug. Gate against results.json.

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

RESULTS_DIR="${RESULTS_DIR:-outputs/eval/c0_gate}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DATASETS="${DATASETS:-BeIR_scifact BeIR_scidocs BeIR_fiqa}"

mkdir -p "$RESULTS_DIR" logs/slurm

HMB_CKPT="outputs/models/dual_encoder/cls_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model"
NDB_CKPT="outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model"

# label|model_path|pooling|batch
ARMS=(
  "c0-mE5-large|intfloat/multilingual-e5-large|mean|64"
  "c0-mE5-base|intfloat/multilingual-e5-base|mean|128"
  "c0-NDB|${NDB_CKPT}|cls|32"
  "c0-HMB-final|${HMB_CKPT}|cls|64"
)

for ds in $DATASETS; do
    CORPUS_DIR=$(find outputs/translation/runs -type d -path "*/${ds}/beir" | head -1)
    if [ -z "$CORPUS_DIR" ]; then
        echo "!! no corpus dir for $ds -- skipping"; continue
    fi
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r label model pooling batch <<< "$arm"
        echo
        echo "=== $label / $ds ==="
        # Use --output_file, not --output_dir. --output_dir writes a FLAT
        # "$RESULTS_DIR/results.json", so every arm overwrites the previous one and only the
        # last survives -- which is exactly what happened on the first run of this gate.
        python src/model/eval/eval_beir_retrieval_zeroshot.py \
            --model_name_or_path "$model" \
            --model_label "$label" \
            --corpus_dir "$CORPUS_DIR" \
            --output_file "$RESULTS_DIR/$label/$ds/results.json" \
            --batch_size "$batch" \
            --max_length "$MAX_LENGTH" \
            --pooling "$pooling" \
            --force_reencode \
          || echo "!! FAILED: $label / $ds"
    done
done

echo
echo "=== c0 gate results ==="
# Only score the datasets this job actually ran, so a per-dataset job gives a meaningful
# verdict instead of reporting the other two as MISSING.
export GATE_DATASETS="$DATASETS"
python - <<'PY'
import json, glob, os
TARGET = {
  ("c0-mE5-large","BeIR_scifact"):0.581, ("c0-mE5-large","BeIR_scidocs"):0.139, ("c0-mE5-large","BeIR_fiqa"):0.335,
  ("c0-mE5-base","BeIR_scifact"):0.549,  ("c0-mE5-base","BeIR_scidocs"):0.125,  ("c0-mE5-base","BeIR_fiqa"):0.241,
  ("c0-NDB","BeIR_scifact"):0.501,       ("c0-NDB","BeIR_scidocs"):0.093,       ("c0-NDB","BeIR_fiqa"):0.288,
  ("c0-HMB-final","BeIR_scifact"):0.312, ("c0-HMB-final","BeIR_scidocs"):0.031, ("c0-HMB-final","BeIR_fiqa"):0.055,
}
TOL = 0.01
only = set(os.environ.get("GATE_DATASETS", "").split()) or None
if only:
    TARGET = {k: v for k, v in TARGET.items() if k[1] in only}
got, fails = {}, 0
for f in glob.glob("outputs/eval/c0_gate/*/*/results.json"):
    d = json.load(open(f))
    got[(d["model"], d["dataset"])] = d["metrics"]["ndcg_at_10"]
print(f"{'arm':14s} {'dataset':9s} {'got':>7s} {'target':>7s} {'delta':>7s}  verdict")
for k in sorted(TARGET):
    t = TARGET[k]
    if k not in got:
        print(f"{k[0]:14s} {k[1].replace('BeIR_',''):9s} {'-':>7s} {t:7.3f} {'-':>7s}  MISSING"); fails += 1; continue
    g = got[k]; dl = g - t
    ok = abs(dl) <= TOL
    fails += (not ok)
    print(f"{k[0]:14s} {k[1].replace('BeIR_',''):9s} {g:7.3f} {t:7.3f} {dl:+7.3f}  {'ok' if ok else 'MISMATCH'}")
print()
print("C0 GATE: PASS" if not fails else f"C0 GATE: FAIL ({fails} cells)")
raise SystemExit(1 if fails else 0)
PY

#!/bin/bash
#SBATCH --job-name=build_longctx
#SBATCH --output=logs/slurm/build_longctx_%j.out
#SBATCH --error=logs/slurm/build_longctx_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Build the translated-BeIR long-context benchmark.
#
# CPU ONLY -- no GPU is requested or needed. Welding is string slicing, filler selection is
# sparse BM25, and the only tokenizer use is measurement. Just the eval encode step needs a
# GPU, which matters here because the B200 partition is usually contended while
# cpu192G-48h/cpu512G-48h sit largely idle: the corpora can be built and verified while
# waiting for GPU time.
#
# Measured cost on the largest corpus (fiqa, 57,600 docs): BM25 index 4.8s, top-64 for every
# seed ~2 min, 0.46GB peak. The 8h wall and 64G are generous headroom, not a requirement.
#
# Datasets are built cheapest-first (scifact -> scidocs -> fiqa) so a mistake surfaces on the
# 5,183-doc corpus rather than after an hour on fiqa.
#
# Usage:
#   sbatch scripts/data/long_context/build_beir_longctx.sh
#   sbatch --export=ALL,DATASETS="BeIR_scifact" scripts/data/long_context/build_beir_longctx.sh
#   bash   scripts/data/long_context/build_beir_longctx.sh          # no SLURM

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
# htr is the only env with scipy + rank_bm25; bert24 has pytest but not scipy, and biu has a
# NumPy ABI conflict that breaks `import transformers` outright.
conda activate htr

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

OUT_ROOT="${OUT_ROOT:-data/retrieval/beir_longctx/v1}"
DATASETS="${DATASETS:-}"
CONDITIONS="${CONDITIONS:-random bm25}"
EXTRA="${EXTRA:-}"

echo "Out root:    $OUT_ROOT"
echo "Datasets:    ${DATASETS:-<all, cheapest first>}"
echo "Conditions:  $CONDITIONS"
echo

mkdir -p logs/slurm "$OUT_ROOT"

# Module self-tests first: both are cheap and both have caught real defects. The BM25 one
# asserts exact equivalence with rank_bm25; the verify one asserts each corpus check still
# fires on the bug it targets.
echo "== self-tests =="
python src/data/long_context/bm25.py --self-test
python src/data/long_context/verify.py --self-test
echo

echo "== build =="
# shellcheck disable=SC2086
python -u src/data/long_context/build_benchmark.py \
    --out_root "$OUT_ROOT" \
    --conditions $CONDITIONS \
    ${DATASETS:+--datasets $DATASETS} \
    $EXTRA

echo
echo "Done. Verify a built dataset with:"
echo "  python src/data/long_context/verify.py --dataset_dir $OUT_ROOT/BeIR_scifact"

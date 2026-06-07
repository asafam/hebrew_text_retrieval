#!/bin/bash -i
#SBATCH --job-name=eval_hmb_beir_lc
#SBATCH --output=logs/slurm/eval_hmb_beir_lc_%j.out
#SBATCH --error=logs/slurm/eval_hmb_beir_lc_%j.err
#SBATCH --partition=A100-4h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

MODEL_PATH="outputs/models/dual_encoder/cls_pooling/beir_hebrew/hebmodernbert/ModernBERT-Hebrew-base_v2"
TOKENIZER_PATH="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_v2"
OUTPUT_DIR="outputs/eval/dual_encoder/beir_long_context/hebmodernbert/ModernBERT-Hebrew-base_v2"
K_VALUES="0,1,3,5"
BATCH_SIZE=128
MAX_LENGTH=512

run_eval() {
    local DATASET="$1"
    echo ""
    echo "=== $DATASET ==="
    python src/model/eval/eval_long_context_hn_patching.py \
        --model_name_or_path     "$MODEL_PATH" \
        --tokenizer_name_or_path "$TOKENIZER_PATH" \
        --pretrain \
        --dataset_name           "$DATASET" \
        --split                  test \
        --k_values               "$K_VALUES" \
        --positive_position      random \
        --batch_size             "$BATCH_SIZE" \
        --max_length             "$MAX_LENGTH" \
        --output_dir             "$OUTPUT_DIR"
}

run_eval BeIR/scifact       # Fact checking          (~5K docs)
run_eval BeIR/scidocs       # Citation prediction    (~25K docs)
run_eval BeIR/nfcorpus      # Bio-medical IR         (~3.6K docs)
run_eval BeIR/arguana       # Argument retrieval     (~8.6K docs)
run_eval BeIR/fiqa          # Financial QA           (~57K docs)

echo ""
echo "Done. Results in $OUTPUT_DIR/"

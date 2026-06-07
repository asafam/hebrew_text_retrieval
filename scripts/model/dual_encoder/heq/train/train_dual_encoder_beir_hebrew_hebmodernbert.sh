#!/bin/bash -i
#SBATCH --job-name=train_dualenc_hmb_beir
#SBATCH --output=logs/slurm/train_dualenc_hmb_beir_%j.out
#SBATCH --error=logs/slurm/train_dualenc_hmb_beir_%j.err
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

MODEL="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_v2"
BEIR_ROOT="outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus"
OUTPUT_DIR="outputs/models/dual_encoder/cls_pooling/beir_hebrew/hebmodernbert/ModernBERT-Hebrew-base_v2"

echo "Model: $MODEL"
echo "BeIR corpora root: $BEIR_ROOT"

python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name beir_hebrew \
    --beir_corpora_root "$BEIR_ROOT" \
    --query_model_name "$MODEL" \
    --doc_model_name "$MODEL" \
    --query_field query \
    --document_field document \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --pooling cls \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --force

echo "Done."

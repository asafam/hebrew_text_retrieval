#!/bin/bash -i
#SBATCH --job-name=train_hmb_mean
#SBATCH --output=logs/slurm/train_hmb_mean_%j.out
#SBATCH --error=logs/slurm/train_hmb_mean_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate bert-b200

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT-base-final"
BEIR_ROOT="outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus"
OUTPUT_DIR="outputs/models/dual_encoder/mean_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final"

echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Recipe: lr=2e-5, hard negatives, MEAN pooling, warmup=0.05, 10 epochs, max_length=512"

python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name beir_hebrew_hn \
    --beir_corpora_root "$BEIR_ROOT" \
    --query_model_name "$MODEL" \
    --doc_model_name "$MODEL" \
    --query_field query \
    --document_field document \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --pooling mean \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --force

echo "Done."

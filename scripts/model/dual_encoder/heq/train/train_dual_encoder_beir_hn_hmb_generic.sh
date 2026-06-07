#!/bin/bash -i
#SBATCH --job-name=train_hmb_gen
#SBATCH --output=logs/slurm/train_hmb_gen_%j.out
#SBATCH --error=logs/slurm/train_hmb_gen_%j.err
#SBATCH --partition=A100-4h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Required env vars (pass via sbatch --export):
#   MODEL_PATH  — base HMB checkpoint dir
#   OUTPUT_DIR  — where to save the trained dual encoder
set -e
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BEIR_ROOT="outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus"
POOLING="${POOLING:-cls}"   # override with POOLING=mean

echo "Base model: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Recipe: lr=2e-5, hard negatives, cosine, warmup 0.05, pooling=$POOLING"

python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name beir_hebrew_hn \
    --beir_corpora_root "$BEIR_ROOT" \
    --query_model_name "$MODEL_PATH" \
    --doc_model_name "$MODEL_PATH" \
    --query_field query \
    --document_field document \
    --max_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --pooling "$POOLING" \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --force

echo "Done."

#!/bin/bash -i
#SBATCH --job-name=train_dualenc_neodictabert
#SBATCH --output=logs/slurm/train_dualenc_neodictabert_%j.out
#SBATCH --error=logs/slurm/train_dualenc_neodictabert_%j.err
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

DATASET_NAME="heq_translated"
QUERY_MODEL_NAME="dicta-il/NeoDictaBERT"
DOC_MODEL_NAME="dicta-il/NeoDictaBERT"
QUERY_FIELD="question_hebrew"
DOCUMENT_FIELD="context_hebrew"
MAX_LENGTH=512
NUM_EPOCHS=10
POOLING="cls"
OUTPUT_DIR="outputs/models/dual_encoder/cls_pooling/heq/neodictabert/dicta-il_NeoDictaBERT"

echo "Model: $QUERY_MODEL_NAME"
echo "Dataset: $DATASET_NAME | Max length: $MAX_LENGTH | Epochs: $NUM_EPOCHS | Pooling: $POOLING"
echo "Output dir: $OUTPUT_DIR"

python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name "$DATASET_NAME" \
    --query_model_name "$QUERY_MODEL_NAME" \
    --doc_model_name "$DOC_MODEL_NAME" \
    --query_field "$QUERY_FIELD" \
    --document_field "$DOCUMENT_FIELD" \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --pooling "$POOLING" \
    --output_dir "$OUTPUT_DIR" \
    --force

echo "Done."

#!/bin/bash -i
#SBATCH --job-name=dualenc_hebmodernbertv1
#SBATCH --output=logs/slurm/dualenc_hebmodernbertv1_%j.out
#SBATCH --error=logs/slurm/dualenc_hebmodernbertv1_%j.err
#SBATCH --partition=A100-4h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
DATASET_NAME="heq_translated"
QUERY_MODEL_NAME="/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT_base_mixed_h50e25c25"
DOC_MODEL_NAME="/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT_base_mixed_h50e25c25"
QUERY_FIELD="question_hebrew"
DOCUMENT_FIELD="context_hebrew"
MAX_LENGTH=1024
NUM_EPOCHS=10
OUTPUT_DIR="./outputs/models/dual_encoder/heq/hebmodernbert/HebrewModernBERT_base_mixed_h50e25c25_1024"

# Print the variables
echo "Running the Python script: train_dual_encoder.py"
echo "Dataset name: $DATASET_NAME"
echo "Query model: $QUERY_MODEL_NAME"
echo "Document model: $DOC_MODEL_NAME"
echo "Query field: $QUERY_FIELD"
echo "Document field: $DOCUMENT_FIELD"
echo "Max length: $MAX_LENGTH"
echo "Number of epochs: $NUM_EPOCHS"
echo "Output dir: $OUTPUT_DIR"

# Run the Python script
python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name "$DATASET_NAME" \
    --query_model_name "$QUERY_MODEL_NAME" \
    --doc_model_name "$DOC_MODEL_NAME" \
    --query_field "$QUERY_FIELD" \
    --document_field "$DOCUMENT_FIELD" \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --force

echo "Done."
#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
DATASET_NAME="heq"
QUERY_MODEL_NAME="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
DOC_MODEL_NAME="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
QUERY_FIELD="question_hebrew"
DOCUMENT_FIELD="context_english"
MAX_LENGTH=8192
NUM_EPOCHS=10
OUTPUT_DIR="./outputs/models/dual_encoder/dual_encoder_infonce_heq/hebmodernbert/ckpt_20250622_1325_ep7-ba896339"

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
    --output_dir "$OUTPUT_DIR"

echo "Done."
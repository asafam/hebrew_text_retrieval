#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=1

# Define variables
QUERY_MODEL_NAME="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250603_1331"
DOC_MODEL_NAME="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250603_1331"
NUM_EPOCHS=10
OUTPUT_DIR="./outputs/models/dual_encoder/dual_encoder_infonce_heq/ckpt_c20250603_1331_ep4-ba628000"

# Print the variables
echo "Running the Python script: train_dual_encoder_heq.py"
echo "Query model: $QUERY_MODEL_NAME"
echo "Document model: $DOC_MODEL_NAME"
echo "Number of epochs: $NUM_EPOCHS"
echo "Output dir: $OUTPUT_DIR"

# Run the Python script
python src/model/dual_encoder/train_dual_encoder_heq.py \
    --query_model_name "$QUERY_MODEL_NAME" \
    --doc_model_name "$DOC_MODEL_NAME" \
    --num_train_epochs "$NUM_EPOCHS" \
    --output_dir "$OUTPUT_DIR" \

echo "Done."
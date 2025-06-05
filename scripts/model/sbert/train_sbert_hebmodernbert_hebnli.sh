#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=2

# Define variables
MODEL_PATH="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250603_1331"
OUTPUT_PATH="./outputs/models/sbert/sbert-hebmodernbert-hebnli/ckpt_c20250603_1331_ep4-ba628000"

# Print the variables
echo "Running the Python script: train_sbert_hebmodernbert_hebnli.py"
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"

# Run the Python script
python src/model/sbert/train_sbert_hebmodernbert_hebnli.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \

echo "Done."
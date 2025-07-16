#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
DATASET_NAME="heq"
MODEL_NAME="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
NUM_SAMPLES=0
MAX_DOCUMENT_SEGMENT_TOKENS=2048
OUTPUT_PATH="./outputs/translation"
SPLIT="test"
RANDOM_STATE=42

# Print the variables
echo "Running the Python script: build_translation_candidates.py"
echo "Dataset name: $DATASET_NAME"
echo "Model name: $MODEL_NAME"
echo "Number of samples: $NUM_SAMPLES"
echo "Max document segment tokens: $MAX_DOCUMENT_SEGMENT_TOKENS"
echo "Output path: $OUTPUT_PATH"
echo "Split: $SPLIT"
echo "Random state: $RANDOM_STATE"

# Run the Python script
python src/translation/build_translation_candidates.py \
    --dataset_names "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --max_document_segment_tokens $MAX_DOCUMENT_SEGMENT_TOKENS \
    --model_name "$MODEL_NAME" \
    --output_path "$OUTPUT_PATH" \
    --split "$SPLIT" \
    --random_state $RANDOM_STATE

echo "Done."
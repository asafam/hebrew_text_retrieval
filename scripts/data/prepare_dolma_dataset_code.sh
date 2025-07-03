#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
DATA_PATH="data/dolma"
OUTPUT_PATH="data/dolma/tmp.jsonl"
TOKEN_BUDGET=25_000
SHUFFLE_BUFFFER=1_000_000
TOKENIZER_PATH="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
INCLUDE_SOURCE="starcoder"

echo "Running the Python script: prepare_dolma_dataset.py"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Include source: $INCLUDE_SOURCE"
echo "Token budget: $TOKEN_BUDGET"
echo "Shuffle buffer: $SHUFFLE_BUFFFER"

# Run the Python script
python src/data/datasets/prepare_dolma_dataset.py \
    --output_file "$OUTPUT_PATH" \
    --token_budget "$TOKEN_BUDGET" \
    --data_path "$DATA_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --include_source "$INCLUDE_SOURCE" \
    --shuffle_buffer "$SHUFFLE_BUFFFER"
echo "Done."

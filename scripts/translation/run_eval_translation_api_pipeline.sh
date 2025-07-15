#!/bin/bash -i

# This script runs the translate_queries Python script with the specified arguments.

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo ".env file not found!"
    exit 1
fi

# Define environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=3

# This script runs the translate_queries Python script with the specified arguments.

# Define variables for input and output
SOURCE_FILE_PATHS=(
    "outputs/translation/BeIR/gpt-4o-mini-2024-07-18/long_documents/long_docs.csv"
) 
GOLD_FILE_PATHS=""
PROMPT_FILE_NAME="prompts/translation/openai/translation_evaluation_nogold_v20250304.yaml"
OUTPUT_DIR="outputs/translation/BeIR/gpt-4o-mini-2024-07-18/long_documents"
MODEL_NAME="gpt-4o-2024-08-06"  
LIMIT=0
FORCE=false 
PARALLEL=true
ENGLISH_KEY="Text"
HEBREW_KEY="Translation"

# Print execution parameters
echo "Running translation API with the following parameters:"
echo "Source input file paths: ${SOURCE_FILE_PATHS[@]}"
echo "Prompt file name: $PROMPT_FILE_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "Limit: $LIMIT"
echo "English key: $ENGLISH_KEY"
echo "Hebrew key: $HEBREW_KEY"
echo "Force re-evaluation: ${FORCE:-false}"
echo "Run in parallel: ${PARALLEL:-false}"

# Run the Python script
python src/translation/api/run_evaluate_translations.py \
    --source_file_paths "${SOURCE_FILE_PATHS[@]}" \
    --prompt_file_name "$PROMPT_FILE_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --limit $LIMIT \
    --text_key "$ENGLISH_KEY" \
    --translation_key "$HEBREW_KEY" \
    ${FORCE:+--force}
    ${PARALLE:+--parallel}
    
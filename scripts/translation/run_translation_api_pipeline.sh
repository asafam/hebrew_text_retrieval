#!/bin/bash -i

# This script runs the translate_queries Python script with the specified arguments.
# 8 8G 71%
# 16 11 98%
# 32 19GB 98%
# 64 71G 99%
# 128 56G 100%


# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Load environment variables from .env file
echo "Loading environment variables from .env file"
if [ -f .env ]; then
    source .env
    echo "Environment variables loaded successfully."
else
    echo ".env file not found!"
    exit 1
fi

# Define environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"


# This script runs the translate_queries Python script with the specified arguments.

# Define variables for input and output
SOURCE_FILE_PATHS=(
   "outputs/translation/rajpurkar_squad_v2/train/documents.csv" "outputs/translation/rajpurkar_squad_v2/train/queries.csv"
) 
PROMPT_FILE_NAME="prompts/translation/openai/translation_prompts_zeroshot_v20250220.yaml"
OUTPUT_DIR="outputs/translation/rajpurkar_squad_v2/train/gemini-2.0-flash-lite"
MODEL_NAME="gemini-2.0-flash-lite" # "gpt-4o-mini-2024-07-18"  
LIMIT=0
SLEEP_TIME=5
FORCE=false 
PARALLEL=true
SOURCE_LANGUAGE="English"
TARGET_LANGUAGE="Hebrew"
TEXT_KEY="Text"
TRANSLATION_KEY="Translation"

# Print execution parameters
echo "Running translation API with the following parameters:"
echo "Source input file paths: ${SOURCE_FILE_PATHS[@]}"
echo "Prompt file name: $PROMPT_FILE_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "Limit: $LIMIT"
echo "Sleep time between requests: $SLEEP_TIME seconds"
echo "Source language: $SOURCE_LANGUAGE"
echo "Target language: $TARGET_LANGUAGE"
echo "Text key: $TEXT_KEY"
echo "Translation key: $TRANSLATION_KEY"
echo "Force re-evaluation: ${FORCE:-false}"
echo "Run in parallel: ${PARALLEL:-false}"

# Run the Python script
python src/translation/api/run_translation_pipeline.py \
    --source_file_paths "${SOURCE_FILE_PATHS[@]}" \
    --prompt_file_name "$PROMPT_FILE_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --limit $LIMIT \
    --sleep_time $SLEEP_TIME \
    --source_lang "$SOURCE_LANGUAGE" \
    --target_lang "$TARGET_LANGUAGE" \
    --text_key "$TEXT_KEY" \
    --translation_key "$TRANSLATION_KEY" \
    ${FORCE:+--force} \
    ${PARALLEL:+--parallel}
echo "Done."
    
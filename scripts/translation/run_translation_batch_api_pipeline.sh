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
SOURCE_FILE_PATHS=("outputs/translation/BeIR/BeIR_msmarco/documents.csv" "outputs/translation/BeIR/BeIR_msmarco/queries.csv") 
PROMPT_FILE_NAME="prompts/translation/openai/translation_prompts_few_shot_v20250105.yaml"
MODEL_NAME="o1-mini-2024-09-12"  
LIMIT=256
FORCE=true 
ENGLISH_KEY="English"
HEBREW_KEY="Hebrew"
CONTEXT_KEY="Context"

# Print execution parameters
echo "Running translation API with the following parameters:"
echo "Source input file paths: ${SOURCE_FILE_PATHS[@]}"
echo "Translation output file path: $TRANSLATION_OUTPUT_FILE_PATH"
echo "Prompt file name: $PROMPT_FILE_NAME"
echo "Model name: $MODEL_NAME"
echo "Limit: $LIMIT"
echo "Force re-translation: ${FORCE:-false}"
echo "English key: $ENGLISH_KEY"
echo "Hebrew key: $HEBREW_KEY"
echo "Context key: $CONTEXT_KEY"

# Run the Python script
python src/translation/api/run_translation_batch_pipeline.py \
  --source_file_paths "${SOURCE_FILE_PATHS[@]}" \
  --prompt_file_name "$PROMPT_FILE_NAME" \
  --model_name "$MODEL_NAME" \
  --limit $LIMIT \
  ${FORCE:+--force} \
  --english_key "$ENGLISH_KEY" \
  --hebrew_key "$HEBREW_KEY" \
  --context_key "$CONTEXT_KEY"

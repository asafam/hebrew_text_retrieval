#!/bin/bash -i

# This script runs the translate_queries Python script with the specified arguments.

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# This script runs the translate_queries Python script with the specified arguments.

# Define variables for input and output
DATA_FILE_PATHS=("output/translation/BeIR/BeIR_msmarco_documents.csv" "output/translation/BeIR/BeIR_msmarco_queries.csv") 
PROMPT_FILE_NAME="prompts/translation/dicta_dictalm2_0/translation_prompts_v20250105.yaml"
MODEL_NAME="dicta-il/dictalm2.0-AWQ"  
BATCH_SIZE=32  
MAX_NEW_TOKENS=0  
USE_CACHED_PREFIX=true
FORCE=false 

# Print execution parameters
echo "Running translate_queries with the following parameters:"
echo "Data file paths: ${DATA_FILE_PATHS[@]}"
echo "Translation output file path: $TRANSLATION_OUTPUT_FILE_PATH"
echo "Prompt file name: $PROMPT_FILE_NAME"
echo "Model name: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Device: $DEVICE"
echo "Use cached prefix: ${USE_CACHED_PREFIX:-false}"
echo "Force re-translation: ${FORCE:-false}"


# Run the Python script
python src/translation/run_translation_pipeline.py \
  --data_file_paths "${DATA_FILE_PATHS[@]}" \
  --prompt_file_name "$PROMPT_FILE_NAME" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  ${USE_CACHED_PREFIX:+--use_cached_prefix} \
  ${FORCE:+--force}

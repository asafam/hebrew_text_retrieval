#!/bin/bash -i

# This script runs the translate_queries Python script with the specified arguments.
# 8 8G 71%
# 16 11 98%
# 32 19GB 98%
# 64 30G 99%
# 128 56G 100%


# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=4

# This script runs the translate_queries Python script with the specified arguments.

# Define variables for input and output
SOURCE_FILE_PATHS=("outputs/translation/BeIR/BeIR_nq/documents.csv" "outputs/translation/BeIR/BeIR_nq/queries.csv" "outputs/translation/BeIR/BeIR_nfcorpus/documents.csv" "outputs/translation/BeIR/BeIR_nfcorpus/queries.csv") 
PROMPT_FILE_NAME="prompts/translation/dicta_dictalm2_0/translation_prompts_few_shot_v20250105.yaml"
MODEL_NAME="dicta-il/dictalm2.0-AWQ"  
BATCH_SIZE=128
MAX_NEW_TOKENS=0  
USE_CACHED_PREFIX=true
LIMIT=256
FORCE=true 
ENGLISH_KEY="אנגלית"
HEBREW_KEY="עברית"
CONTEXT_KEY="הקשר"

# Print execution parameters
echo "Running translation model with the following parameters:"
echo "Source input file paths: ${SOURCE_FILE_PATHS[@]}"
echo "Translation output file path: $TRANSLATION_OUTPUT_FILE_PATH"
echo "Prompt file name: $PROMPT_FILE_NAME"
echo "Model name: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Limit: $LIMIT"
echo "Use cached prefix: ${USE_CACHED_PREFIX:-false}"
echo "Force re-translation: ${FORCE:-false}"
echo "English key: $ENGLISH_KEY"
echo "Hebrew key: $HEBREW_KEY"
echo "Context key: $CONTEXT_KEY"

# Run the Python script
python src/translation/model/run_translation_pipeline.py \
  --source_file_paths "${SOURCE_FILE_PATHS[@]}" \
  --prompt_file_name "$PROMPT_FILE_NAME" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --limit $LIMIT \
  ${USE_CACHED_PREFIX:+--use_cached_prefix} \
  ${FORCE:+--force} \
  --english_key "$ENGLISH_KEY" \
  --hebrew_key "$HEBREW_KEY" \
  --context_key "$CONTEXT_KEY"

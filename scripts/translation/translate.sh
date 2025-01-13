#!/bin/bash

# Activate conda environment
source activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# This script runs the translate_queries Python script with the specified arguments.

# Define variables for input and output
DATA_FILE_PATHS=("output/translation/BeIR/BeIR_msmarco_documents.csv" "output/translation/BeIR/BeIR_msmarco_queries.csv") 
PROMPT_FILE_NAME="prompts/translation/dicta_dictalm2_0/translation_prompts_v20250105.yaml"
MODEL_NAME="dicta-il/dictalm2.0-AWQ"  
BATCH_SIZE=32  
MAX_NEW_TOKENS=0  
CACHE_PREFIX=true
FORCE=false 

# Run the Python script
python src/translation/translate.py \
  --data_file_paths "${DATA_FILE_PATHS[@]}" \
  --prompt_file_name "$PROMPT_FILE_NAME" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  ${CACHE_PREFIX:+--cache_prefix} \
  ${FORCE:+--force}

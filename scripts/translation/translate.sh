#!/bin/bash

# This script runs the translate_queries Python script with the specified arguments.

# Define variables for input and output
DATA_FILE_PATHS=("output/translation/BeIR/BeIR_msmarco_documents.csv" "output/translation/BeIR/BeIR_msmarco_queries.csv")  # Modify these paths to point to your input files
TRANSLATION_OUTPUT_FILE_PATH="output/translated_queries.json"  # Modify this to your desired output path
PROMPT_FILE_NAME="prompts/translation/translation_v20250105"  # Modify this to your prompt file
MODEL_NAME="dicta-il/dictalm2.0-AWQ"  # Modify this to the model you are using
BATCH_SIZE=32  # Adjust the batch size as needed
MAX_NEW_TOKENS=0  # Adjust the max tokens as needed, or set to 0 if unused
CACHE_PREFIX=false  # Set to true to use cached results
FORCE=false  # Set to true to force re-translation

# Run the Python script
python translate_queries_main.py \
  --data_file_paths "${DATA_FILE_PATHS[@]}" \
  --translation_output_file_path "$TRANSLATION_OUTPUT_FILE_PATH" \
  --prompt_file_name "$PROMPT_FILE_NAME" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --cache_prefix $CACHE_PREFIX \
  --force $FORCE

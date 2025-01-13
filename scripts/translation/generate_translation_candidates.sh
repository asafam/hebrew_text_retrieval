#!/bin/bash

# Define variables
DATASET_NAMES="BeIR/msmarco"
NUM_SAMPLES=1024
MAX_DOCUMENT_SEGMENT_TOKENS=256
MODEL_NAME="dicta-il/dictalm2.0-AWQ"
OUTPUT_PATH="./output/traslation/BeIR/dicta-il_dictalm2.0-AWQ"
RANDOM_SEED=42

# Run the Python script
python ./src/translation/generate_translation_candidates.py \
  --dataset_names $DATASET_NAMES \
  --num_samples $NUM_SAMPLES \
  --max_document_segment_tokens $MAX_DOCUMENT_SEGMENT_TOKENS \
  --model_name $MODEL_NAME \
  --output_path $OUTPUT_PATH \
  --random_seed $RANDOM_SEED
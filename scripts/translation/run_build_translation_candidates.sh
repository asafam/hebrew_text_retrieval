#!/bin/bash

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
DATASET_NAMES="BeIR/msmarco" # BeIR/fever BeIR/climate-fever BeIR/scifact BeIR/scidocs BeIR/quora BeIR/arguana BeIR/nq BeIR/hotpotqa BeIR/trec-covid BeIR/nfcorpus BeIR/dbpedia-entity"
NUM_SAMPLES=1024
MAX_DOCUMENT_SEGMENT_TOKENS=2048
MODEL_NAME="gpt-4o-mini-2024-07-18"
OUTPUT_PATH="./outputs/translation/BeIR"
RANDOM_SEED=42

# Run the Python script
python ./src/translation/build_translation_candidates.py \
  --dataset_names $DATASET_NAMES \
  --num_samples $NUM_SAMPLES \
  --max_document_segment_tokens $MAX_DOCUMENT_SEGMENT_TOKENS \
  --model_name $MODEL_NAME \
  --output_path $OUTPUT_PATH \
  --random_seed $RANDOM_SEED
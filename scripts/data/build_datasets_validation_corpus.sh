#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
CONFIG_FILE="config/data/train/datasets_h50_e75_c25/datasets_validation_corpus.yaml"
OUTPUT_PATH="data/hebrew_modernbert/mixed_h50_e75_c25"
FORMAT="mds"
SHARD_SIZE_LIMIT=67108864
BUFFER_SIZE=1000000
SPLIT="validation"

echo "Running the Python script: build_datasets.py"
echo "Config file path: $CONFIG_FILE"
echo "Output directory: $OUTPUT_PATH"
echo "Split: $SPLIT"
echo "Format: $FORMAT"
echo "Shard size limit: $SHARD_SIZE_LIMIT"
echo "Buffer size: $BUFFER_SIZE"

# Run the Python script
python src/data/datasets/build_datasets.py \
    --config_file "$CONFIG_FILE" \
    --output_path "$OUTPUT_PATH" \
    --split "$SPLIT" \
    --format $FORMAT \
    --shard_size_limit $SHARD_SIZE_LIMIT \
    --buffer_size $BUFFER_SIZE
echo "Dataset validation corpus built successfully."
#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
CONFIG_FILE="config/data/datasets_tokenizer_corpus.yaml"
OUTPUT_PATH="data/hebrew_modernbert/v20250421/tokenizer_corpus_1M.txt"
FORMAT="txt"
SHARD_SIZE_LIMIT=67108864
BUFFER_SIZE=1000000

echo "Running the Python script: jsonl_to_mds.py"
echo "Config file path: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Format: $FORMAT"
echo "Shard size limit: $SHARD_SIZE_LIMIT"
echo "Buffer size: $BUFFER_SIZE"

# Run the Python script
python src/data/preprocessing/build_datasets.py \
    --config_file "$CONFIG_FILE" \
    --output_path "$OUTPUT_PATH" \
    --format $FORMAT \
    --shard_size_limit $SHARD_SIZE_LIMIT \
    --buffer_size $BUFFER_SIZE \

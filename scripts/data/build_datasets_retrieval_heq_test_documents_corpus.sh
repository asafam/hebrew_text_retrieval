#!/bin/bash -i

# Activate the bert24 conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
CONFIG_FILE="config/data/heq/datasets_retrieval_test_documents_corpus.yaml"
OUTPUT_PATH="data/retrieval/heq/test/documents.jsonl"
FORMAT="jsonl"
SHARD_SIZE_LIMIT=67108864
BUFFER_SIZE=1000000
SPLIT="test"

echo "Running the Python script: build_datasets.py"
echo "Config file path: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
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

echo "Done."

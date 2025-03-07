#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
JSON_FILES_PATH="data/mafat/hebrew/sources"
SIZE_LIMIT=67108864
OUTPUT_DIR="data/mafat/hebrew/data"

echo "Running the Python script: jsonl_to_mds.py"
echo "JSON files path: $JSON_FILES_PATH"
echo "Size limit: $SIZE_LIMIT"
echo "Output directory: $OUTPUT_DIR"

# Run the Python script
python src/data/preprocessing/jsonl_to_mds.py \
    --jsonl_files "$JSON_FILES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --size_limit $SIZE_LIMIT

#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
JSON_FILES_PATH="data/mafat/hebrew/sources/*.jsonl"
LIMIT=500000
OUTPUT_FILE="data/mafat/hebrew/tokenizer_corpus_500K.txt"
RANDOM_STATE=42
FORCE=true

# Print the variables
echo "Running the Python script: build_tokenizer_corpus.py"
echo "JSON files path: $JSON_FILES_PATH"
echo "Limit: $LIMIT"
echo "Output file: $OUTPUT_FILE"
echo "Random state: $RANDOM_STATE"
echo "Force: $FORCE"


# Run the Python script
python src/data/preprocessing/build_tokenizer_corpus.py \
    --jsonl_files_path "$JSON_FILES_PATH" \
    --limit $LIMIT \
    --output_file "$OUTPUT_FILE" \
    --random_state $RANDOM_STATE \
    ${FORCE:+--force} \

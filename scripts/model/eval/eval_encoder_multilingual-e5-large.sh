#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
MODEL_PATH="intfloat/multilingual-e5-large"
TOKENIZER_PATH="intfloat/multilingual-e5-large"
DOCUMENTS_PATH="data/retrieval/documents/test_v2.jsonl"
OUTPUT_FILE="outputs/eval/encoder/heq_test_docs/intfloat_multilingual-e5-large/results.json"
BATCH_SIZE=32

# Print the variables
echo "Running the Python script: eval_encoding.py"
echo "Model path: $MODEL_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Documents path: $DOCUMENTS_PATH"
echo "Output file: $OUTPUT_FILE"
echo "Batch size: $BATCH_SIZE"

# Run the Python script
python src/model/eval/eval_encoding.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE"

echo "Done."
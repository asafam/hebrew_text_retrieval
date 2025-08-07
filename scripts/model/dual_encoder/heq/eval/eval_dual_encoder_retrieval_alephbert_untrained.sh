#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
MODEL_PATH="onlplab/alephbert-base"
TOKENIZER_PATH="onlplab/alephbert-base"
QUERIES_PATH="data/retrieval/heq/test/queries.jsonl"
DOCUMENTS_PATH="data/retrieval/heq/test/documents.jsonl"
BATCH_SIZE=1024
MAX_LENGTH=512
EMBEDDING_FILES_PATH="outputs/eval/dual_encoder/heq/onlplab_alephbert-base/model_untrained"
OUTPUT_FILE="outputs/eval/dual_encoder/heq/onlplab_alephbert-base/model_untrained/results.txt"

# Print the variables
echo "Running the Python script: eval_retrieval.py"
echo "Model path: $MODEL_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Queries path: $QUERIES_PATH"
echo "Documents path: $DOCUMENTS_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Embeddings files path: $EMBEDDING_FILES_PATH"
echo "Output file: $OUTPUT_FILE"

# Run the Python script
python src/model/eval/eval_retrieval.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --queries_path "$QUERIES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --embeddings_files_path "$EMBEDDING_FILES_PATH" \
    --output_file "$OUTPUT_FILE"

echo "Done."
#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
MODEL_PATH="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
TOKENIZER_PATH="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
QUERIES_PATH="data/retrieval/heq/test/queries.jsonl"
DOCUMENTS_PATH="data/retrieval/heq/test/documents_long_context_2048_random.jsonl"
BATCH_SIZE=64
MAX_LENGTH=2048
EMBEDDING_FILES_PATH="outputs/eval/dual_encoder/heq_long_contexts/hebmodernbert/model_untrained/doc_embeddings_2048.pt"
OUTPUT_FILE="outputs/eval/dual_encoder/heq_long_contexts/hebmodernbert/model_untrained/results_2048.txt"
DOCUMENT_TEXT_FIELD="long_context"

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
echo "Document text field: $DOCUMENT_TEXT_FIELD"

# Run the Python script
python src/model/eval/eval_retrieval.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --queries_path "$QUERIES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --embeddings_files_path "$EMBEDDING_FILES_PATH" \
    --output_file "$OUTPUT_FILE" \
    --document_text_field "$DOCUMENT_TEXT_FIELD"

echo "Done."
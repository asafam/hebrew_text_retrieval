#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
MODEL_PATH="/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT_base_mixed_h50e25c25_1024_0.2"
TOKENIZER_PATH="/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT_base_mixed_h50e25c25_1024_0.2"
QUERIES_PATH="data/retrieval/heq/test/queries_hebrew.jsonl"
DOCUMENTS_PATH="data/retrieval/heq/test/documents_hebrew_long_context_1024_random.jsonl"
BATCH_SIZE=128
MAX_LENGTH=8192
EMBEDDING_FILES_PATH="outputs/eval/dual_encoder/heq_long_contexts/hebmodernbert/HebrewModernBERT_base_mixed_h50e25c25_1024_0.2/model_untrained/doc_embeddings_1024.pt"
OUTPUT_FILE="outputs/eval/dual_encoder/heq_long_contexts/hebmodernbert/HebrewModernBERT_base_mixed_h50e25c25_1024_0.2/model_untrained/results_1024.txt"
QUERY_TEXT_FIELD="question_hebrew"
QUERY_CONTEXT_FIELD="context_hebrew"
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
echo "Query text field: $QUERY_TEXT_FIELD"
echo "Query context field: $QUERY_CONTEXT_FIELD"
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
    --query_text_field "$QUERY_TEXT_FIELD" \
    --query_context_field "$QUERY_CONTEXT_FIELD" \
    --document_text_field "$DOCUMENT_TEXT_FIELD"

echo "Done."
#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
MODEL_PATH="outputs/models/dual_encoder/dual_encoder_infonce_squad_v2/intfloat_multilingual-e5-large/model"
TOKENIZER_PATH="intfloat/multilingual-e5-large"
QUERIES_PATH="data/retrieval/squad_v2/validation/queries.jsonl"
DOCUMENTS_PATH="data/retrieval/squad_v2/validation/documents.jsonl"
BATCH_SIZE=1024
MAX_LENGTH=512
DOCUMENTS_EMBEDDING_FILE="outputs/eval/dual_encoder/dual_encoder_infonce_squad_v2_q_en_d_he/intfloat_multilingual-e5-large/model/doc_embeddings.pt"
OUTPUT_FILE="outputs/eval/dual_encoder/dual_encoder_infonce_squad_v2_q_en_d_he/intfloat_multilingual-e5-large/model/results.txt"
QUERY_CONTEXT_FIELD="context_hebrew"
MAIN_SOURCE="rajpurkar_squad_v2"

# Print the variables
echo "Running the Python script: eval_retrieval.py"
echo "Model path: $MODEL_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Queries path: $QUERIES_PATH"
echo "Documents path: $DOCUMENTS_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Documents embeddings file: $DOCUMENTS_EMBEDDING_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Query context field: $QUERY_CONTEXT_FIELD"
echo "Main source: $MAIN_SOURCE"


# Run the Python script
python src/model/eval/eval_retrieval.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --queries_path "$QUERIES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --documents_embeddings_file "$DOCUMENTS_EMBEDDING_FILE" \
    --output_file "$OUTPUT_FILE" \
    --query_context_field "$QUERY_CONTEXT_FIELD" \
    --main_source "$MAIN_SOURCE"

echo "Done."
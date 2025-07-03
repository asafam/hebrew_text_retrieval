#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: htr"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate htr

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=2

# Define variables
MODEL_PATH="outputs/models/dual_encoder/dual_encoder_infonce_heq/hebmodernbert/ckpt_20250622_1325_ep7-ba896339/model"
TOKENIZER_PATH="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0"
QUERIES_PATH="data/retrieval/queries/test.jsonl"
DOCUMENTS_PATH="data/retrieval/documents/test.jsonl"
BATCH_SIZE=64
MAX_LENGTH=8192
DOCUMENTS_EMBEDDING_FILE="outputs/eval/dual_encoder/dual_encoder_infonce_heq/ckpt_20250622_1325_ep7-ba896339/model/doc_embeddings.pt"
OUTPUT_FILE="outputs/eval/dual_encoder/dual_encoder_infonce_heq/ckpt_20250622_1325_ep7-ba896339/model/results.txt"

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

# Run the Python script
python src/model/eval/eval_retrieval.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --queries_path "$QUERIES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --documents_embeddings_file "$DOCUMENTS_EMBEDDING_FILE" \
    --output_file "$OUTPUT_FILE"

echo "Done."
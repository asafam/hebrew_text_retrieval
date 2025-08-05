#!/bin/bash -i

# Activate the bert24 conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define parameters
DOCUMENTS_FILE_PATH="data/retrieval/heq/test/documents.jsonl"
TOKENIZER="intfloat/multilingual-e5-base"
CONTEXT_WINDOW=1024
GT_LOCATION="random"
DOCUMENT_FIELD="text"
OUTPUT_FIELD="long_context"
DISTRACTOR_SOURCE_FOLDER="data/mafat/hebrew/sources/"
# Compose output file name dynamically
OUTPUT_DIR="data/retrieval/heq/test"
OUTPUT_FILE="documents_${OUTPUT_FIELD}_${CONTEXT_WINDOW}_${GT_LOCATION}.jsonl"
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"

# Show parameters
echo "Documents file path:    $DOCUMENTS_FILE_PATH"
echo "Output path:            $OUTPUT_PATH"
echo "Tokenizer:              $TOKENIZER"
echo "Context window:         $CONTEXT_WINDOW"
echo "GT location:            $GT_LOCATION"
echo "Document field:         $DOCUMENT_FIELD"
echo "Output field:           $OUTPUT_FIELD"
echo "Distractor source dir:  $DISTRACTOR_SOURCE_FOLDER"

# Run the Python script
python src/data/datasets/build_long_context_dataset.py \
    --documents_file_path "$DOCUMENTS_FILE_PATH" \
    --output "$OUTPUT_PATH" \
    --tokenizer "$TOKENIZER" \
    --context-window $CONTEXT_WINDOW \
    --gt_location $GT_LOCATION \
    --document_field "$DOCUMENT_FIELD" \
    --output_field "$OUTPUT_FIELD" \
    --distractor_source_folder "$DISTRACTOR_SOURCE_FOLDER"

echo "Done."

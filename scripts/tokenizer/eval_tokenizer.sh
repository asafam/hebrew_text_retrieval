#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
MODEL_FILE="outputs/tokenizer/HebrewModernBERT_heb_10M_128K.model"
EVAL_FILE="data/mafat/hebrew/tokenizer_validation_1M.txt"
OUTPUT_FILE="outputs/tokenizer/evaluations/tokenizer_eval.csv"

# Print the variables
echo "Running the Python script: eval_tokenizer.py"
echo "Model file: $MODEL_FILE"
echo "Eval file: $EVAL_FILE"
echo "Output file: $OUTPUT_FILE"

# Run the Python script
python src/tokenizer/eval_tokenizer.py \
    --model_file "$MODEL_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_file "$OUTPUT_FILE" \

echo "Done."
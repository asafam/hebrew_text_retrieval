#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Define variables
CORPUS="data/mafat/hebrew/tokenizer_corpus_100K.txt"
VOCAB_SIZE=96000
OUTPUT_DIR="outputs/tokenizer/HebrewModernBERT_heb_100K_96K"

# Print the variables
echo "Running the Python script: build_tokenizer_corpus.py"
echo "Corpus: $CORPUS"
echo "Vocab size: $VOCAB_SIZE"
echo "Output dir: $OUTPUT_DIR"

# Run the Python script
python src/tokenizer/train_tokenizer.py \
    --corpus "$CORPUS" \
    --vocab_size $VOCAB_SIZE \
    --output_dir "$OUTPUT_DIR" \

echo "Done."
echo "Corpus: $CORPUS"
echo "Vocab size: $VOCAB_SIZE"
echo "Output dir: $OUTPUT_DIR"
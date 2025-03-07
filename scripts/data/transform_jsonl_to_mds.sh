#!/bin/bash -i

# Activate the htr conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert24

# Add src folder to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Run the Python script
python src/data/preprocessing/jsonl_to_mds.py \
    --jsonl_files "data/mafat/hebrew/sources" \
    --output_dir "data/mafat/hebrew/data" \
    --size_limit 67108864

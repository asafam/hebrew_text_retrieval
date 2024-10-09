#!/bin/bash

# Ensure that conda is properly initialized
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate biu

# Set the PYTHONPATH to include the src directory
export PYTHONPATH="./src:$PYTHONPATH"

python src/train_model.py \
    --model_name intfloat/multilingual-e5-large \
    --dataset_name synthesized_query_document \
    --epochs 10 \
    --batch_size 32 \
    --cuda_visible_devices 5

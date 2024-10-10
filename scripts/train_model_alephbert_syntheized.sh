#!/bin/bash

# Ensure that conda is properly initialized
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate biu

# Set the PYTHONPATH to include the src directory
export PYTHONPATH="./src:$PYTHONPATH"

python src/train_model.py \
    --model_name onlplab/alephbert-base \
    --dataset_name synthesized_query_document \
    --epochs 10 \
    --batch_size 32 \
    --source_checkpoint_dir checkpoints/intfloat_multilingual_e5_large/checkpoints_synthesized_query_document \
    --cuda_visible_devices 0

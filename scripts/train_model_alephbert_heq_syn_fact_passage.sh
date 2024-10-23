#!/bin/bash

# Ensure that conda is properly initialized
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate biu

# Set the PYTHONPATH to include the src directory
export PYTHONPATH="./src:$PYTHONPATH"

python src/train_model.py \
    --model_name onlplab/alephbert-base \
    --task_name fact_passage \
    --dataset_name heq_syn_fact_passage \
    --epochs 5 \
    --batch_size 32 \
    --source_checkpoint_dir checkpoints/onlplab_alephbert_base/checkpoints_fact_passage \
    --cuda_visible_devices 7

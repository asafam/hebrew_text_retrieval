#!/bin/bash

# Ensure that conda is properly initialized
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate biu

# Set the PYTHONPATH to include the src directory
export PYTHONPATH="./src:$PYTHONPATH"

python src/train_model.py \
    --model_name dicta-il/dictabert-large \
    --task_name title_passage \
    --dataset_name wiki40b \
    --epochs 5 \
    --batch_size 32 \
    --source_checkpoint_dir checkpoints/dicta_il_dictabert_large/checkpoints_title_passage \
    --cuda_visible_devices 5

#!/bin/bash
#SBATCH --job-name=lang_compare
#SBATCH --output=logs/slurm/lang_compare_%j.out
#SBATCH --error=logs/slurm/lang_compare_%j.err
#SBATCH --partition=H200-4h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
set -e
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
python scripts/analysis/per_query_lang_compare.py \
    --model intfloat/multilingual-e5-base --model_tag mE5-base
echo "DONE"

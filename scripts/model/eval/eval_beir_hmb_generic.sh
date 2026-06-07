#!/bin/bash -i
#SBATCH --job-name=eval_hmb_gen
#SBATCH --output=logs/slurm/eval_hmb_gen_%j.out
#SBATCH --error=logs/slurm/eval_hmb_gen_%j.err
#SBATCH --partition=L4-4h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Required env vars (pass via sbatch --export):
#   MODEL_PATH   — trained dual encoder model dir (the .../model subdir)
#   MODEL_LABEL  — short label for results
set -e
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

BATCH_SIZE=256
MAX_LENGTH=512
RESULTS_DIR="outputs/eval/beir_zeroshot"

echo "Model: $MODEL_PATH (label: $MODEL_LABEL)"

while IFS= read -r CORPUS_DIR; do
    echo ""
    echo "=== Local BeIR: $CORPUS_DIR ==="
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_label "$MODEL_LABEL" \
        --corpus_dir "$CORPUS_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH"
done < <(find outputs/translation/runs -name "corpus.jsonl" -path "*/beir/corpus.jsonl" 2>/dev/null | xargs -I{} dirname {} 2>/dev/null | sort)

echo ""
python src/model/eval/collect_beir_results.py --results_dir "$RESULTS_DIR"
echo "Done."

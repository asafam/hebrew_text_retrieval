#!/bin/bash -i
#SBATCH --job-name=eval_hmb_mean
#SBATCH --output=logs/slurm/eval_hmb_mean_%j.out
#SBATCH --error=logs/slurm/eval_hmb_mean_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate bert-b200

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

MODEL="outputs/models/dual_encoder/mean_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model"
LABEL="hebrewmodernbert-base-final-mean"
BATCH_SIZE=256
MAX_LENGTH=512
RESULTS_DIR="outputs/eval/beir_zeroshot"

echo "Model: $MODEL (mean pooling — label: $LABEL)"

while IFS= read -r CORPUS_DIR; do
    echo ""
    echo "=== Local BeIR: $CORPUS_DIR ==="
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path "$MODEL" \
        --model_label "$LABEL" \
        --corpus_dir "$CORPUS_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --pooling mean
done < <(find outputs/translation/runs -name "corpus.jsonl" -path "*/beir/corpus.jsonl" 2>/dev/null | xargs -I{} dirname {} 2>/dev/null | sort)

echo ""
echo "=== Results summary ==="
python src/model/eval/collect_beir_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_csv "${RESULTS_DIR}/summary_hmb_final_mean.csv"

echo "Done."

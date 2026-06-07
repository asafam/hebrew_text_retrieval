#!/bin/bash -i
#SBATCH --job-name=eval_beir_ndb_dualenc
#SBATCH --output=logs/slurm/eval_beir_ndb_dualenc_%j.out
#SBATCH --error=logs/slurm/eval_beir_ndb_dualenc_%j.err
#SBATCH --partition=L4-4h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

MODEL="outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model"
LABEL="neodictabert-dualenc-beir"
BATCH_SIZE=256
MAX_LENGTH=512
RESULTS_DIR="outputs/eval/beir_zeroshot"

echo "Model: $MODEL (label: $LABEL)"

while IFS= read -r CORPUS_DIR; do
    echo ""
    echo "=== Local BeIR: $CORPUS_DIR ==="
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path "$MODEL" \
        --model_label "$LABEL" \
        --corpus_dir "$CORPUS_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH"
done < <(find outputs/translation/runs -name "corpus.jsonl" -path "*/beir/corpus.jsonl" 2>/dev/null | xargs -I{} dirname {} 2>/dev/null | sort)

echo ""
echo "=== Results summary ==="
python src/model/eval/collect_beir_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_csv "${RESULTS_DIR}/summary_neodictabert_dualenc.csv"

echo "Done."

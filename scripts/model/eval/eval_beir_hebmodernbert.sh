#!/bin/bash -i
#SBATCH --job-name=eval_beir_hebmodernbert
#SBATCH --output=logs/slurm/eval_beir_hebmodernbert_%j.out
#SBATCH --error=logs/slurm/eval_beir_hebmodernbert_%j.err
#SBATCH --partition=L4-12h
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e
mkdir -p logs/slurm

# MODEL_PATH must be set to the HF-converted HebrewModernBERT checkpoint directory.
# Example:
#   MODEL_PATH=/path/to/HebrewModernBERT/hf/checkpoint sbatch scripts/model/eval/eval_beir_hebmodernbert.sh
if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH environment variable is not set."
    echo "Usage: MODEL_PATH=/path/to/checkpoint sbatch $0"
    exit 1
fi


source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate htr

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

BATCH_SIZE=256
MAX_LENGTH=512
RESULTS_DIR="outputs/eval/beir_zeroshot"

echo "Model: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE | Max length: $MAX_LENGTH"

# --- Local translated BeIR corpora (output of build_hf_dataset.py) ---
while IFS= read -r CORPUS_DIR; do
    echo ""
    echo "=== Local BeIR: $CORPUS_DIR ==="
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path "$MODEL_PATH" \
        --corpus_dir "$CORPUS_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH"
done < <(find outputs/translation/runs -name "corpus.jsonl" -path "*/beir/corpus.jsonl" 2>/dev/null | xargs -I{} dirname {} 2>/dev/null | sort)

# --- Aggregate results ---
echo ""
echo "=== Results summary ==="
python src/model/eval/collect_beir_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_csv "${RESULTS_DIR}/summary_hebmodernbert.csv"

echo "Done."

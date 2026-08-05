#!/bin/bash
#SBATCH --job-name=hmb_longctx_sft
#SBATCH --output=logs/slurm/hmb_longctx_sft_%j.out
#SBATCH --error=logs/slurm/hmb_longctx_sft_%j.err
#SBATCH --partition=H200-12h
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Continued SFT of HMB on WELDED long documents.
#
# Why this and not just --max_length 4096: the standard training documents are median ~264
# tokens, so no example approaches even a 512-token limit and raising the ceiling changes
# nothing. The measured consequence is that HMB and NeoDictaBERT both collapse to ~0.000
# NDCG@10 when asked to encode a padded document natively, while chunk-and-max-pool over the
# same content holds up. This run gives the model documents that look like what it will be
# asked to encode: the answer as a small span inside mostly-irrelevant filler.
#
# Starting point is the ALREADY FINE-TUNED dual-encoder checkpoint, not the base model, so this
# is length adaptation on top of retrieval ability rather than retraining from scratch. Far
# fewer epochs are needed as a result.
#
# MAX_LENGTH=2048 by default, not 8192. The welded set's mean document is ~10,870 chars
# (~2,700 HMB tokens), so 2048 covers the bulk of it. Going to 8192 multiplies cost for the
# ~10% of examples at the largest rung and risks blowing the 4h wall; raise it only after the
# timing probe below shows there is room.

# NOTE: no `set -u` -- the conda cuda-nvcc activation hook references NVCC_PREPEND_FLAGS
# unguarded, so `-u` kills the job during `conda activate` before training starts.
set -o pipefail
mkdir -p logs/slurm

source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate ${TRAIN_ENV:-bert-b200}

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Continue from the FINE-TUNED dual encoder. An earlier version of this script carried this
# same comment while INIT pointed at the raw base model, so the run silently restarted from
# scratch: it compared "2 epochs on 40k welded examples" against "10 epochs of standard BeIR
# fine-tuning" and regressed 6x on the c3700 anchor. The comment was right; the path was not.
INIT_CKPT="${INIT_CKPT:-outputs/models/dual_encoder/cls_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model}"
# Base model, used only for tokenizers and for the config's encoder paths.
INIT="${INIT:-/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT-base-final}"
WELDED_ROOT="${WELDED_ROOT:-data/retrieval/beir_longctx_train/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/dual_encoder/cls_pooling/beir_longctx/hebmodernbert/HMB-base-final-longctx}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-16}"
LR="${LR:-1e-5}"          # below the original 2e-5: adaptation, not fresh training
MAX_STEPS="${MAX_STEPS:-}"
MAX_PER_DATASET="${MAX_PER_DATASET:-25000}"

echo "Init ckpt:   ${INIT_CKPT:-<none, fresh from base>}"
echo "Base:        $INIT"
echo "Welded data: $WELDED_ROOT"
echo "Output:      $OUTPUT_DIR"
echo "max_length=$MAX_LENGTH epochs=$EPOCHS batch=$BATCH accum=$ACCUM lr=$LR"
echo "effective batch = $((BATCH * ACCUM))"
[ -n "$MAX_STEPS" ] && echo "MAX_STEPS=$MAX_STEPS (timing probe)"
echo

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name beir_hebrew_longctx \
    --welded_root "$WELDED_ROOT" \
    --query_model_name "$INIT" \
    --doc_model_name "$INIT" \
    --query_field query \
    --document_field document \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH" \
    --gradient_accumulation_steps "$ACCUM" \
    --learning_rate "$LR" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --pooling cls \
    ${INIT_CKPT:+--init_from_checkpoint "$INIT_CKPT"} \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
    ${MAX_PER_DATASET:+--max_per_dataset "$MAX_PER_DATASET"} \
    --force
status=$?

# Without this the script exits 0 from the final echo even when training crashed, and SLURM
# reports COMPLETED -- which is exactly how a TypeError at startup looked like a finished run.
if [ $status -ne 0 ]; then
    echo "TRAINING FAILED (exit $status)" >&2
    exit $status
fi
echo "Done."

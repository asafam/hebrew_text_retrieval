#!/bin/bash
#SBATCH --job-name=lr_sweep_lc
#SBATCH --output=logs/slurm/lr_sweep_lc_%j.out
#SBATCH --error=logs/slurm/lr_sweep_lc_%j.err
#SBATCH --partition=H200-12h
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Learning-rate sweep for long-context continued SFT.
#
# The previous length-adaptation run had TWO uncontrolled variables: it initialised from the
# raw base model instead of the fine-tuned checkpoint, and its learning rate was picked by
# heuristic ("half the recipe's 2e-5") rather than measured. It regressed 6x on the c3700
# anchor. Fixing only the init would leave the LR still guessed, so a second failure would be
# unattributable.
#
# Candidates, all continued from the fine-tuned checkpoint:
#   5e-6  the project record says this UNDER-TRAINS HMB -- low bound
#   1e-5  the value previously guessed
#   2e-5  the value the existing recipe established for HMB
#
# Short runs by design: 150 steps at effective batch 32 sees ~4,800 examples, enough to expose
# catastrophic damage, which is the failure mode that actually occurred. It does NOT prove a
# full 2-epoch run will hold -- a config that survives 150 steps can still drift over 4,700.
#
# Decision rule, fixed before seeing results:
#   1. reject any LR whose c3700 chunked anchor falls below the pre-SFT 0.209 (damage)
#   2. among survivors take the highest anchor, validation loss as tiebreak
#   3. if all three regress, the problem is not the LR -- stop rather than keep tuning

set -eo pipefail   # -e so a crash cannot be reported as COMPLETED; no -u (conda hook needs it unset)
mkdir -p logs/slurm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${TRAIN_ENV:-bert-b200}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

INIT_CKPT="${INIT_CKPT:-outputs/models/dual_encoder/cls_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model}"
BASE="${BASE:-/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT-base-final}"
WELDED_ROOT="${WELDED_ROOT:-data/retrieval/beir_longctx_train/v1}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/models/dual_encoder/cls_pooling/beir_longctx/lr_sweep}"
BENCH="${BENCH:-data/retrieval/beir_longctx/v1/BeIR_scifact}"
EVAL_OUT="${EVAL_OUT:-outputs/eval/longctx_lr_sweep}"

LRS="${LRS:-5e-6 1e-5 2e-5}"
STEPS="${STEPS:-150}"
PER_DS="${PER_DS:-5000}"

echo "init:   $INIT_CKPT"
echo "LRs:    $LRS   steps=$STEPS  per_dataset=$PER_DS"
echo "anchor: c3700 chunked, pre-SFT reference 0.209"
echo

for lr in $LRS; do
    out="$SWEEP_ROOT/lr$lr"
    echo "########## TRAIN lr=$lr ##########"
    python src/model/dual_encoder/train_dual_encoder.py \
        --dataset_name beir_hebrew_longctx \
        --welded_root "$WELDED_ROOT" \
        --max_per_dataset "$PER_DS" \
        --init_from_checkpoint "$INIT_CKPT" \
        --query_model_name "$BASE" \
        --doc_model_name "$BASE" \
        --query_field query \
        --document_field document \
        --max_length 8192 \
        --max_steps "$STEPS" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate "$lr" \
        --warmup_ratio 0.05 \
        --lr_scheduler_type cosine \
        --pooling cls \
        --bf16 \
        --output_dir "$out" \
        --force
done

# Evaluate every survivor on the anchor. Chunked, because it is the most sensitive detector of
# damage: the broken run showed 0.209 -> 0.035 here.
conda activate htr
for lr in $LRS; do
    ck="$SWEEP_ROOT/lr$lr/model"
    [ -d "$ck" ] || ck="$SWEEP_ROOT/lr$lr"
    echo "########## ANCHOR lr=$lr ##########"
    python src/model/eval/eval_longctx.py \
        --benchmark_dir "$BENCH" \
        --condition random --rung 3700 --position middle \
        --model_name_or_path "$ck" --model_label "lr$lr" \
        --pooling cls --strategy chunked_para --window 512 \
        --batch_size 64 \
        --output_file "$EVAL_OUT/lr$lr.json" \
        --force_reencode || echo "!! eval failed for lr=$lr"
done

echo
echo "================ LR SWEEP VERDICT ================"
python - <<'PY'
import glob, json, os
PRE_SFT = 0.209   # fine-tuned checkpoint, c3700 chunked, before any length adaptation
rows = []
for f in sorted(glob.glob(os.path.join(os.environ.get("EVAL_OUT",
                "outputs/eval/longctx_lr_sweep"), "*.json"))):
    d = json.load(open(f))
    rows.append((d["arm"], d["metrics"]["ndcg_at_10"]))

print(f"{'lr':10s} {'c3700 anchor':>13s} {'vs pre-SFT':>11s}  verdict")
survivors = []
for arm, n in rows:
    delta = n - PRE_SFT
    ok = n >= PRE_SFT
    if ok:
        survivors.append((n, arm))
    print(f"{arm:10s} {n:13.3f} {delta:+11.3f}  {'keeps ability' if ok else 'DAMAGED - reject'}")
print(f"{'(pre-SFT)':10s} {PRE_SFT:13.3f}")
print()
if survivors:
    best = max(survivors)
    print(f"CHOSEN: {best[1]} (anchor {best[0]:.3f})")
    print("Note: 150 steps rules out catastrophic damage, not slow drift over a full run.")
else:
    print("ALL CANDIDATES REGRESS -> the learning rate is not the problem.")
    print("Do not keep tuning; the welded-data adaptation itself needs rethinking.")
PY

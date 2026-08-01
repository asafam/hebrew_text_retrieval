#!/bin/bash
#SBATCH --job-name=lc_smoke
#SBATCH --output=logs/slurm/lc_smoke_%j.out
#SBATCH --error=logs/slurm/lc_smoke_%j.err
#SBATCH --partition=L4-12h
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Smoke test for the long-context driver, on the cheapest corpus (scifact, 5,183 docs).
#
# This is a correctness check, not a result. Its value is the comparison to c0: at the
# smallest rung the gold passage is byte-identical to the unpadded corpus and only filler was
# added around it, so a model that still retrieves well should land in the same neighbourhood
# as its c0 score. A wildly different number means the driver is broken -- wrong corpus, wrong
# qrels alignment, or windows not mapping back to documents -- not that long context is hard.
#
# c0 reference (scifact NDCG@10): mE5-large .581 | mE5-base .549 | NDB .501 | HMB-final .309

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

BENCH="${BENCH:-data/retrieval/beir_longctx/v1/BeIR_scifact}"
OUT="${OUT:-outputs/eval/longctx_smoke}"
RUNG="${RUNG:-3700}"
COND="${COND:-random}"
POS="${POS:-middle}"

HMB="outputs/models/dual_encoder/cls_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model"
NDB="outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model"

# label|model|pooling|strategy|window|batch
ARMS=(
  "HMB-native|${HMB}|cls|native|8192|16"
  "mE5-base-trunc|intfloat/multilingual-e5-base|mean|truncate|512|128"
  "mE5-base-chunkpara|intfloat/multilingual-e5-base|mean|chunked_para|512|128"
  "NDB-native|${NDB}|cls|native|4096|16"
)

mkdir -p "$OUT" logs/slurm
for arm in "${ARMS[@]}"; do
    IFS='|' read -r label model pooling strategy window batch <<< "$arm"
    echo
    echo "############ $label  (c${RUNG}/${COND}/${POS}) ############"
    python src/model/eval/eval_longctx.py \
        --benchmark_dir "$BENCH" \
        --condition "$COND" --rung "$RUNG" --position "$POS" \
        --model_name_or_path "$model" --model_label "$label" \
        --pooling "$pooling" --strategy "$strategy" --window "$window" \
        --batch_size "$batch" \
        --output_file "$OUT/c${RUNG}_${COND}_${POS}/${label}.json" \
        --force_reencode \
      || echo "!! FAILED: $label"
done

echo
echo "=== smoke summary (compare to c0 scifact) ==="
python - <<'PY'
import glob, json, os
C0 = {"HMB-native": 0.309, "mE5-base-trunc": 0.549,
      "mE5-base-chunkpara": 0.549, "NDB-native": 0.501}
print(f"{'arm':20s} {'NDCG@10':>8s} {'c0':>7s} {'delta':>7s} {'win/doc':>8s} {'gold vis':>9s}")
for f in sorted(glob.glob(os.path.join(os.environ.get("OUT", "outputs/eval/longctx_smoke"),
                                       "**", "*.json"), recursive=True)):
    d = json.load(open(f))
    arm = d["arm"]; n = d["metrics"]["ndcg_at_10"]; c0 = C0.get(arm)
    vis = d.get("visibility", {}).get("gold_visible_frac")
    vis_s = f"{vis:.1%}" if vis is not None else "n/a"
    print(f"{arm:20s} {n:8.3f} {c0 if c0 else 0:7.3f} {(n - c0) if c0 else 0:+7.3f} "
          f"{d['windows_per_doc']:8.2f} {vis_s:>9s}")
PY

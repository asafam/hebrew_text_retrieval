#!/bin/bash
#SBATCH --job-name=lc_probe
#SBATCH --output=logs/slurm/lc_probe_%j.out
#SBATCH --error=logs/slurm/lc_probe_%j.err
#SBATCH --partition=L4-12h
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Is a long window itself harmful to these checkpoints, independent of long documents?
#
# In the smoke test HMB-native and NDB-native scored ~0.02 at c3700 while seeing 100% of the
# gold, versus 0.309 / 0.501 at c0. Two explanations fit:
#   (a) the models genuinely collapse on padded documents, or
#   (b) merely *setting* a 4096/8192 window breaks them, since both were fine-tuned at
#       --max_length 512.
#
# This separates them by holding the documents fixed at c0 -- the exact corpus that passed the
# sanity gate -- and varying only max_seq_length. If NDB scores ~0.50 at window=512 and ~0.03
# at window=4096 on identical text, the window is the problem and every "native" long-context
# number in this benchmark is measuring a configuration artifact, not long-context ability.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

OUT="${OUT:-outputs/eval/window_probe}"
CORPUS_DIR=$(find outputs/translation/runs -type d -path "*/BeIR_scifact/beir" | head -1)
HMB="outputs/models/dual_encoder/cls_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model"
NDB="outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model"

mkdir -p "$OUT" logs/slurm

# label|model|window   -- documents identical throughout (unpadded scifact)
PROBES=(
  "NDB-w512|${NDB}|512"
  "NDB-w1024|${NDB}|1024"
  "NDB-w4096|${NDB}|4096"
  "HMB-w512|${HMB}|512"
  "HMB-w1024|${HMB}|1024"
  "HMB-w8192|${HMB}|8192"
)

for p in "${PROBES[@]}"; do
    IFS='|' read -r label model window <<< "$p"
    echo
    echo "########## $label (window=$window, unpadded scifact) ##########"
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path "$model" \
        --model_label "$label" \
        --corpus_dir "$CORPUS_DIR" \
        --output_file "$OUT/$label/results.json" \
        --batch_size 16 \
        --max_length "$window" \
        --pooling cls \
        --force_reencode \
      || echo "!! FAILED: $label"
done

echo
echo "=== window probe: same documents, only max_seq_length varies ==="
python - <<'PY'
import glob, json, os
rows = []
for f in glob.glob(os.path.join(os.environ.get("OUT", "outputs/eval/window_probe"),
                                "*", "results.json")):
    d = json.load(open(f))
    rows.append((d["model"], d["config"]["max_length"], d["metrics"]["ndcg_at_10"]))
print(f"{'arm':12s} {'window':>7s} {'NDCG@10':>8s}")
for m, w, n in sorted(rows, key=lambda r: (r[0].split('-')[0], r[1])):
    print(f"{m:12s} {w:7d} {n:8.4f}")
print()
print("Expect ~0.501 (NDB) / ~0.309 (HMB) at window=512 -- these reproduce the c0 gate.")
print("If the score collapses as the window grows on IDENTICAL text, the long-window")
print("configuration is broken for these checkpoints and 'native' arms measure nothing.")
PY

#!/bin/bash
#SBATCH --job-name=lc_decisive_sft
#SBATCH --output=logs/slurm/lc_decisive_sft_%j.out
#SBATCH --error=logs/slurm/lc_decisive_sft_%j.err
#SBATCH --partition=L4-12h
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Decisive cells for the LENGTH-ADAPTED checkpoint.
#
# The question this answers: does training on welded long documents make HMB's 8192-token
# window usable? Before adaptation, native encoding scored 0.000 at c27000 while the same
# model chunked scored 0.129 -- the window was not merely unhelpful, it was worse than not
# using it.
#
# Only the two new-checkpoint arms are run here. Every comparison point already exists and is
# reused rather than recomputed:
#
#   HMB-native      (pre-SFT)  0.000   does adaptation move it at all?
#   HMB-chunkpara   (pre-SFT)  0.129   bar 1: is the window better than chunking the same model?
#   BM25 full-doc              0.218   bar 2: is it better than lexical matching with no window?
#   mE5L-chunkpara             0.403   bar 3: is it competitive?
#
# c3700 runs first as an anchor -- a length-adapted model should not have regressed on the
# short rung, and a collapse there would indicate the SFT damaged retrieval rather than
# extending it.

set -o pipefail
mkdir -p logs/slurm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

BENCH="${BENCH:-data/retrieval/beir_longctx/v1/BeIR_scifact}"
OUT="${OUT:-outputs/eval/longctx_decisive}"
COND="${COND:-random}"
POS="${POS:-middle}"
RUNGS="${RUNGS:-3700 19000 27000}"

SFT_DIR="${SFT_DIR:-outputs/models/dual_encoder/cls_pooling/beir_longctx/hebmodernbert/HMB-base-final-longctx}"
CKPT="$SFT_DIR/model"
[ -d "$CKPT" ] || CKPT="$SFT_DIR"
echo "Length-adapted checkpoint: $CKPT"
ls "$CKPT" | head -5
echo

# label|model|pooling|strategy|window|batch
ARMS=(
  "HMBlc-native|${CKPT}|cls|native|8192|8"
  "HMBlc-chunkpara|${CKPT}|cls|chunked_para|512|64"
)

for rung in $RUNGS; do
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r label model pooling strategy window batch <<< "$arm"
        dest="$OUT/c${rung}_${COND}_${POS}/${label}.json"
        echo
        echo "########## $label @ c${rung} ##########"
        python src/model/eval/eval_longctx.py \
            --benchmark_dir "$BENCH" \
            --condition "$COND" --rung "$rung" --position "$POS" \
            --model_name_or_path "$model" --model_label "$label" \
            --pooling "$pooling" --strategy "$strategy" --window "$window" \
            --batch_size "$batch" \
            --output_file "$dest" \
            --force_reencode \
          || echo "!! FAILED: $label @ c$rung"
    done
done

echo
echo "============ DID LENGTH ADAPTATION WORK? ============"
python - <<'PY'
import glob, json, os
root = os.environ.get("OUT", "outputs/eval/longctx_decisive")
data = {}
for f in glob.glob(os.path.join(root, "*", "*.json")):
    d = json.load(open(f))
    data[(d["rung_chars"], d["arm"])] = d["metrics"]["ndcg_at_10"]

BM25 = {3700: 0.4284, 19000: None, 27000: 0.2184}   # measured, full-document, no window
rungs = sorted({k[0] for k in data})
arms = ["HMBlc-native", "HMBlc-chunkpara", "HMB-native", "HMB-chunkpara",
        "NDB-chunkpara", "mE5L-chunkpara"]

print(f"{'arm':17s} " + " ".join(f"{('c%d'%r):>9s}" for r in rungs))
print("-" * (18 + 10 * len(rungs)))
for a in arms:
    if not any((r, a) in data for r in rungs):
        continue
    cells = [f"{data[(r,a)]:9.3f}" if (r, a) in data else f"{'-':>9s}" for r in rungs]
    tag = "  <- length-adapted" if a.startswith("HMBlc") else ""
    print(f"{a:17s} " + " ".join(cells) + tag)
print(f"{'BM25 (full doc)':17s} " +
      " ".join(f"{BM25[r]:9.3f}" if BM25.get(r) else f"{'-':>9s}" for r in rungs))

for r in [x for x in rungs if x >= 19000]:
    new = data.get((r, "HMBlc-native"))
    if new is None:
        continue
    print(f"\n=== VERDICT @ c{r} ===")
    for label, ref in [("pre-SFT HMB-native", data.get((r, "HMB-native"))),
                       ("its own chunked variant", data.get((r, "HMBlc-chunkpara"))),
                       ("pre-SFT HMB-chunkpara", data.get((r, "HMB-chunkpara"))),
                       ("BM25 full-document", BM25.get(r)),
                       ("mE5L-chunkpara", data.get((r, "mE5L-chunkpara")))]:
        if ref is None:
            continue
        verdict = "BEATS" if new > ref else "loses to"
        print(f"  HMBlc-native {new:.3f} {verdict:9s} {label:24s} {ref:.3f}  ({new-ref:+.3f})")
PY

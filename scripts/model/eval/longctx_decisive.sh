#!/bin/bash
#SBATCH --job-name=lc_decisive
#SBATCH --output=logs/slurm/lc_decisive_%j.out
#SBATCH --error=logs/slurm/lc_decisive_%j.err
#SBATCH --partition=L4-12h
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# THE DECISIVE CELLS: c19000 and c27000.
#
# These are the only rungs where HMB reads the whole document and no baseline can:
#
#              c19000 (19,000 chars)        c27000 (27,000 chars)
#   HMB   8192  4,738 tok -> 100% native     6,733 tok -> 100% native
#   NDB   4096  4,620 tok ->  ~89%           6,564 tok ->  ~62%
#   mE5    512  6,825 tok ->  ~7.5%          9,698 tok ->  ~5.3%
#
# If HMB's window advantage is worth anything, it shows here or nowhere. The comparison that
# decides it is HMB-native vs the *chunked* baselines -- beating mE5-truncate only proves a
# 512-token model cannot see token 5,000, which nobody disputes.
#
# c3700 runs first as an ANCHOR, not a result: mE5-chunkpara scored 0.443 and mE5-trunc 0.082
# there on a previously-validated run, so reproducing those confirms the per-arm embedding
# cache fix. (All four arms previously shared one cache directory and overwrote each other,
# which made NDB score 0.026 while reading 100% of the gold.)
#
# Position bin = middle throughout: the neutral choice, and the one a truncating model cannot
# reach by luck. start/end are a separate sweep.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htr
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export TOKENIZERS_PARALLELISM=false

BENCH="${BENCH:-data/retrieval/beir_longctx/v1/BeIR_scifact}"
OUT="${OUT:-outputs/eval/longctx_decisive}"
COND="${COND:-random}"
POS="${POS:-middle}"
RUNGS="${RUNGS:-3700 19000 27000}"

HMB="outputs/models/dual_encoder/cls_pooling/beir_hebrew_hn/hebmodernbert/HebrewModernBERT-base-final/model"
NDB="outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model"

# label|model|pooling|strategy|window|batch
ARMS=(
  "HMB-native|${HMB}|cls|native|8192|8"
  "HMB-chunkpara|${HMB}|cls|chunked_para|512|64"
  "NDB-native|${NDB}|cls|native|4096|16"
  "NDB-chunkpara|${NDB}|cls|chunked_para|512|64"
  "mE5L-trunc|intfloat/multilingual-e5-large|mean|truncate|512|64"
  "mE5L-chunkpara|intfloat/multilingual-e5-large|mean|chunked_para|512|64"
  "mE5B-chunkpara|intfloat/multilingual-e5-base|mean|chunked_para|512|128"
)

mkdir -p "$OUT" logs/slurm

for rung in $RUNGS; do
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r label model pooling strategy window batch <<< "$arm"
        dest="$OUT/c${rung}_${COND}_${POS}/${label}.json"
        if [ -f "$dest" ]; then
            echo "[skip] $label c$rung already done"; continue
        fi
        echo
        echo "########## $label @ c${rung} (${COND}/${POS}) ##########"
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
echo "=================== DECISIVE RESULTS ==================="
python - <<'PY'
import glob, json, os, re
root = os.environ.get("OUT", "outputs/eval/longctx_decisive")
C0 = {"HMB-native":0.309,"HMB-chunkpara":0.309,"NDB-native":0.501,"NDB-chunkpara":0.501,
      "mE5L-trunc":0.581,"mE5L-chunkpara":0.581,"mE5B-chunkpara":0.549}
data = {}
for f in glob.glob(os.path.join(root, "*", "*.json")):
    d = json.load(open(f))
    data[(d["rung_chars"], d["arm"])] = d
rungs = sorted({k[0] for k in data})
arms = ["mE5L-chunkpara","mE5B-chunkpara","NDB-chunkpara","NDB-native",
        "HMB-native","HMB-chunkpara","mE5L-trunc"]

print(f"{'arm':16s} {'c0':>7s} " + " ".join(f"{('c%d'%r):>9s}" for r in rungs))
print("-" * (24 + 10*len(rungs)))
for a in arms:
    if not any((r,a) in data for r in rungs): continue
    cells = []
    for r in rungs:
        d = data.get((r,a))
        cells.append(f"{d['metrics']['ndcg_at_10']:9.3f}" if d else f"{'-':>9s}")
    print(f"{a:16s} {C0.get(a,0):7.3f} " + " ".join(cells))

print("\n--- retention vs each arm's own c0 (robustness, NOT superiority) ---")
for a in arms:
    if not any((r,a) in data for r in rungs): continue
    c0 = C0.get(a)
    cells = []
    for r in rungs:
        d = data.get((r,a))
        cells.append(f"{d['metrics']['ndcg_at_10']/c0:8.0%} " if d and c0 else f"{'-':>9s}")
    print(f"{a:16s} {'100%':>7s} " + " ".join(cells))

print("\n--- windows encoded per document ---")
for a in arms:
    if not any((r,a) in data for r in rungs): continue
    cells = [f"{data[(r,a)]['windows_per_doc']:9.1f}" if (r,a) in data else f"{'-':>9s}"
             for r in rungs]
    print(f"{a:16s} {'1.0':>7s} " + " ".join(cells))

for r in [x for x in rungs if x >= 19000]:
    hmb = data.get((r,"HMB-native"))
    if not hmb: continue
    print(f"\n=== VERDICT @ c{r} ===")
    h = hmb["metrics"]["ndcg_at_10"]
    for rival in ["mE5L-chunkpara","NDB-chunkpara","NDB-native","HMB-chunkpara"]:
        d = data.get((r,rival))
        if not d: continue
        v = d["metrics"]["ndcg_at_10"]
        verdict = "HMB wins" if h > v else "HMB loses"
        print(f"  HMB-native {h:.3f} vs {rival} {v:.3f}  -> {verdict} ({h-v:+.3f})")
PY

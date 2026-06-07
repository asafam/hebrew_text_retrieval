# Hebrew Retrieval Benchmark

Evaluation of Hebrew-specific retrieval models vs multilingual baselines on translated BeIR datasets.

---

## Bottom Line (NDCG@10)

> Primary metric. Higher is better. `—` = pending. Partial = eval still running.

| Model | arguana | fiqa | nfcorpus | scidocs | scifact | **avg** |
|-------|--------:|-----:|---------:|--------:|--------:|--------:|
| mE5-large | 0.327 | 0.382 | 0.385 | 0.232 | 0.584 | **0.382** |
| **NeoDictaBERT** dual-enc (BeIR, no hard-neg) | 0.350 | 0.324 | 0.350 | 0.157 | 0.501 | **0.336** |
| mE5-base | 0.272 | 0.288 | 0.335 | 0.217 | 0.553 | **0.333** |
| HMB 20250622 + hard-neg (lr 2e-5) | 0.087 | 0.082 | 0.271 | 0.066 | 0.341 | **0.169** |
| HMB 20250619 + hard-neg (lr 2e-5) | 0.061 | 0.082 | 0.258 | 0.064 | 0.301 | **0.153** |
| HMB v2 (lr 2e-5, no hard-neg) | 0.024 | 0.102 | 0.286 | 0.062 | 0.233 | 0.141 |
| HMB v2 + hard-neg (lr 5e-6) | 0.006 | 0.045 | 0.211 | 0.041 | 0.171 | 0.095 |
| HMB v2 (lr 5e-6, no hard-neg) | 0.003 | 0.055 | 0.205 | 0.041 | 0.132 | 0.087 |
| HMB HeQ (discarded) | 0.001 | 0.084 | 0.066 | 0.003 | 0.012 | 0.033 |
| NeoDictaBERT (BeIR + hard-neg, bs=16) | 0.267 | 0.228 | 0.336 | 0.116 | 0.525 | **0.294** |
| HMB **new base** ba48000 (ep0, early) + hard-neg | 0.005 | 0.036 | 0.258 | 0.050 | 0.260 | **0.122** |
| HMB **new base** ba76000 (ep0, early) + hard-neg | 0.003 | 0.049 | 0.252 | 0.046 | 0.222 | **0.114** |

### New-base sneak peek (ba48000, epoch 0 — very early)
Converted + SFT'd the in-progress new HMB base at its earliest available checkpoint (ba48000, ~5% of the
training the 2025 checkpoints had). Result: 0.122 avg. In-domain datasets already reasonable (nfcorpus 0.258,
scifact 0.260 — comparable to fully-trained 2025 checkpoints), but **zero-shot domains still collapse**
(arguana 0.005, scidocs 0.050) — the same pattern as all HMB checkpoints. **Inconclusive** because it's epoch 0:
a low score this early is expected. Re-test at a much later checkpoint (e.g. ep5+) once the new run progresses.
NOTE: the new pretraining appeared stalled at ba48000 since Jun 4 despite the job running — worth checking the
H200 job is actually advancing.

### Pooling experiment (CLS vs mean) — mean pooling does NOT help
Tested mean pooling against CLS on all trained models (hypothesis: NeoBERT/ModernBERT drop NSP so CLS may be
weak). Result — flat to slightly worse:
| Model | CLS | mean | Δ |
|-------|----:|----:|---:|
| NeoDictaBERT | 0.336 | 0.331 | −0.005 |
| HMB 20250622 | 0.169 | 0.160 | −0.009 |
| HMB 20250619 | 0.153 | 0.154 | +0.000 |
| HMB new-base ba76000 | 0.114 | 0.141 | **+0.027** |
InfoNCE fine-tuning already shapes CLS into a good sentence vector regardless of pretraining objective, so
CLS is fine. Only exception: the very-early new base (ep0) gains from mean pooling — use mean pooling when
probing future early-pretraining checkpoints. **Keep CLS for mature checkpoints.**

### Model sizes (verified)
mE5-large 560M | HMB-large (est, 28L/1024/vocab150K) ~497M | NeoDictaBERT 363M | mE5-base 278M (86M body —
69% is the 250K multilingual vocab) | HMB-base new 226M | HMB-base v2 187M.

### Standings (NOTE: NeoDictaBERT is Dicta's model, NOT ours — it's a baseline)
- Best overall: **mE5-large 0.382** (Microsoft)
- **NeoDictaBERT 0.336** (Dicta) — beats mE5-base
- mE5-base 0.333 (Microsoft)
- **OUR model = HebrewModernBERT (HMB)** — best variant only **0.169**, ~2x behind the baselines.
  All external Hebrew/multilingual options outperform our current HMB checkpoints. A competitive own model
  depends on the new HMB base pretraining (and likely an HMB-large ~497M to match mE5-large's size class).

**Hard negatives HURT NeoDictaBERT** (0.336 → 0.294). Two likely causes: (1) BM25 false negatives — on
datasets like nfcorpus (many relevant docs/query) BM25's top hits are often genuinely relevant but unlabeled,
so training wrongly pushes them away; (2) the hard-neg run was forced to bs=16 (OOM at higher) vs bs=32 for the
plain run, giving fewer in-batch negatives. **Recommendation stands: ship plain NeoDictaBERT (no hard-neg).**
Hard negatives are not worth pursuing for NDB unless false-negative filtering is added (e.g. denoise via a
cross-encoder, or skip top-K BM25 hits before sampling negatives).

### Investigating the HMB base checkpoint (in progress)
HMB v2 base appears to produce weak retrieval representations (best HMB result 0.141 vs NDB 0.336).
Testing two alternative HMB base checkpoints through SFT (lr=2e-5 + hard negatives, the best recipe found)
to determine whether it's this specific checkpoint or the HMB family:
- `ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0` (most trained — 7ep/896K batches) → label `hmb-20250622-hn`
- `ModernBERT-Hebrew-base_20250619_2241/ep7-ba895014-rank0` → label `hmb-20250619-hn`
A newer HMB base is pretraining in the background (~1-2 weeks out).

### HMB checkpoint experiment — conclusion
Tested 3 different HMB base checkpoints (v2, 20250622/ep7, 20250619/ep7) through SFT with the best recipe
(lr=2e-5 + hard negatives + cosine). Checkpoint quality *does* matter — 20250622 reached 0.169, the best HMB
result (vs 0.141 for v2). But **even the best HMB checkpoint lands ~2× below NeoDictaBERT (0.336) and mE5-base
(0.333).** All HMB checkpoints show the same pattern: they fit high-data training domains (nfcorpus, scifact)
but collapse on zero-shot domains (arguana 0.06-0.09, scidocs 0.06). The existing HMB family does not build
transferable retrieval representations. **Decision: ship NeoDictaBERT; revisit HMB when the new base finishes
pretraining (~1-2 weeks out).**

**STATUS (held):** The 2025 HMB pretraining is considered flawed for retrieval — no further HMB SFT/eval
experiments until the new HMB base completes pretraining (job hmbbh-p01 on H200). When that checkpoint is
ready, run it through the existing pipeline:
`sbatch --export=ALL,MODEL_PATH=<new_ckpt>,OUTPUT_DIR=<out> scripts/model/dual_encoder/heq/train/train_dual_encoder_beir_hn_hmb_generic.sh`
then `eval_beir_hmb_generic.sh`. The pipeline (BeIR hard-neg SFT at lr=2e-5 + eval) is ready and reusable.

### Key finding so far
- **NeoDictaBERT matches mE5-base** (0.336 vs 0.333) with only 125K Hebrew training pairs — a strong Hebrew-specific result.
- **HebrewModernBERT lags badly (0.087).** The LR fix (v2) recovered it from fully-broken to partially-working *on in-domain data* (nfcorpus 0.205, which had 110K train pairs), but it **fails to generalize to zero-shot domains** (arguana 0.003, scidocs 0.041 — datasets with no train data, where NDB scored 0.350 / 0.157). This points to the base model: HebrewModernBERT's representations do not transfer for retrieval the way NeoDictaBERT's do. Hard-negative training (v3) is in progress but is unlikely to close a 4× gap on its own.

---

## Models

| Model | Type | Training data | Checkpoint |
|-------|------|--------------|------------|
| `intfloat/multilingual-e5-large` | Multilingual baseline | MS-MARCO, NLI, 1B web pairs | HuggingFace hub |
| `intfloat/multilingual-e5-base` | Multilingual baseline | Same | HuggingFace hub |
| NeoDictaBERT dual-enc | Hebrew-specific | Hebrew BeIR train splits | `outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model` |
| HMB dual-enc v2 *(active)* | Hebrew-specific | Hebrew BeIR train splits | `outputs/models/dual_encoder/cls_pooling/beir_hebrew/hebmodernbert/ModernBERT-Hebrew-base_v2_lr5e6_ep10/model` |

**Base checkpoints:**
- HebrewModernBERT: `ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_v2`
- NeoDictaBERT: `dicta-il/NeoDictaBERT` (HuggingFace) — **28 layers, 768 width** (~257M params); context up to 4,096 tokens. Deeper than BERT-base (12L) and HMB (22L), with the same width. See Shalumov et al.

---

## Datasets

Five BeIR datasets translated to Hebrew (Gemini 2.0 Flash Lite, prompt v20260531):

| Dataset | Domain | Train pairs | Test queries | Corpus size |
|---------|--------|------------:|-------------:|------------:|
| nfcorpus | Medical / nutritional | 110,575 | 323 | 3,633 |
| fiqa | Financial QA | 14,166 | 648 | 57,600 |
| scifact | Scientific fact verification | 919 | 300 | 5,183 |
| arguana | Argument retrieval | — (test-only) | 1,401 | 8,674 |
| scidocs | Scientific citation | — (test-only) | 1,000 | 25,657 |

**Source:** `outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus/`

---

## Methodology

### Evaluation
- **Metric:** NDCG@10 (primary), NDCG@100, R@100, MRR
- **Search:** FAISS IndexFlatIP on L2-normalized embeddings (cosine similarity), top-100
- **Script:** `src/model/eval/eval_beir_retrieval_zeroshot.py`
- **Partition:** L4 (H200 reserved for pretraining only)

### mE5 instruction prefixes
mE5 models use `"query: "` / `"passage: "` prefixes (auto-detected by model name). Hebrew models use no prefix.

### Fairness note
mE5 was trained on English MS-MARCO + NLI — not on nfcorpus/fiqa/scifact/arguana/scidocs directly. Hebrew is not in mE5's training data. HMB and NDB are trained on Hebrew BeIR **train** splits, evaluated on **test** splits — in-domain but not contaminated. The comparison: *multilingual scale + English retrieval training* vs *Hebrew-specific pretraining + Hebrew in-domain retrieval training*.

---

## Training Log

### Run 1 — NeoDictaBERT (BeIR) ✅
- **Script:** `train_dual_encoder_beir_hebrew_neodictabert.sh`
- **Job:** 16635969 | A100, dsiasaf01 | 47 min
- **Config:** LR=2e-5, epochs=10, bs=32, grad_accum=2, CLS pooling, bf16
- **Final train loss:** 0.056 (converged well)
- **Checkpoint:** `outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model`
- **Early result:** NDCG@10 arguana = **0.350** (beats mE5-large 0.327)

### Run 2 — HebrewModernBERT (BeIR v1) ⚠️ discarded
- **Script:** `train_dual_encoder_beir_hebrew_hebmodernbert.sh`
- **Job:** 16635970 | A100, dsiasaf01 | 3h 20min
- **Config:** LR=2e-5, epochs=10, bs=32, grad_accum=2, CLS pooling, bf16
- **Final train loss:** 0.505 (did not converge)
- **Diagnosis:** Loss plateaued at ~1.1 from epoch 5 onwards with no improvement. LR decayed to near-zero before the model broke through the plateau. ModernBERT requires more conservative LR than BERT-style models.
- **Result:** NDCG@10 arguana = 0.024 (broken — essentially untrained)
- **Checkpoint kept at:** `outputs/models/dual_encoder/cls_pooling/beir_hebrew/hebmodernbert/ModernBERT-Hebrew-base_v2/model`

### Run 3 — HebrewModernBERT (BeIR v2) 🔄 running
- **Script:** `train_dual_encoder_beir_hebrew_hebmodernbert_v2.sh`
- **Job:** 16693346 | A100, dsiuriofir01
- **Config:** LR=**5e-6**, epochs=10, bs=32, grad_accum=2, CLS pooling, bf16, warmup_ratio=**0.1**, scheduler=**cosine**
- **Why different:** cosine schedule keeps LR higher for longer; warmup prevents early overshooting; 4× lower LR gives ModernBERT time to converge steadily
- **Checkpoint will be at:** `outputs/models/dual_encoder/cls_pooling/beir_hebrew/hebmodernbert/ModernBERT-Hebrew-base_v2_lr5e6_ep10/model`

---

## Extended Results (all metrics)

<details>
<summary>mE5-large</summary>

| Dataset | NDCG@10 | NDCG@100 | R@100 | MRR |
|---------|--------:|---------:|------:|----:|
| arguana | 0.3271 | 0.3880 | 0.9536 | 0.2293 |
| fiqa | 0.3821 | 0.4614 | 0.8179 | 0.4123 |
| nfcorpus | 0.3853 | 0.5081 | 0.8173 | 0.4939 |
| scidocs | 0.2316 | 0.3332 | 0.7380 | 0.2692 |
| scifact | 0.5839 | 0.6212 | 0.8833 | 0.5547 |

</details>

<details>
<summary>mE5-base</summary>

| Dataset | NDCG@10 | NDCG@100 | R@100 | MRR |
|---------|--------:|---------:|------:|----:|
| arguana | 0.2723 | 0.3404 | 0.8944 | 0.1904 |
| fiqa | 0.2882 | 0.3669 | 0.7222 | 0.3045 |
| nfcorpus | 0.3354 | 0.4590 | 0.7771 | 0.4308 |
| scidocs | 0.2166 | 0.3078 | 0.6900 | 0.2408 |
| scifact | 0.5525 | 0.5878 | 0.8600 | 0.5141 |

</details>

<details>
<summary>HebrewModernBERT dual-enc (HeQ — discarded)</summary>

Trained on HeQ (Hebrew QA only). Near-zero on BeIR — single-domain QA training does not generalize.

| Dataset | NDCG@10 |
|---------|--------:|
| arguana | 0.0007 |
| fiqa | 0.0835 |
| nfcorpus | 0.0663 |
| scidocs | 0.0029 |
| scifact | 0.0117 |

</details>

---

## Next Steps

1. Wait for HMB v2 training (job 16693346) to finish
2. Check loss curve — if still plateauing, may need even lower LR or more epochs
3. Create eval script pointing to `..._lr5e6_ep10/model` and submit on L4
4. Fill in bottom-line table once all evals complete
5. Run `python src/model/eval/collect_beir_results.py` to regenerate table

```bash
# After HMB v2 training finishes — submit eval
sbatch scripts/model/eval/eval_beir_hebmodernbert_beir_dualenc.sh  # update MODEL path to lr5e6 checkpoint first
```

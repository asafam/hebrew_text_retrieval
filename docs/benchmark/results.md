# Hebrew Retrieval Benchmark

Evaluation of Hebrew-specific retrieval models vs multilingual baselines on translated BeIR datasets.

> **Numbers regenerated 2026-07-30** after two metric bugs were fixed (commits `bd362aa`, `1b47216`).
> Every figure below is post-fix and matches `pytrec_eval` to <1e-12. See
> [Metric corrections](#metric-corrections-2026-07-30) for what changed and why. Pre-fix values are
> preserved in each `results.json` under `metrics_pre_fix`.

---

## Bottom Line (NDCG@10)

> Primary metric. Higher is better.

| Model | arguana | fiqa | nfcorpus | scidocs | scifact | **avg** |
|-------|--------:|-----:|---------:|--------:|--------:|--------:|
| mE5-large | 0.440 | 0.335 | 0.294 | 0.139 | 0.581 | **0.358** |
| **NeoDictaBERT** dual-enc (BeIR, no hard-neg) | 0.451 | 0.288 | 0.329 | 0.093 | 0.501 | **0.332** |
| NeoDictaBERT, mean pooling | 0.456 | 0.285 | 0.327 | 0.092 | 0.484 | **0.329** |
| mE5-base | 0.361 | 0.241 | 0.248 | 0.125 | 0.549 | **0.305** |
| NeoDictaBERT (BeIR + hard-neg, bs=16) | 0.333 | 0.196 | 0.319 | 0.064 | 0.523 | **0.287** |
| HMB 20250622 + hard-neg (lr 2e-5) | 0.101 | 0.061 | 0.253 | 0.034 | 0.340 | **0.158** |
| HMB phase0.2 (CLS) | 0.100 | 0.061 | 0.253 | 0.039 | 0.309 | 0.152 |
| HMB 20250622, mean pooling | 0.105 | 0.063 | 0.249 | 0.034 | 0.294 | 0.149 |
| HMB final plain (CLS) | 0.063 | 0.114 | 0.262 | 0.044 | 0.238 | 0.144 |
| HMB 20250619, mean pooling | 0.101 | 0.063 | 0.242 | 0.030 | 0.285 | 0.144 |
| HMB 20250619 + hard-neg (lr 2e-5) | 0.067 | 0.064 | 0.242 | 0.034 | 0.296 | **0.141** |
| HMB v2 (lr 2e-5, no hard-neg) | 0.026 | 0.081 | 0.267 | 0.034 | 0.232 | 0.128 |
| HMB **new base** ba76000, mean pooling | 0.012 | 0.062 | 0.244 | 0.025 | 0.295 | 0.128 |
| HMB **new base** ba48000 (ep0) + hard-neg | 0.005 | 0.024 | 0.237 | 0.023 | 0.258 | **0.109** |
| HMB **new base** ba76000 (ep0) + hard-neg | 0.004 | 0.032 | 0.232 | 0.023 | 0.220 | **0.102** |
| HMB random init (control) | 0.001 | 0.004 | 0.190 | 0.002 | 0.223 | 0.084 |
| HMB v2 + hard-neg (lr 5e-6) | 0.006 | 0.031 | 0.189 | 0.019 | 0.170 | 0.083 |
| HMB v2 (lr 5e-6, no hard-neg) | 0.003 | 0.041 | 0.184 | 0.020 | 0.129 | 0.076 |
| HMB HeQ (discarded) | 0.001 | 0.058 | 0.037 | 0.002 | 0.012 | 0.022 |

### Standings (NOTE: NeoDictaBERT is Dicta's model, NOT ours — it's a baseline)
- Best overall: **mE5-large 0.358** (Microsoft)
- **NeoDictaBERT 0.332** (Dicta) — now clearly ahead of mE5-base, and within 0.026 of mE5-large
- mE5-base 0.305 (Microsoft)
- **OUR model = HebrewModernBERT (HMB)** — best variant only **0.158**, ~2× behind the baselines.
  All external Hebrew/multilingual options outperform our current HMB checkpoints. A competitive own model
  depends on the new HMB base pretraining (and likely an HMB-large ~497M to match mE5-large's size class).

### ⚠️ Conclusion strengthened by the metric fix
Previously NeoDictaBERT and mE5-base looked **tied** (0.336 vs 0.333, +0.003). With correct metrics
NeoDictaBERT leads by **+0.028** (0.332 vs 0.305) — a ~9% relative margin, no longer a coin flip. The gap to
mE5-large also narrowed from 0.046 to 0.026.

The reason is arguana, where the old code let every model retrieve the query's own document at rank 1
(see [Metric corrections](#metric-corrections-2026-07-30)). NeoDictaBERT benefits most from the fix
(0.350 → 0.451) and now **beats mE5-large on arguana outright** (0.451 vs 0.440) — the only dataset where
any Hebrew model leads. **This makes the "ship NeoDictaBERT" decision stronger than when it was taken.**

### Per-dataset winners
| Dataset | Winner | Score | Runner-up |
|---|---|---:|---|
| arguana | **NeoDictaBERT (mean pool)** | 0.456 | NeoDictaBERT CLS 0.451, mE5-large 0.440 |
| fiqa | mE5-large | 0.335 | NeoDictaBERT 0.288 |
| nfcorpus | **NeoDictaBERT** | 0.329 | mE5-large 0.294 |
| scidocs | mE5-large | 0.139 | mE5-base 0.125 |
| scifact | mE5-large | 0.581 | mE5-base 0.549 |

NeoDictaBERT wins the two datasets where the corpus is Hebrew-translated *and* the task rewards semantic
matching over lexical overlap. It is weakest on scidocs, the hardest dataset for every model.

### New-base sneak peek (ba48000, epoch 0 — very early)
Converted + SFT'd the in-progress new HMB base at its earliest available checkpoint (ba48000, ~5% of the
training the 2025 checkpoints had). Result: 0.109 avg. In-domain datasets already reasonable (nfcorpus 0.237,
scifact 0.258 — comparable to fully-trained 2025 checkpoints), but **zero-shot domains still collapse**
(arguana 0.005, scidocs 0.023) — the same pattern as all HMB checkpoints. **Inconclusive** because it's epoch 0:
a low score this early is expected. Re-test at a much later checkpoint (e.g. ep5+) once the new run progresses.
NOTE: the new pretraining appeared stalled at ba48000 since Jun 4 despite the job running — worth checking the
H200 job is actually advancing.

### Random-init control
A randomly initialised HMB scores **0.084** — mostly from nfcorpus (0.190) and scifact (0.223). This is the
floor: any model near it has learned nothing transferable. It also shows nfcorpus/scifact NDCG@10 has a high
"free" baseline (small corpora, many relevant docs), so **only arguana and scidocs cleanly separate a working
model from an untrained one.** Judge HMB progress on those two.

### Pooling experiment (CLS vs mean) — mean pooling does NOT help
Tested mean pooling against CLS on all trained models (hypothesis: NeoBERT/ModernBERT drop NSP so CLS may be
weak). Result — flat to slightly worse:

| Model | CLS | mean | Δ |
|-------|----:|----:|---:|
| NeoDictaBERT | 0.332 | 0.329 | −0.004 |
| HMB 20250622 | 0.158 | 0.149 | −0.008 |
| HMB 20250619 | 0.141 | 0.144 | +0.004 |
| HMB new-base ba76000 | 0.102 | 0.128 | **+0.025** |

InfoNCE fine-tuning already shapes CLS into a good sentence vector regardless of pretraining objective, so
CLS is fine. Only exception: the very-early new base (ep0) gains from mean pooling — use mean pooling when
probing future early-pretraining checkpoints. **Keep CLS for mature checkpoints.** (Conclusion unchanged by
the metric fix; all four deltas moved by ≤0.004.)

### Model sizes (verified)
mE5-large 560M | HMB-large (est, 28L/1024/vocab150K) ~497M | NeoDictaBERT 363M | mE5-base 278M (86M body —
69% is the 250K multilingual vocab) | HMB-base new 226M | HMB-base v2 187M.

**Hard negatives HURT NeoDictaBERT** (0.332 → 0.287; conclusion unchanged by the fix). Two likely causes:
(1) BM25 false negatives — on datasets like nfcorpus (many relevant docs/query) BM25's top hits are often
genuinely relevant but unlabeled, so training wrongly pushes them away; (2) the hard-neg run was forced to
bs=16 (OOM at higher) vs bs=32 for the plain run, giving fewer in-batch negatives. **Recommendation stands:
ship plain NeoDictaBERT (no hard-neg).** Hard negatives are not worth pursuing for NDB unless false-negative
filtering is added (e.g. denoise via a cross-encoder, or skip top-K BM25 hits before sampling negatives).

### HMB checkpoint experiment — conclusion
Tested 3 different HMB base checkpoints (v2, 20250622/ep7, 20250619/ep7) through SFT with the best recipe
(lr=2e-5 + hard negatives + cosine). Checkpoint quality *does* matter — 20250622 reached 0.158, the best HMB
result (vs 0.128 for v2). But **even the best HMB checkpoint lands ~2× below NeoDictaBERT (0.332) and mE5-base
(0.305).** All HMB checkpoints show the same pattern: they fit high-data training domains (nfcorpus, scifact)
but collapse on zero-shot domains (arguana 0.06–0.10, scidocs 0.03). The existing HMB family does not build
transferable retrieval representations. **Decision: ship NeoDictaBERT; revisit HMB when the new base finishes
pretraining.**

**STATUS (held):** The 2025 HMB pretraining is considered flawed for retrieval — no further HMB SFT/eval
experiments until the new HMB base completes pretraining (job hmbbh-p01 on H200). When that checkpoint is
ready, run it through the existing pipeline:
`sbatch --export=ALL,MODEL_PATH=<new_ckpt>,OUTPUT_DIR=<out> scripts/model/dual_encoder/heq/train/train_dual_encoder_beir_hn_hmb_generic.sh`
then `eval_beir_hmb_generic.sh`. The pipeline (BeIR hard-neg SFT at lr=2e-5 + eval) is ready and reusable.

### Key findings
- **NeoDictaBERT clearly beats mE5-base** (0.332 vs 0.305) with only 125K Hebrew training pairs — a strong
  Hebrew-specific result, and stronger than it appeared before the metric fix.
- **HebrewModernBERT lags badly.** It works *on in-domain data* (nfcorpus 0.253, which had 110K train pairs)
  but **fails to generalize to zero-shot domains** (arguana 0.10, scidocs 0.034 — datasets with no train data,
  where NDB scored 0.451 / 0.093). This points to the base model: HebrewModernBERT's representations do not
  transfer for retrieval the way NeoDictaBERT's do.
- **Translation quality is not the bottleneck.** A separate controlled analysis (`failure-analysis.md`)
  found 63% of Hebrew failures also fail on the English source, and the Hebrew-only failures show no
  translation defect. The lever is the encoder, not the translation.

---

## Metric corrections (2026-07-30)

Two bugs in `src/model/eval/eval_beir_retrieval_zeroshot.py` made earlier numbers non-comparable to published
BeIR figures. All 120 stored results were recomputed from cached embeddings via
`scripts/model/eval/rescore_beir_results.py`.

**1. Ideal DCG was truncated.** NDCG normalized against only the retrieved top-k rather than the full qrels,
inflating scores in proportion to positives-per-query. Impact scaled with dataset density — largest on
nfcorpus (38 positives/query), negligible on scifact (~1).

**2. ArguAna self-retrieval.** ArguAna's query arguments are themselves corpus documents (92.4% of query ids
appear as doc ids). Models retrieved the query's own near-duplicate at rank 1, demoting the true
counterargument. Now excluded, decided per dataset by structural id overlap so fiqa's coincidental 9.3%
overlap is left alone.

**Mean effect per dataset:**

| Dataset | old NDCG@10 | new | Δ | cause |
|---|---:|---:|---:|---|
| arguana | 0.091 | 0.114 | **+0.023** | self-exclusion |
| scifact | 0.310 | 0.308 | −0.002 | IDCG (~1 positive/query) |
| fiqa | 0.121 | 0.098 | −0.023 | IDCG |
| nfcorpus | 0.268 | 0.242 | −0.026 | IDCG (38 positives/query) |
| scidocs | 0.078 | 0.043 | −0.035 | IDCG |

**Rankings barely moved** — 4 of 24 models changed position, all by one place. The exception that matters is
NeoDictaBERT vs mE5-base, discussed above.

**`recall_at_100` changed meaning.** It previously reported *hit rate* (any relevant doc in the top 100),
which saturates near 1.0 on dense datasets and carries little signal. It is now true recall
(`|relevant ∩ top-100| / |all relevant|`). The old figure is retained as `hit_rate_at_100`.

**A third suspected bug was not real.** The DCG gain function was briefly changed to `2^rel − 1` on the belief
that published BeIR used it. Cross-checking against `pytrec_eval` showed `trec_eval`'s `ndcg_cut` uses
**linear** gain, so the original was already correct. Default remains linear.

---

## Models

| Model | Type | Training data | Checkpoint |
|-------|------|--------------|------------|
| `intfloat/multilingual-e5-large` | Multilingual baseline | MS-MARCO, NLI, 1B web pairs | HuggingFace hub |
| `intfloat/multilingual-e5-base` | Multilingual baseline | Same | HuggingFace hub |
| NeoDictaBERT dual-enc | Hebrew-specific | Hebrew BeIR train splits | `outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model` |
| HMB dual-enc v2 | Hebrew-specific | Hebrew BeIR train splits | `outputs/models/dual_encoder/cls_pooling/beir_hebrew/hebmodernbert/ModernBERT-Hebrew-base_v2_lr5e6_ep10/model` |

**Base checkpoints:**
- HebrewModernBERT: `ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_v2`
- NeoDictaBERT: `dicta-il/NeoDictaBERT` (HuggingFace) — **28 layers, 768 width** (~257M params); context up to 4,096 tokens. Deeper than BERT-base (12L) and HMB (22L), with the same width. See Shalumov et al.

---

## Datasets

Five BeIR datasets translated to Hebrew (`gemini-3.1-flash-lite`, prompt v20260531). Corpus/query counts are
measured from the exported files. See `tasks.md` for what each task asks, and `datasets.md`
for full statistics.

| Dataset | Domain | Train pairs | Test queries | Corpus size | Positives/query |
|---------|--------|------------:|-------------:|------------:|----------------:|
| nfcorpus | Medical / nutritional | 110,575 | 323 | 3,633 | 38.2 |
| fiqa | Financial QA | 14,166 | 648 | 57,600 | 2.6 |
| scifact | Scientific fact verification | 919 | 300 | 5,183 | 1.1 |
| arguana | Argument retrieval | — (test-only) | 1,401 | 8,674 | 1.0 |
| scidocs | Scientific citation | — (test-only) | 1,000 | 25,313 | 4.9 |

**Source:** `outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus/`

Two data caveats that affect scores:
- **scidocs** ships 25 explicit *negative* candidates per query (score 0) alongside its ~5 positives. 115
  positive documents are also missing from the translated corpus (25,313 exported vs 25,657 upstream), making
  scidocs recall slightly pessimistic.
- **nfcorpus** uses a graded 0/1/2 scale in test/validation, but its **train split has no grade-2 judgments** —
  training and evaluation define relevance differently.

---

## Methodology

### Evaluation
- **Metric:** NDCG@10 (primary), NDCG@100, recall@100, MRR — verified against `pytrec_eval` to <1e-12
- **Search:** FAISS IndexFlatIP on L2-normalized embeddings (cosine similarity), top-100
- **Script:** `src/model/eval/eval_beir_retrieval_zeroshot.py`
- **Self-exclusion:** `--exclude_self auto` (structural id-overlap test; fires on arguana only)
- **Partition:** L4 (H200 reserved for pretraining only)

### mE5 instruction prefixes
mE5 models use `"query: "` / `"passage: "` prefixes (auto-detected by model name). Hebrew models use no prefix.

### Fairness note
mE5 was trained on English MS-MARCO + NLI — not on nfcorpus/fiqa/scifact/arguana/scidocs directly. Hebrew is not in mE5's training data. HMB and NDB are trained on Hebrew BeIR **train** splits, evaluated on **test** splits — in-domain but not contaminated. The comparison: *multilingual scale + English retrieval training* vs *Hebrew-specific pretraining + Hebrew in-domain retrieval training*.

Note that arguana and scidocs are **test-only** for every model, so they are the cleanest zero-shot signal.

---

## Training Log

### Run 1 — NeoDictaBERT (BeIR) ✅
- **Script:** `train_dual_encoder_beir_hebrew_neodictabert.sh`
- **Job:** 16635969 | A100, dsiasaf01 | 47 min
- **Config:** LR=2e-5, epochs=10, bs=32, grad_accum=2, CLS pooling, bf16
- **Final train loss:** 0.056 (converged well)
- **Checkpoint:** `outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model`
- **Result:** avg NDCG@10 **0.332**; arguana **0.451** (beats mE5-large 0.440)

### Run 2 — HebrewModernBERT (BeIR v1) ⚠️ discarded
- **Script:** `train_dual_encoder_beir_hebrew_hebmodernbert.sh`
- **Job:** 16635970 | A100, dsiasaf01 | 3h 20min
- **Config:** LR=2e-5, epochs=10, bs=32, grad_accum=2, CLS pooling, bf16
- **Final train loss:** 0.505 (did not converge)
- **Diagnosis:** Loss plateaued at ~1.1 from epoch 5 onwards with no improvement. LR decayed to near-zero before the model broke through the plateau. ModernBERT requires more conservative LR than BERT-style models.
- **Result:** avg 0.128, arguana 0.026 (essentially untrained)

### Run 3 — HebrewModernBERT (BeIR v2) ✅ completed
- **Script:** `train_dual_encoder_beir_hebrew_hebmodernbert_v2.sh`
- **Job:** 16693346 | A100, dsiuriofir01
- **Config:** LR=**5e-6**, epochs=10, bs=32, grad_accum=2, CLS pooling, bf16, warmup_ratio=**0.1**, scheduler=**cosine**
- **Result:** avg 0.076 — the lower LR did *not* help; lr=2e-5 remains the better recipe.

---

## Extended Results (all metrics)

`recall@100` is true recall; `hit@100` is the older "any relevant doc retrieved" figure, kept for continuity.
Note that NDCG@100 can fall *below* NDCG@10 on nfcorpus: with 38 relevant docs per query the @100 ideal
denominator is far larger, so the two are not comparable across cutoffs on dense datasets.

<details>
<summary>mE5-large — avg NDCG@10 0.358</summary>

| Dataset | NDCG@10 | NDCG@100 | recall@100 | hit@100 | MRR |
|---------|--------:|---------:|-----------:|--------:|----:|
| arguana | 0.4404 | 0.4950 | 0.9536 | 0.9536 | 0.3698 |
| fiqa | 0.3351 | 0.4005 | 0.6504 | 0.8179 | 0.4123 |
| nfcorpus | 0.2939 | 0.2613 | 0.2559 | 0.8173 | 0.4939 |
| scidocs | 0.1386 | 0.1968 | 0.3139 | 0.7380 | 0.2692 |
| scifact | 0.5810 | 0.6180 | 0.8777 | 0.8833 | 0.5547 |

</details>

<details>
<summary>NeoDictaBERT dual-enc (BeIR, no hard-neg) — avg NDCG@10 0.332</summary>

| Dataset | NDCG@10 | NDCG@100 | recall@100 | hit@100 | MRR |
|---------|--------:|---------:|-----------:|--------:|----:|
| arguana | 0.4509 | 0.5069 | 0.9736 | 0.9736 | 0.3793 |
| fiqa | 0.2882 | 0.3651 | 0.6429 | 0.8040 | 0.3629 |
| nfcorpus | 0.3292 | 0.3728 | 0.4613 | 0.7988 | 0.4305 |
| scidocs | 0.0930 | 0.1447 | 0.2539 | 0.6190 | 0.1814 |
| scifact | 0.5008 | 0.5397 | 0.8279 | 0.8367 | 0.4641 |

</details>

<details>
<summary>mE5-base — avg NDCG@10 0.305</summary>

| Dataset | NDCG@10 | NDCG@100 | recall@100 | hit@100 | MRR |
|---------|--------:|---------:|-----------:|--------:|----:|
| arguana | 0.3613 | 0.4253 | 0.8958 | 0.8958 | 0.3012 |
| fiqa | 0.2407 | 0.3008 | 0.5283 | 0.7222 | 0.3045 |
| nfcorpus | 0.2476 | 0.2218 | 0.2223 | 0.7771 | 0.4308 |
| scidocs | 0.1251 | 0.1766 | 0.2810 | 0.6900 | 0.2408 |
| scifact | 0.5492 | 0.5843 | 0.8540 | 0.8600 | 0.5141 |

</details>

<details>
<summary>NeoDictaBERT + hard negatives (bs=16) — avg NDCG@10 0.287</summary>

| Dataset | NDCG@10 | NDCG@100 | recall@100 | hit@100 | MRR |
|---------|--------:|---------:|-----------:|--------:|----:|
| arguana | 0.3332 | 0.4121 | 0.9365 | 0.9365 | 0.2765 |
| fiqa | 0.1963 | 0.2713 | 0.5394 | 0.7191 | 0.2578 |
| nfcorpus | 0.3188 | 0.3623 | 0.4534 | 0.8390 | 0.4290 |
| scidocs | 0.0639 | 0.1103 | 0.2080 | 0.5620 | 0.1340 |
| scifact | 0.5227 | 0.5524 | 0.7638 | 0.7767 | 0.5003 |

</details>

<details>
<summary>HMB 20250622 + hard-neg — best HMB variant, avg NDCG@10 0.158</summary>

| Dataset | NDCG@10 | NDCG@100 | recall@100 | hit@100 | MRR |
|---------|--------:|---------:|-----------:|--------:|----:|
| arguana | 0.1006 | 0.1709 | 0.5353 | 0.5353 | 0.0872 |
| fiqa | 0.0607 | 0.1089 | 0.2845 | 0.4738 | 0.0859 |
| nfcorpus | 0.2527 | 0.2902 | 0.3663 | 0.7771 | 0.3789 |
| scidocs | 0.0345 | 0.0641 | 0.1295 | 0.3840 | 0.0730 |
| scifact | 0.3395 | 0.3872 | 0.6772 | 0.6900 | 0.3168 |

</details>

<details>
<summary>HebrewModernBERT dual-enc (HeQ — discarded) — avg NDCG@10 0.022</summary>

Trained on HeQ (Hebrew QA only). Near-zero on BeIR — single-domain QA training does not generalize.

| Dataset | NDCG@10 | NDCG@100 | recall@100 | hit@100 | MRR |
|---------|--------:|---------:|-----------:|--------:|----:|
| arguana | 0.0007 | 0.0089 | 0.0521 | 0.0521 | 0.0011 |
| fiqa | 0.0577 | 0.0893 | 0.2010 | 0.3488 | 0.0902 |
| nfcorpus | 0.0373 | 0.0496 | 0.0722 | 0.5356 | 0.1030 |
| scidocs | 0.0016 | 0.0032 | 0.0070 | 0.0230 | 0.0029 |
| scifact | 0.0117 | 0.0290 | 0.1194 | 0.1267 | 0.0088 |

</details>

---

## Reproducing these numbers

```bash
# Full eval for one model (GPU)
sbatch --export=ALL,MODEL_PATH=<ckpt>,MODEL_LABEL=<label> scripts/model/eval/eval_beir_hmb_generic.sh

# Recompute metrics for all stored results from cached embeddings (CPU, no re-encoding)
python scripts/model/eval/rescore_beir_results.py --verify

# Regenerate the model x dataset table
python src/model/eval/collect_beir_results.py
```

---

## Next Steps

1. **Blocked:** no further HMB experiments until the new base finishes pretraining — verify the H200 job is
   actually advancing past ba48000 (it appeared stalled since Jun 4).
2. Judge future HMB checkpoints on **arguana and scidocs**, not nfcorpus/scifact — the random-init control
   shows the latter two have a high free baseline.
3. If pursuing hard negatives for NeoDictaBERT, add false-negative filtering first; the current recipe
   costs 0.045 NDCG@10.
4. Consider an HMB-large (~497M) to compete in mE5-large's size class.

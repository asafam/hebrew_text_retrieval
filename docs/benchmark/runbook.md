# Hebrew Retrieval Model Evaluations

Zero-shot BEIR benchmark comparing Hebrew-specific retrieval models against multilingual baselines.

> **Numbers regenerated 2026-07-30** after two metric bugs were fixed. All figures are post-fix and verified
> against `pytrec_eval`. Analysis, per-dataset commentary and decisions live in **`results.md`** — this
> file is the results table and the runbook.

## Benchmark Design

**Research question:** Can a Hebrew-specific model fine-tuned for retrieval match or beat multilingual-E5 on Hebrew IR tasks?

**Answer so far: yes for mE5-base, not yet for mE5-large.** NeoDictaBERT (Hebrew-specific, Dicta's model)
scores 0.332 vs mE5-base 0.305 and mE5-large 0.358.

**2×2 comparison:**

|                       | Multilingual | Hebrew-specific |
|-----------------------|-------------|-----------------|
| **Retrieval-trained** | mE5-base, mE5-large | HebrewModernBERT dual-enc, NeoDictaBERT dual-enc |

All models evaluated on the same 5 translated Hebrew BeIR datasets (NDCG@10 primary metric).

---

## Results (NDCG@10)

<!-- Regenerate with: python src/model/eval/collect_beir_results.py -->

| Model | arguana | fiqa | nfcorpus | scidocs | scifact | **avg** |
|-------|--------:|-----:|---------:|--------:|--------:|--------:|
| mE5-large | 0.440 | **0.335** | 0.294 | **0.139** | **0.581** | **0.358** |
| NeoDictaBERT dual-enc (no hard-neg) | 0.451 | 0.288 | **0.329** | 0.093 | 0.501 | **0.332** |
| NeoDictaBERT, mean pooling | **0.456** | 0.285 | 0.327 | 0.092 | 0.484 | 0.329 |
| mE5-base | 0.361 | 0.241 | 0.248 | 0.125 | 0.549 | 0.305 |
| NeoDictaBERT + hard-neg (bs=16) | 0.333 | 0.196 | 0.319 | 0.064 | 0.523 | 0.287 |
| HMB 20250622 + hard-neg — best HMB | 0.101 | 0.061 | 0.253 | 0.034 | 0.340 | 0.158 |
| HMB new-base ba48000 (ep0) + hard-neg | 0.005 | 0.024 | 0.237 | 0.023 | 0.258 | 0.109 |
| HMB random init (control / floor) | 0.001 | 0.004 | 0.190 | 0.002 | 0.223 | 0.084 |
| HMB HeQ (discarded) | 0.001 | 0.058 | 0.037 | 0.002 | 0.012 | 0.022 |

Bold = best per column. Full 24-model table and all metrics: `results.md`.

**Reading the table:** the random-init control scores 0.190 on nfcorpus and 0.223 on scifact without any
training, so those two columns have a high free baseline. **arguana and scidocs are the columns that actually
separate models.**

---

## Models

| Label | Model / Checkpoint | Retrieval training | Status |
|-------|-------------------|-------------------|--------|
| `intfloat_multilingual-e5-large` | HF hub | MS-MARCO + NLI + web pairs | ✅ evaluated |
| `intfloat_multilingual-e5-base` | HF hub | same | ✅ evaluated |
| `neodictabert-dualenc-beir` | `outputs/models/dual_encoder/cls_pooling/beir_hebrew/neodictabert/dicta-il_NeoDictaBERT/model` | InfoNCE on Hebrew BeIR train splits | ✅ evaluated — **recommended** |
| `neodictabert-dualenc-beir-hn` | same base, hard-negative recipe | InfoNCE + BM25 hard negatives, bs=16 | ✅ evaluated — worse, do not ship |
| `hmb-20250622-hn` | `ModernBERT-Hebrew-base_20250622_1325/ep7-ba896339-rank0` | InfoNCE + hard negatives, lr 2e-5 | ✅ evaluated — best HMB |
| `hmb-newbase-ba48000-hn` | new HMB base, ep0 checkpoint | InfoNCE + hard negatives | ⏸️ inconclusive (epoch 0) |
| `hebmodernbert-dualenc-heq` | `outputs/archive/.../heq/hebmodernbert/ModernBERT-Hebrew-base_v2/model` | InfoNCE on HeQ (Hebrew QA) | ❌ discarded — does not generalize |

24 model/config combinations have been evaluated in total; the table above lists the ones that inform a
decision. See `outputs/eval/beir_zeroshot/` for all of them.

---

## Datasets

All 5 translated BeIR corpora (`gemini-3.1-flash-lite`, prompt v20260531). Counts measured from the export.
What each task actually asks: `tasks.md`.

| Dataset | Docs | Test queries | Positives/query | Note |
|---------|-----:|-------------:|----------------:|------|
| BeIR_arguana | 8,674 | 1,401 | 1.0 | test-only; queries also appear in the corpus (self-exclusion required) |
| BeIR_fiqa | 57,600 | 648 | 2.6 | train/val/test |
| BeIR_nfcorpus | 3,633 | 323 | 38.2 | graded 0/1/2; train split has no grade-2 |
| BeIR_scidocs | 25,313 | 1,000 | 4.9 | test-only; qrels include 25 score-0 negatives per query |
| BeIR_scifact | 5,183 | 300 | 1.1 | train/test, no validation split |

**Base path:** `outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus/<dataset>/beir`

---

## How to Run / Re-run

```bash
# Eval a model across all 5 datasets (L4 partition — do NOT use H200)
sbatch --export=ALL,MODEL_PATH=<ckpt>,MODEL_LABEL=<label> scripts/model/eval/eval_beir_hmb_generic.sh

# Baselines
sbatch scripts/model/eval/eval_beir_mE5_base.sh
sbatch scripts/model/eval/eval_beir_mE5_large.sh
sbatch scripts/model/eval/eval_beir_neodictabert_dualenc.sh

# Recompute metrics for ALL stored results from cached embeddings (CPU, minutes, no GPU)
python scripts/model/eval/rescore_beir_results.py --verify

# Regenerate the results table
python src/model/eval/collect_beir_results.py
```

Training logs: `logs/slurm/train_dualenc_*.out`
Eval logs: `logs/slurm/eval_beir_*_%j.out`
Results: `outputs/eval/beir_zeroshot/{model_label}/{dataset}/results.json`
Discarded results: `outputs/eval/beir_zeroshot/_invalid/`

### Metric flags that matter

| Flag | Default | Why |
|---|---|---|
| `--exclude_self` | `auto` | Removes the query's own document on datasets with structural query-in-corpus overlap. Fires on arguana (92.4%), not fiqa (9.3%). Required for comparable arguana numbers. |
| `--ndcg_gain` | `linear` | Matches `trec_eval`/`pytrec_eval` and published BeIR. Do not change. |

Each `results.json` carries the pre-fix values under `metrics_pre_fix`, so old and new numbers can be compared
directly.

---

## Discarded Runs

Moved to `outputs/eval/beir_zeroshot/_invalid/`. Do not use.

| Directory | Why discarded |
|-----------|--------------|
| `_home_nlp_achimoa_..._ep7-ba896339-rank0/` | Raw base LM, no retrieval fine-tuning |
| `dicta-il_NeoDictaBERT/` | Zero-shot (no retrieval training), incomplete run |
| `dicta-il_dictaneobert/` | Wrong model ID — failed immediately |
| `hebmodernbert-sbert-hebnli/` | NLI-tuned SBERT — cancelled, not part of benchmark |

Two additional stale directories are **not** in `_invalid/` but should be ignored:
`intfloat_multilingual-e5-{base,large}/beir/` — duplicate arguana results saved under the wrong dataset name
by an older version of the eval script. They were skipped during the 2026-07-30 rescore and still hold
pre-fix numbers.

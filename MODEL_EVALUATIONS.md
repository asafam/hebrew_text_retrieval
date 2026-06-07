# Hebrew Retrieval Model Evaluations

Zero-shot BEIR benchmark comparing Hebrew-specific retrieval models against multilingual baselines.

## Benchmark Design

**Research question:** Can a Hebrew-specific model fine-tuned for retrieval match or beat multilingual-E5 on Hebrew IR tasks?

**2×2 comparison:**

|                       | Multilingual | Hebrew-specific |
|-----------------------|-------------|-----------------|
| **Retrieval-trained** | mE5-base, mE5-large | HebrewModernBERT dual-enc, NeoDictaBERT dual-enc |

All models evaluated on the same 5 translated Hebrew BeIR datasets (NDCG@10 primary metric).

---

## Models

| Label | Model / Checkpoint | Retrieval training | Status |
|-------|-------------------|-------------------|--------|
| `intfloat/multilingual-e5-base` | HF hub | MS-MARCO + NLI + web pairs | ✅ done |
| `intfloat/multilingual-e5-large` | HF hub | same | 🔄 running (job 16635633) |
| `hebmodernbert-dualenc-heq` | `outputs/archive/models/dual_encoder/cls_pooling/heq/hebmodernbert/ModernBERT-Hebrew-base_v2/model` | InfoNCE on HeQ (Hebrew QA) | 🔄 eval running (job 16635760) |
| `neodictabert-dualenc-heq` | `outputs/models/dual_encoder/cls_pooling/heq/neodictabert/dicta-il_NeoDictaBERT/` | InfoNCE on HeQ — **training** (job 16635800) | ⏳ training on A100 |

---

## Datasets

All 5 translated BeIR corpora (Gemini 2.0 Flash Lite, prompt v20260531):

| Dataset | Path | Docs | Queries |
|---------|------|------|---------|
| BeIR_arguana | `outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus/BeIR_arguana/beir` | 8,674 | 1,401 |
| BeIR_fiqa | same run / BeIR_fiqa/beir | — | — |
| BeIR_nfcorpus | same run / BeIR_nfcorpus/beir | 3,633 | 3,237 |
| BeIR_scidocs | same run / BeIR_scidocs/beir | — | — |
| BeIR_scifact | same run / BeIR_scifact/beir | — | — |

---

## How to Run / Re-run

```bash
# Eval scripts (L4 partition — do NOT use H200)
sbatch scripts/model/eval/eval_beir_mE5_base.sh
sbatch scripts/model/eval/eval_beir_mE5_large.sh
sbatch scripts/model/eval/eval_beir_hebmodernbert_dualenc.sh

# Once NeoDictaBERT training finishes:
sbatch scripts/model/eval/eval_beir_neodictabert_dualenc.sh  # to be created after training

# Regenerate results table
python src/model/eval/collect_beir_results.py
```

Training logs: `logs/slurm/train_dualenc_neodictabert_*.out`
Eval logs: `logs/slurm/eval_beir_*_%j.out`
Valid results: `outputs/eval/beir_zeroshot/{model_label}/{dataset}/results.json`
Discarded results: `outputs/eval/beir_zeroshot/_invalid/` (base LM, wrong IDs, cancelled runs)

---

## Results (NDCG@10)

<!-- Regenerate with: python src/model/eval/collect_beir_results.py -->

| Model | arguana | fiqa | nfcorpus | scidocs | scifact |
|-------|---------|------|----------|---------|---------|
| mE5-base | 0.2723 | 0.2882 | 0.3354 | 0.2166 | 0.5525 |
| mE5-large | 0.3271 | 0.3821 | — | — | — |
| HebrewModernBERT dual-enc | — | — | — | — | — |
| NeoDictaBERT dual-enc | — | — | — | — | — |

---

## Discarded Runs

Moved to `outputs/eval/beir_zeroshot/_invalid/`. Do not use.

| Directory | Why discarded |
|-----------|--------------|
| `_home_nlp_achimoa_..._ep7-ba896339-rank0/` | Raw base LM, no retrieval fine-tuning — NDCG@10 ~0.03–0.18 |
| `dicta-il_NeoDictaBERT/` | Zero-shot (no retrieval training), incomplete run |
| `dicta-il_dictaneobert/` | Wrong model ID — failed immediately |
| `hebmodernbert-sbert-hebnli/` | NLI-tuned SBERT — cancelled, not part of benchmark |

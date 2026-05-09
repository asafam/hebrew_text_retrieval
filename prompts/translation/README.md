# Translation Prompts

All prompts translate or evaluate English→Hebrew translations of BeIR datasets for information retrieval.

---

## Directory structure

```
openai/          API-based prompts (OpenAI, Gemini, Claude via compatible endpoints)
  archives/      Superseded versions kept for reproducibility
dicta_dictalm2_0/         DictaLM 2.0 base model prompts
dicta_dictalm2_0_instruct/ DictaLM 2.0 instruct-tuned prompts
```

---

## Translation prompts (`openai/`)

All translation prompts share the same 4 few-shot examples (weight loss, Python, pasta, gaming laptops) for consistency across versions.

| File | Date | Strategy | Notes |
|---|---|---|---|
| `translation_prompts_fewshot_v20250220.yaml` | 2025-02-20 | Few-shot, with document context | Main production prompt. Query translated in context of its relevant document to resolve ambiguity. |
| `translation_prompts_fewshot_searchopt_v20250220.yaml` | 2025-02-20 | Few-shot, search-optimised | Instructs model to use Hebrew terms as a native speaker would search; keyword retention guidance. |
| `translation_prompts_fewshot_nocontext_v20250220.yaml` | 2025-02-20 | Few-shot, no context | Baseline: query translated without document context. |
| `translation_prompts_fewshot_unified_v20250220.yaml` | 2025-02-20 | Few-shot, unified | Single call returns both document and query translation (`UnifiedTranslation` response format). |
| `translation_prompts_fewshot_unified_singlesent_v20250220.yaml` | 2025-02-20 | Few-shot, unified, delimited | Unified output with `###` separator (`UnifiedSingleSentenceTranslation` response format). |
| `translation_prompts_fewshot_examples1st_v20250220.yaml` | 2025-02-20 | Few-shot, examples-first | Examples appear before the instruction; tests ordering effect. |
| `translation_prompts_zeroshot_v20250220.yaml` | 2025-02-20 | Zero-shot, with context | No examples. |
| `translation_prompts_zeroshot_searchopt_v20250220.yaml` | 2025-02-20 | Zero-shot, search-optimised | |
| `translation_prompts_zeroshot_nocontext_v20250220.yaml` | 2025-02-20 | Zero-shot, no context | |
| `translation_prompts_zeroshot_unified_v20250220.yaml` | 2025-02-20 | Zero-shot, unified | |
| `translation_prompts_zeroshot_unified_singlesent_v20250220.yaml` | 2025-02-20 | Zero-shot, unified, delimited | |

**Archived (superseded by v20250220):** `archives/translation_prompts_*_v20250128.yaml`, `archives/translation_prompts_*_v20250105.yaml`

---

## Evaluation prompts (`openai/`) — LLM-as-a-judge

All evaluation prompts use a **0–5 additive rubric** and return a `TranslationCritique(critique: str, score: int)` structured response.

### General-purpose

| File | Date | With gold? | Notes |
|---|---|---|---|
| `translation_evaluation_v20250304.yaml` | 2025-03-04 | Yes | Requires a gold reference translation. |
| `translation_evaluation_nogold_v20250304.yaml` | 2025-03-04 | No | First no-gold version; basic rubric. |
| `translation_evaluation_nogold_v20250323.yaml` | 2025-03-23 | No | Added step-by-step reasoning; clearer 0-point condition. **Superseded.** |
| `translation_evaluation_nogold_v20250406.yaml` | 2025-04-06 | No | **Current general prompt.** Sharpened score 2/3/4 definitions; IR keyword preservation added as an explicit criterion (score 4). Hebrew-specific system context (niqqud norms, transliteration). |

### Category-specific (introduced 2025-04-06)

Use these when evaluating translations from specific BeIR dataset categories to get more sensitive scoring.

| File | Category | Key criterion |
|---|---|---|
| `translation_evaluation_nogold_technical_v20250406.yaml` | Bio-medical IR (`trec-covid`, `nfcorpus`), Citation (`scidocs`), `scifact` | Score 3 = technical term handling (translated vs. transliterated vs. kept in English per Israeli scientific convention). Score 4 = IR keyword precision for a professional researcher. |
| `translation_evaluation_nogold_qa_v20250406.yaml` | Question answering (`nq`, `hotpotqa`), Argument retrieval (`arguana`), Duplicate questions (`quora`) | Score 3 = logical/rhetorical connectives preserved. Score 4 = natural Hebrew question phrasing (not a calque of English word order). Keyword matching de-emphasised. |

**Category → prompt mapping used in `run_eval_translation_api_pipeline.sh`:**

| BeIR category | Prompt |
|---|---|
| Misc, Fact checking (fever / climate-fever), Entity retrieval | general |
| scifact, Citation-Prediction, Bio-medical IR | technical |
| Question answering, Argument retrieval, Duplicate question retrieval | QA |

---

## DictaLM prompts

Local-model prompts for `dicta-lm/dictalm2.0` (base) and `dicta-lm/dictalm2.0-instruct`. Written in Hebrew. Use with `src/translation/model/` pipeline, not the API pipeline. Last updated 2025-01-05; not actively maintained.

---

## Response formats (`src/translation/api/utils.py`)

| Class | Used by |
|---|---|
| `Translation` | Single-text translation prompts |
| `UnifiedTranslation` | Unified prompts (separate fields for doc + query) |
| `UnifiedSingleSentenceTranslation` | Unified prompts with `###` delimiter |
| `TranslationCritique` | All evaluation prompts |

# Are the Hebrew Retrieval Failures Caused by Translation, or by the Model?

**Answer: the model.** About 63% of Hebrew failures happen in English too, and the
Hebrew-only failures show no sign of bad translation. Re-translating would buy
almost nothing; a stronger Hebrew encoder would.

Date: 2026-07-30 · Model tested: `intfloat/multilingual-e5-base` · 3,672 queries across the 5 translated datasets

---

## 1. The question

We translated 5 BeIR datasets into Hebrew and our retrieval scores are lower than
the published English ones. There are two possible reasons:

1. **Translation is to blame** — the Hebrew text is damaged, so the right answer
   can no longer be found. → *Fix: re-translate.*
2. **The model is to blame** — the Hebrew text is fine, the model is simply worse
   at Hebrew than at English. → *Fix: a better Hebrew encoder.*

These lead to completely different, expensive decisions. This document separates them.

---

## 2. The idea behind the test

When we translated each dataset, we kept the **English original next to every
Hebrew translation**. That makes a controlled experiment possible:

> Give the **same model** the **same retrieval task** twice — once in Hebrew,
> once in English — and compare, query by query.

Everything is held constant: same model, same questions, same documents, same
answer key. The *only* difference is the language of the text.

That gives us a clean rule for reading each query:

| What happens | What it means |
|---|---|
| Fails in Hebrew, **succeeds** in English | Suspicious — maybe translation broke it |
| Fails in Hebrew **and** in English | Translation is irrelevant — the query is just hard |

We used mE5-base because it is the one model in our lineup that genuinely handles
both languages, so the comparison is fair.

---

## 3. What was actually run

1. **Built an English copy** of all 5 datasets by swapping in the stored English
   fields. Identical IDs and identical answer keys — literally the same task in
   another language. *(0 records had to fall back to Hebrew, so the English copy is genuinely English.)*
2. **Ran mE5-base over both versions** — 7 minutes on one H200 GPU.
3. **Recorded, for every query**, whether a correct document appeared in the
   top 10 in Hebrew, and whether it did in English.

Scripts: `scripts/analysis/build_english_beir.py`, `per_query_lang_compare.py`,
`attribute_failures.py`.

---

## 4. Headline numbers

Across 3,672 queries:

- **Hebrew found the right answer 54% of the time.**
- **English found it 66% of the time.**
- So there is a **12-point gap** to explain.

Hebrew failed on **1,700 queries** in total. Those break down as follows.

---

## 5. Splitting the 1,700 Hebrew failures

| Group | Queries | Share | Interpretation |
|---|---:|---:|---|
| Failed in English too | **1,070** | **63%** | Not translation. Hard query or imperfect answer key. |
| Succeeded in English | 630 | 37% | Suspect — possibly translation |
| *less* the noise floor | −178 | −10% | Measurement jitter (explained below) |
| **Hebrew-specific failures** | **≈450** | **27%** | The real Hebrew-only problem |

### What the "noise floor" is

178 queries went the **opposite** way: Hebrew got them right and English got them
wrong. Nobody believes the English is broken — those are simply queries sitting on
the boundary of the top-10 cutoff, where tiny score differences flip the result
either way.

That same random jitter inflates the other direction by roughly the same amount.
So we subtract it. Without this correction the translation effect would look
almost 40% larger than it really is.

---

## 6. The decisive test

We now have ~450 genuinely Hebrew-specific failures. **Are their translations
actually worse?**

We compared them against the translations of queries Hebrew got **right**, using
four mechanical checks — length loss, untranslated Latin text left behind, share
of Hebrew characters, and whether numbers survived.

**Result: the two groups are indistinguishable.** Failed queries do not have worse
translations than successful ones.

Two comparisons looked significant at first. Both collapsed on inspection:

| Apparent signal | Why it is not real |
|---|---|
| nfcorpus: failed queries have "shorter" translations | The short ones are simply **correct**. `pineapples → אננסים`, `molasses → מולסה`. Queries Hebrew got *right* were equally short (ratios 0.47, 0.50). Hebrew is just a compact language, and this was confounded with query length. |
| arguana: gold documents differ in Latin residue | The effect points **backwards** — the failing documents were *more* fully translated into Hebrew, not less. |

Supporting evidence: the corpora are mechanically clean overall — under 2%
truncation, zero empty records, no untranslated documents.

---

## 7. Per-dataset results

| Dataset | Hebrew hit@10 | English hit@10 | Fail in both languages | Translation-defect evidence |
|---|---:|---:|---:|---|
| scifact | 71.3% | 82.0% | 52% | none |
| arguana | 59.8% | 69.6% | 60% | points the wrong way |
| nfcorpus | 58.8% | 69.0% | 68% | spurious (see §6) |
| fiqa | 48.3% | 66.4% | 60% | none — biggest gap, zero defect evidence |
| scidocs | 41.7% | 55.0% | 68% | none |

Note **fiqa**: it has the largest Hebrew–English gap of all five, and the *least*
evidence that translation caused it. That is the pattern the whole analysis points to.

---

## 8. Conclusion and recommendation

1. **The translations are sound.** No mechanical defect distinguishes failing
   queries from succeeding ones.
2. **Most Hebrew failures are not Hebrew's fault at all** — 63% fail in English
   too, meaning the query is intrinsically hard or the answer key is imperfect.
3. **The remaining ~27% reflects the model being weaker in Hebrew than English**,
   not damaged text.

**Recommendation:** invest in the Hebrew encoder, not in re-translation. This also
supports continuing with the remaining 10 datasets using the current pipeline —
translation quality is not what is limiting the scores.

---

## 9. Limitations — read before acting on this

1. **The checks are mechanical, not semantic.** They measure length, script and
   numbers. They cannot detect a translation that is *correct but ambiguous*.
   The clearest example in our data:

   > `suppositories` → `נרות` — a correct translation, but נרות also means
   > "candles." English rank 1, Hebrew rank 45.

   Nothing we measured would flag this, because nothing is *wrong* with the
   translation. We cannot say how much of the ~450 is this effect. **An LLM
   reading those 450 pairs side by side would close this gap** — the obvious next
   step if a firmer number is needed.

2. **One model only.** All of this is mE5-base. A Hebrew-specialised encoder
   (e.g. NeoDictaBERT) could show a different profile; "weaker in Hebrew" is a
   statement about a multilingual model.

3. **A side-analysis was inconclusive.** We also tested Hebrew queries against
   English documents and vice versa, to see whether the query side or document
   side carries the loss. mE5's cross-language matching is too weak for this to
   resolve anything (both mixed conditions scored ~39%, far below either
   same-language condition). The only usable finding is that the two directions
   are **symmetric** — no evidence Hebrew documents are more damaged than Hebrew
   queries.

4. **English is an upper bound, not a ceiling to aim at.** Part of the 12-point
   gap is mE5 simply being trained on far more English than Hebrew. That portion
   is unreachable by any amount of translation work.

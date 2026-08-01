# Are the Hebrew Retrieval Failures Caused by Translation, or by the Model?

**Answer: the model.** About 63% of Hebrew failures happen in English too, and the
Hebrew-only failures show no sign of bad translation. Re-translating would buy
almost nothing; a stronger Hebrew encoder would.

> **Confirmed by LLM judge (§10).** A blind, controlled judge run over 1,254 items found
> translation quality **identical** between failures and controls (53.5% vs 54.6%,
> p=0.70) — but did isolate one real defect worth ~2% of failures: the same English term
> rendered differently in the query and the document (`ECMO` → `אקמו` vs `ECMO`), because
> the two were translated in separate passes at temperature 0.7. The fix is a caching and
> temperature change that **costs nothing and reduces spend** — worth applying before the
> remaining 10 datasets. The main recommendation is unchanged.

> **Looking for the argument rather than the method?** See
> [why-not-translation.md](why-not-translation.md), which assembles the case and the
> resulting recommendation. This document is the underlying method, statistics and
> per-dataset detail.

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

---

## 10. LLM judge follow-up (2026-07-30) — closing the semantic gap

Limitation 1 above said the mechanical checks cannot see semantics, and that an LLM
reading the failures would close the gap. That was done.

**Method.** `gemini-3.1-pro-preview` judged 1,254 items: 626 Hebrew-only failures and
**628 blind controls** — queries Hebrew answered correctly, drawn from the same
datasets, shuffled together. The judge saw neither the group label nor any retrieval
outcome. Judging failures alone would have been uninterpretable: translation noise
exists everywhere, so only an *elevated* rate implicates translation.

Scripts: `scripts/analysis/llm_judge_failures.py`, `analyze_judge_verdicts.py`.
Raw verdicts: `outputs/analysis/judge/verdicts.jsonl`.

### The headline conclusion survives

| Measure | Failures | Controls | Difference |
|---|---:|---:|---|
| **Any translation fault** | 53.5% | 54.6% | **−1.1 pts (p=0.70)** |
| Query translation not faithful | 24.0% | 24.5% | −0.6 (p=0.82) |
| Document translation not faithful | 40.3% | 42.4% | −2.1 (p=0.45) |

Overall translation quality is **identical** in the two groups. The semantic judge
confirms what the mechanical signals found: most Hebrew failures are not caused by bad
translation. Per dataset, the fault rate difference is non-significant everywhere
(arguana −0.9, fiqa −4.4, nfcorpus −4.7, scidocs −0.6, scifact +9.8; all p>0.35).

### But it found one real, specific defect the mechanical checks missed

| Measure | Failures | Controls | Difference |
|---|---:|---:|---|
| **Judge-rated HIGH retrieval risk** | **11.3%** | **5.6%** | **+5.8 pts, RR 2.04, p=2.4e-4** |
| Key term lost | 9.3% | 5.7% | +3.5 (p=0.018) |

Failures are **twice as likely** to carry a translation issue the judge considers
likely to break retrieval, even though overall faithfulness is unchanged. This
survives Bonferroni correction across the ~10 metrics tested (threshold 0.005); the
weaker `key_term_lost` signal (p=0.018) does not.

The excess is **~5.8 pts ≈ 36 of 626 judged failures**, i.e. roughly **6% of
Hebrew-only failures**, or ~2% of all Hebrew failures. Concentrated in the technical
datasets — fiqa +11.9, scifact +9.8, scidocs +7.3, nfcorpus +7.0 — and **entirely
absent from arguana (+0.0)**, which is prose rather than terminology.

### The mechanism: queries and documents were translated independently

Reading the judge's notes on the high-risk cases, one cause dominates. It is not
mistranslation — it is the *same English term rendered differently* in the query and in
the document, which destroys the lexical overlap retrieval depends on:

| English term | In the query | In the document |
|---|---|---|
| ECMO | `אקמו` (transliterated) | `ECMO` (left in English) |
| short (sale) | `שורט` (loanword) | `מכירה בחסר` (formal term) |
| margin account | `חשבון מרווח` (literal) | `חשבון ביטחונות` (financial term) |
| evolvability | `יכולת התפתחות` | `יכולת אבולוציונית` |
| sketch | `שרבוט` (scribble) | `שרטוט` (drafting) |
| debit / credit | `חיוב` / `זיכוי` (banking) | `חובה` / `זכות` (accounting) |
| New START | `ניו סטארט` (transliterated) | `New START` (left in English) |

Each rendering is defensible in isolation, so no faithfulness check flags it — which is
exactly why both the mechanical signals and the per-text quality ratings missed it.
Mentioned in 18.7% of failure notes vs 13.4% of control notes (+5.3 pts, p=0.010), and
in 33 of the 71 high-risk failures.

**Root cause:** queries and documents were translated in **separate passes at
temperature 0.7**, with no shared glossary and no context. Nothing tied the two
renderings of a term together. This is the same mechanism found earlier in arguana,
where the identical English argument produced different Hebrew as a query and as a
document (only 125 of 954 duplicated pairs were byte-identical).

### An expected finding that at first looks like a problem

`pair_relevance` is judged from the **English** text, so it should not differ between
groups — yet it does: 37.9% of failures have a loose or unrelated answer key vs 29.1%
of controls (p=1.1e-3). This is not a leak. A marginal query–document pair is a fragile
match: it clears the top-10 bar in the model's stronger language and falls below it in
the weaker one. Fragile items flipping first in Hebrew is exactly the signature of a
model-capability gap, and it independently corroborates the main conclusion. It also
means **a substantial share of the "failures" are answer-key noise rather than failures
at all.**

### Revised bottom line

| Cause | Share of Hebrew failures |
|---|---|
| Fails in English too — hard query or noisy qrels | **63%** |
| Model weaker in Hebrew (semantics fine, no defect found) | **~31%** |
| **Translation defect — terminology drift between query and document** | **~2%** |

**The recommendation does not change: invest in the encoder, not re-translation.** A
full re-translation would address ~2% of failures.

**But one cheap, targeted fix is now justified** for the remaining 10 datasets — and it
is a pipeline change, not a quality change:

1. **Translate each unique source string once and cache by text hash.** Queries and
   documents share terminology and often whole strings; today the same English produces
   different Hebrew in each pass. This also fixes the arguana near-duplicate problem and
   *reduces* cost.
2. **Lower the temperature** from 0.7 for translation. There is no benefit to sampling
   diversity here, and it is what makes the two passes diverge.
3. **Pin a glossary** of domain terms and acronyms per dataset, especially for fiqa,
   scidocs and scifact where the effect concentrates. Prefer keeping established English
   acronyms (ECMO) over transliterating them.

Expected gain is small in absolute terms (~2% of failures), but items 1 and 2 cost
nothing and reduce spend, so they are worth applying before the msmarco run.

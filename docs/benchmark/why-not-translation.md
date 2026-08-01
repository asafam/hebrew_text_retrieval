# Why Translation Is Not the Cause of Our Lower Hebrew Scores

Our Hebrew BeIR scores are lower than the published English ones. The obvious suspect is
the translation. This document shows it isn't.

**We took all 1,700 queries the model failed on and sorted them by cause:**

| Cause | Queries | Share |
|---|---:|---:|
| Fails on the **English original** too | 1,070 | 63% |
| Fails only in Hebrew — but **another model retrieves it from the same Hebrew text** | 489 | 29% |
| Fails for **every** model we tried | 141 | 8% |
| **Total** | **1,700** | **100%** |

The first two rows — **92% of all failures** — cannot be caused by translation. The third
row is genuinely open.

**Recommendation: invest in the Hebrew encoder, not in re-translating.**

*mE5-base, 3,672 queries over the 5 translated datasets. Method:
[failure-analysis.md](failure-analysis.md).*

---

## Why this was worth measuring

Two explanations, two very different bills:

- **The Hebrew text is damaged** → re-translate 5 datasets, revise the pipeline, redo QA.
- **The model is weaker in Hebrew** → better encoder, which is work already planned.

It also decides whether to keep translating the remaining 10 BeIR datasets.

## The setup

Every translated record kept its **English source**. So the same model can run the same
retrieval task twice — once in Hebrew, once in English — with identical IDs, identical
documents, identical answer keys. **The only variable is the language.**

Hebrew answered 54% of queries correctly; English 66%. Hebrew failed on **1,700**. Those
1,700 are what the table above sorts.

---

## Row 1 — 1,070 failures aren't about Hebrew at all (63%)

**The test:** did the query also fail on the *English original*?

The English was never translated by anyone — it's the source text. If a query fails there
too, the Hebrew cannot be the reason.

**1,070 of 1,700 fail in both languages.** These are hard queries, or queries whose answer
key is wrong. They would fail in any language.

That leaves **630** failures where English succeeded and Hebrew didn't. Those are the
suspects, and the obvious reading is "the translation broke them."

## Row 2 — 489 of those suspects are retrievable from the same Hebrew (29%)

**The test:** hand the *identical Hebrew text* to a different model. If the Hebrew were
damaged — wrong words, missing terms, altered meaning — no model could retrieve it.

We tried five: NeoDictaBERT, mE5-large, and three others.

| Model | Retrieves, from the same Hebrew |
|---|---:|
| mE5-large | 321 |
| NeoDictaBERT | 313 |
| NeoDictaBERT (mean pooling) | 314 |
| NeoDictaBERT + hard negatives | 262 |
| HebrewModernBERT | 108 |
| **At least one of them** | **489 of 630 (78%)** |

Note the models rescue **different** queries — NeoDictaBERT and mE5-large agree on only
203, with 110 and 118 unique to each. That's why the union reaches 78% while no single
model exceeds 51%. Failures are model-specific, not text-specific.

**489 documents that mE5-base could not find are sitting in that Hebrew corpus, findable.**
The text was never the obstacle for them.

## Row 3 — 141 that no model retrieves (8%)

For these, every model we tried failed on the Hebrew while English succeeded. We do not
have a demonstrated cause.

Three candidates, and we cannot cleanly separate them with what we have:

1. **A weak answer key.** The labelled "correct" document may not really answer the query.
   English's stronger representations can bridge a weak link that no Hebrew-side model can.
2. **A real translation defect.** One is confirmed and quantified below.
3. **Genuinely hard queries** that happen to sit just beyond every Hebrew model's reach.

**8% of failures, cause undetermined.** Stated as such rather than assigned.

---

## The one confirmed translation defect

Separately from the rows above, we found a specific, real defect: the **same English term
rendered differently in the query and in the document**, which breaks the word-match
retrieval depends on.

| English | In the query | In the document |
|---|---|---|
| ECMO | `אקמו` | `ECMO` |
| short sale | `שורט` | `מכירה בחסר` |
| evolvability | `יכולת התפתחות` | `יכולת אבולוציונית` |
| debit / credit | `חיוב` / `זיכוי` | `חובה` / `זכות` |

Each rendering is correct on its own. The damage exists only in the *mismatch*, so no
per-text quality check can see it.

**Cause:** queries and documents were translated in separate runs, at temperature 0.7,
through two prompts differing by one word (`...English query` vs `...English document`).
Nothing kept them in agreement. Tested directly: **38% of texts got different Hebrew from
those two words alone.**

**Fixed** — one prompt for both, temperature 0.0, shared translation cache.
Query/document agreement went from 47% to 100% with no quality change. Settings:
[../translation/ledger.md](../translation/ledger.md).

This affects roughly 36 queries — about 2% of all failures — concentrated in the hard core
of row 3.

---

## What to do

1. **Don't re-translate.** 92% of failures are provably not translation, and the one real
   defect was two config settings, now changed.
2. **Keep translating the remaining 10 datasets.** Quality was never the limiter.
3. **Put the effort into the Hebrew encoder.** Row 2 is the proof it pays: swapping
   mE5-base for a better model recovers 489 queries from text that hasn't changed.

## What this doesn't prove

1. **Row 3 is unexplained.** 141 queries, cause undetermined. If a large share turned out
   to be translation, the 2% figure would rise — but it would still be under 10%.
2. **Five models, all imperfect.** "Another model retrieves it" proves the text is usable.
   It does not prove the text is *ideal*.
3. **One base measurement.** The 63% / 54% / 66% figures are mE5-base. A different
   reference model would shift them.

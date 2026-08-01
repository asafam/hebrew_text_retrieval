# Why Translation Is Not the Cause of Our Lower Hebrew Scores

Our Hebrew BeIR scores trail the published English ones. The obvious suspect is the
translation. It isn't — and this document is the evidence.

| Cause of Hebrew retrieval failure | Share |
|---|---:|
| Fails on the English original too — hard query or noisy answer key | **63%** |
| Model is weaker in Hebrew; no translation problem detectable | **~31%** |
| Actual translation defect | **~2%** |

**Recommendation: invest in the Hebrew encoder, not in re-translating.**

*Measured on the 5 translated datasets, 3,672 queries, using mE5-base. Method detail:
[failure-analysis.md](failure-analysis.md).*

---

## Why it mattered

Two explanations, two very different bills:

- **Damaged Hebrew text** → re-translate 5 datasets, revise the pipeline, redo QA.
- **A model weaker in Hebrew** → better encoder, which is work already planned.

It also decided whether to keep translating the remaining 10 BeIR datasets.

## How we tested it

Every translated record kept its English source. So the same model can run the same
retrieval task twice — once in Hebrew, once in English — with identical IDs, documents
and answer keys. **The only variable is the language.**

Each query then lands in one of four cells:

|  | Succeeds in English | Fails in English |
|---|---|---|
| **Succeeds in Hebrew** | fine | *noise* (see below) |
| **Fails in Hebrew** | **suspect** | **not translation** |

---

## Finding 1 — Two thirds of the failures aren't about Hebrew at all

Hebrew answered 54% of queries; English 66%. Hebrew failed on **1,700**:

| | Queries | Share |
|---|---:|---:|
| Failed in English too | **1,070** | **63%** |
| Succeeded in English | 630 | 37% |
| *minus* measurement noise | −178 | −10% |
| **Genuinely Hebrew-specific** | **~450** | **27%** |

**The noise subtraction matters.** 178 queries went the *other* way — Hebrew right,
English wrong. Nobody thinks the English is broken; those sit on the top-10 boundary
where trivial score differences flip the result. The same jitter inflates the other
direction equally. Skipping this control would overstate the translation effect by
nearly 40%.

So 63% of the problem is intrinsic query difficulty and imperfect answer keys — it
would appear in any language.

## Finding 2 — The remaining failures show no mechanical defect

For the ~450, we compared their translations against those of queries Hebrew answered
**correctly**, on four model-free signals: length loss, untranslated Latin text, share
of Hebrew characters, and survival of numbers — on both the query and its gold document.

**Indistinguishable.** Two comparisons looked significant and both collapsed:

- *nfcorpus failures have "shorter" translations* — the short ones are simply correct
  (`pineapples → אננסים`). Successful queries were equally short. Hebrew is compact;
  the signal was confounded with query length.
- *arguana gold documents differ in Latin residue* — the effect ran **backwards**;
  failing documents were *more* fully Hebraized.

The corpora are also clean overall: under 2% truncation, zero empty records.

## Finding 3 — And no semantic defect either

Mechanical checks can't see a translation that is faithful but ambiguous:

> `suppositories` → `נרות` — correct. But נרות also means **"candles."**
> English rank 1, Hebrew rank 45.

So we asked an LLM judge (`gemini-3.1-pro-preview`), with the control that makes the
answer mean anything: **626 Hebrew-only failures judged alongside 628 blind controls**
from queries Hebrew got right, shuffled together, group labels hidden. Judging failures
alone would prove nothing — noise exists everywhere; only an *elevated* rate implicates
translation.

| | Failures | Controls | Difference |
|---|---:|---:|---|
| **Any translation fault** | 53.5% | 54.6% | **−1.1 pts (p = 0.70)** |
| Query not faithful | 24.0% | 24.5% | −0.6 (p = 0.82) |
| Document not faithful | 40.3% | 42.4% | −2.1 (p = 0.45) |

**Identical**, and not significant in any single dataset. A capable judge reading Hebrew
against English cannot tell failures from successes on translation quality.

## Finding 4 — A corroboration we didn't plan

The judge also rated whether each pair is genuinely relevant, **judged from the English**.
That should be identical across groups. It wasn't: **37.9% of failures have a loose or
unrelated answer key vs 29.1% of controls** (p = 1.1e-3).

Not a leak — a mechanism. A marginal pair is a *fragile* match: it clears the top-10 bar
in the model's stronger language and slips below in the weaker one. Fragile items failing
first in the weaker language is exactly the signature of a **capability gap**, not damaged
text. It also means a real share of these "failures" are answer-key noise.

---

## The 2%, and why it was a setting, not a quality problem

The judge did find one real signal. Failures were twice as likely to carry an issue it
judged likely to break retrieval, despite unchanged faithfulness:

| | Failures | Controls | Difference |
|---|---:|---:|---|
| **High retrieval risk** | **11.3%** | **5.6%** | **+5.8 pts · RR 2.04 · p = 2.4e-4** |

Survives Bonferroni across the ~10 metrics tested. That is **~36 of 626 judged failures,
roughly 2% of all Hebrew failures.**

The cause was not mistranslation. It was the **same English term rendered differently in
the query and in the document**, destroying the lexical overlap retrieval depends on:

| English | In the query | In the document |
|---|---|---|
| ECMO | `אקמו` | `ECMO` |
| short sale | `שורט` | `מכירה בחסר` |
| evolvability | `יכולת התפתחות` | `יכולת אבולוציונית` |
| debit / credit | `חיוב` / `זיכוי` | `חובה` / `זכות` |

Each rendering is defensible alone — which is why every faithfulness check missed it. The
damage is *relational*, invisible to any measure that looks at one text at a time.

**Root cause:** queries and documents were translated in separate passes, at temperature
0.7, through two prompt variants that differed by one word (`...English query` vs
`...English document`). Nothing tied the two renderings together. Testing that word
directly: **38% of source strings got different Hebrew from those two words alone**, even
at temperature 0.

**This is now fixed in the pipeline** — one prompt for both passes, temperature 0.0, and
a shared translation cache. Query/document agreement went from 47% to 100% with no
quality change (p = 0.79).

**What that means for the numbers above:** the 2% is accurate for the 5 datasets we have
— they were translated with the old settings and still carry the defect; nothing was
re-translated. Datasets translated from now on should not have it. Whether that actually
drops the 2% to zero is unmeasured, because no dataset has been translated with the new
settings yet. The other two rows (63%, ~31%) are properties of the task and the model and
are unaffected by any of this.

Settings and how to re-measure: [../translation/ledger.md](../translation/ledger.md).

---

## What to do

1. **Don't re-translate.** It addresses ~2% of failures, and that 2% was two config
   settings, now changed.
2. **Do continue** with the remaining 10 datasets — translation quality was never the
   limiter.
3. **Put the effort into the Hebrew encoder.** That's where the ~31% lives.

## Caveats

1. **One model.** All of this is mE5-base. "Weaker in Hebrew" is a claim about a
   multilingual model; a Hebrew-specialised encoder could distribute errors differently.
2. **English is an upper bound, not a target.** Part of the 12-point gap is mE5 simply
   having seen far more English than Hebrew. No translation work reaches it.
3. **The judge is a model.** Run blind with matched controls, which guards the main
   biases — but it isn't human adjudication.
4. **A cross-lingual probe was inconclusive.** Hebrew queries against English documents
   and vice versa was meant to isolate which side carries the loss. mE5's cross-language
   matching is too weak to resolve it (both mixed conditions ≈39%). The only usable read:
   the two directions are **symmetric**, so no evidence Hebrew documents are more damaged
   than Hebrew queries.

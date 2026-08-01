# Why Translation Is Not the Cause of Our Lower Hebrew Scores

**The case, assembled.** Our Hebrew BeIR scores trail the published English ones. The
obvious suspect is the translation. This document sets out the evidence that it is not,
and the one narrow place where it *is*.

For method detail, scripts and raw numbers, see
[failure-analysis.md](failure-analysis.md). This file is the argument; that one is the
lab notebook.

**Bottom line:** ~63% of Hebrew failures also fail on the original English, another ~31%
show no detectable translation problem under either mechanical or semantic inspection,
and ~2% trace to a real but narrow defect that is a pipeline setting, not translation
quality. **Spend on the encoder, not on re-translating.**

> **⚠️ All figures here are PRE-FIX.** They were measured on the Run A corpus
> (`..._promptv20260531`): split query/document prompts, temperature 0.7, no
> translation cache. The two settings responsible for the ~2% have since been fixed
> for Run B — see [§ Where translation *is* at fault](#where-translation-is-at-fault-2-pre-fix-and-it-is-a-setting-not-a-quality-problem)
> and [../translation/ledger.md](../translation/ledger.md). **The ~2% is therefore an
> upper bound on what remains**, and is expected to fall toward zero in Run B. It has
> not been re-measured, because no Run B corpus exists yet. Re-run this analysis on
> Run B's first completed dataset — see [§ Revisit after Run B](#revisit-after-run-b).

---

## The question

Two explanations, two very different bills:

| If the cause is… | Then the fix is… | Cost |
|---|---|---|
| **Damaged Hebrew text** | Re-translate 5 datasets, revise the pipeline, redo QA | Weeks + API spend |
| **A model weaker in Hebrew** | Better Hebrew encoder | The work already planned |

It also decides whether to continue translating the remaining 10 BeIR datasets on the
current pipeline. So it is worth answering properly rather than by intuition.

---

## The experimental design

Every translated record kept its **English source** next to the Hebrew. That allows a
genuinely controlled comparison:

> Give the **same model** the **same retrieval task** twice — once in Hebrew, once in
> English. Identical IDs, identical answer keys, identical documents. The only variable
> is the language of the text.

Model: `intfloat/multilingual-e5-base`, the one model in our lineup that handles both
languages competently. 3,672 queries across the 5 translated datasets.

Each query then falls into one of four cells:

|  | Succeeds in English | Fails in English |
|---|---|---|
| **Succeeds in Hebrew** | concordant success | *He-only win* (noise) |
| **Fails in Hebrew** | **suspect** — maybe translation | **fails in both** — not translation |

---

## Evidence 1 — Most Hebrew failures are not Hebrew's fault

Hebrew found the answer for 54% of queries; English for 66%. Hebrew failed on **1,700**
queries. Where those land:

| Group | Queries | Share | What it means |
|---|---:|---:|---|
| Fails in English too | **1,070** | **63%** | Cannot be translation — the English original fails identically |
| Succeeds in English | 630 | 37% | Suspect |
| *minus* noise floor | −178 | −10% | See below |
| **Genuinely Hebrew-specific** | **~450** | **27%** | The real question |

**The noise floor matters.** 178 queries went the *other* way — Hebrew succeeded where
English failed. Nobody argues the English is broken; those are items sitting on the
top-10 boundary where trivial score differences flip the outcome. The same jitter
inflates the other direction equally, so it must be subtracted. Skipping this control
would overstate the translation effect by nearly 40%.

**Nearly two thirds of the problem is therefore not about Hebrew at all** — it is
intrinsic query difficulty and imperfect answer keys, and it would show up in any
language.

---

## Evidence 2 — The remaining failures show no mechanical defect

For the ~450 Hebrew-specific failures we compared their translations against those of
queries Hebrew answered **correctly**, on four model-free signals: length loss,
untranslated Latin text left behind, share of Hebrew characters, and survival of
numbers. Both the query and its gold document were checked.

**The two groups are indistinguishable.** Two comparisons initially looked significant;
both collapsed under inspection:

| Apparent signal | Why it is not real |
|---|---|
| nfcorpus failures have "shorter" translations | The short ones are simply **correct** — `pineapples → אננסים`, `molasses → מולסה`. Queries Hebrew got *right* were equally short (ratios 0.47, 0.50). Hebrew is a compact language, and the signal was confounded with query length. |
| arguana gold documents differ in Latin residue | The effect runs **backwards** — failing documents were *more* fully Hebraized, not less. |

Supporting context: the corpora are mechanically clean overall — under 2% truncation,
zero empty records, no untranslated documents.

---

## Evidence 3 — And no semantic defect either

Mechanical checks have a real blind spot: a translation can be perfectly faithful and
still break retrieval by introducing ambiguity. The clearest case in our own data:

> `suppositories` → `נרות` — a correct translation. But נרות also means **"candles."**
> English rank 1, Hebrew rank 45.

Nothing measurable is *wrong* there. So we put it to an LLM judge
(`gemini-3.1-pro-preview`) — with the control that makes the answer interpretable:

**626 Hebrew-only failures were judged alongside 628 blind controls** drawn from queries
Hebrew answered correctly, shuffled together. The judge saw neither the group label nor
any retrieval outcome. Judging failures alone would have proved nothing: some
translation noise exists everywhere, so only an *elevated* rate implicates translation.

| Measure | Failures | Controls | Difference |
|---|---:|---:|---|
| **Any translation fault** | 53.5% | 54.6% | **−1.1 pts (p = 0.70)** |
| Query not faithful | 24.0% | 24.5% | −0.6 (p = 0.82) |
| Document not faithful | 40.3% | 42.4% | −2.1 (p = 0.45) |

**Identical.** Not significant in any of the five datasets individually. A capable
semantic judge, reading the Hebrew against the English, cannot tell the failures from
the successes on translation quality.

---

## Evidence 4 — An independent corroboration we did not plan

The judge also rated whether each query–document pair is genuinely relevant, **judged
from the English text**. That should be identical between groups, since both read the
same English. It was not:

| | Failures | Controls |
|---|---:|---:|
| Answer key loose or unrelated | **37.9%** | 29.1% (p = 1.1e-3) |

This is not a leak — it is a mechanism. A marginal query–document pair is a *fragile*
match: it clears the top-10 bar in the model's stronger language and slips below it in
the weaker one. **Fragile items failing first in the weaker language is precisely the
signature of a capability gap**, not of damaged text. It also means a meaningful share
of these "failures" are answer-key noise rather than failures at all.

---

## Where translation *is* at fault: ~2% (pre-fix), and it is a setting, not a quality problem

The judge did find one real signal. Failures are twice as likely to carry an issue it
considers likely to break retrieval, even though overall faithfulness is unchanged:

| Measure | Failures | Controls | Difference |
|---|---:|---:|---|
| **High retrieval risk** | **11.3%** | **5.6%** | **+5.8 pts · RR 2.04 · p = 2.4e-4** |

That survives Bonferroni correction across the ~10 metrics tested. It amounts to **~36
of 626 judged failures — roughly 2% of all Hebrew failures, on the Run A corpus.**

Reading the judge's notes, one cause dominates, and it is not mistranslation. It is the
**same English term rendered differently in the query and in the document**, which
destroys the lexical overlap retrieval depends on:

| English term | In the query | In the document |
|---|---|---|
| ECMO | `אקמו` (transliterated) | `ECMO` (left in English) |
| short sale | `שורט` (loanword) | `מכירה בחסר` (formal term) |
| margin account | `חשבון מרווח` (literal) | `חשבון ביטחונות` (financial term) |
| evolvability | `יכולת התפתחות` | `יכולת אבולוציונית` |
| sketch | `שרבוט` (scribble) | `שרטוט` (drafting) |
| debit / credit | `חיוב` / `זיכוי` (banking) | `חובה` / `זכות` (accounting) |

**Each rendering is defensible in isolation** — which is exactly why every faithfulness
check missed it, ours and the judge's per-text ratings alike. The damage is *relational*,
invisible to any measure that looks at one text at a time.

**Root cause:** queries and documents were translated in **separate passes at temperature
0.7**, with no shared glossary and no context. Nothing tied the two renderings together.
The same mechanism explains the arguana near-duplicates, where identical English produced
different Hebrew as a query and as a document (only 125 of 954 pairs came out
byte-identical).

The effect concentrates exactly where terminology carries the meaning — fiqa **+11.9**
pts, scifact **+9.8**, scidocs **+7.3**, nfcorpus **+7.0** — and is **exactly zero on
arguana (+0.0)**, which is ordinary prose.

---

## Conclusion

| Cause of Hebrew failure | Share (Run A corpus, pre-fix) |
|---|---:|
| Fails in English too — hard query or noisy answer key | **63%** |
| Model weaker in Hebrew; no detectable translation problem | **~31%** |
| Translation defect — terminology drift between query and document | **~2%** ⚠️ fixed for Run B |

**Do not re-translate.** It would address about 2% of failures, and even that 2% is not a
translation-*quality* problem — it is two pipeline settings, both of which are now
changed for Run B. The first two rows are properties of the task and the model and are
unaffected by any translation work.

**Do continue with the remaining 10 datasets** on the current pipeline, with three cheap
changes applied first. The first two **reduce** cost:

1. **Cache translations by source-text hash.** Translate each unique string once so a
   term cannot be rendered two ways. Fixes the terminology drift and the arguana
   near-duplicates, and cuts spend because queries and documents share many strings.
   ✅ **Implemented 2026-08-01** — `dedup.enabled` in `config/translation/full_corpus.yaml`
   wires a shared SQLite cache into the shard ladder, which previously had no cache at
   all. Tests in `tests/test_translation_dedup.py`.
2. **Lower the temperature** from 0.7. Sampling diversity has no value in translation and
   is what makes the two passes diverge.
   ✅ **Implemented 2026-08-01** — queries, documents, titles and the repair pass all
   translate at temperature 0.0.
3. **Pin a per-dataset glossary** of domain terms and acronyms, especially for fiqa,
   scidocs and scifact. Prefer keeping established English acronyms (ECMO) over
   transliterating them.
   ⬜ **Not done.** Needs a term list per dataset and a prompt change, so it should be
   QA-gated on nfcorpus before the large tier.

One residual after (1) and (2), documented in
`src/translation/api/ladder_dedup_INTEGRATION.md`: the cache unifies a query and a
document only when one pass finalizes before the other prefills, and within a single
shard step the two are submitted in parallel. Temperature 0.0 covers that case in
practice, but the query and document prompt variants still differ by one word
(`...English query` vs `...English document`), so identical source text can still yield
slightly different Hebrew. Unifying the two prompt variants — already 98.9% identical,
with byte-identical system prompts — would close it completely and is the cheapest
remaining fix.

**Put the effort into the Hebrew encoder.** That is where the remaining ~31% lives.

---

## Revisit after Run B

Every number above describes the **Run A** corpus, translated with split
query/document prompts at temperature 0.7 with no cache. Run B changes all three.
The analysis should be repeated once Run B has a completed dataset, to replace the
pre-fix upper bound with a measured post-fix figure.

**The prediction being tested:** the ~2% terminology-drift share should fall close to
zero, because the mechanism behind it — the same English string receiving different
Hebrew as a query and as a document — is removed by construction. The 63% and ~31%
shares should be unchanged, since neither has anything to do with translation. If the
63% moves, something other than the prompt changed and the comparison is not clean.

**How to repeat it** (roughly one GPU-hour plus judge calls, all scripted):

```bash
# 1. English mirrors of the Run B export
python scripts/analysis/build_english_beir.py \
    --src_root outputs/translation/runs/<run_b_id>/corpus \
    --out_root outputs/analysis/english_mirror_runB

# 2. Per-query Hebrew vs English with the same model
sbatch scripts/analysis/run_lang_compare.sh          # edit HE_ROOT/EN_ROOT to run B

# 3. Mechanical defect signals + blind-control LLM judge
python scripts/analysis/translation_defect_signals.py --he_root <run_b>/corpus
python scripts/analysis/llm_judge_failures.py
python scripts/analysis/attribute_failures.py
python scripts/analysis/analyze_judge_verdicts.py
```

Keep the blind control group — it is what makes the judge output interpretable, and
it is also what will show whether the fix worked: a post-fix run should show the
failure and control groups converging on `retrieval_risk high`, which stood at
11.3% vs 5.6% pre-fix.

Two things to hold fixed so the comparison is clean: use **mE5-base** again, and the
same **hit@10** threshold. Changing either makes the before/after incomparable.

---

## Limitations, stated plainly

1. **One model.** All of this uses mE5-base. "Weaker in Hebrew" is a statement about a
   multilingual model; a Hebrew-specialised encoder could distribute its errors
   differently. Repeating the comparison with NeoDictaBERT would strengthen the claim.
2. **English is an upper bound, not a target.** Part of the 12-point gap is simply mE5
   having seen far more English than Hebrew in pretraining. No amount of translation work
   reaches it.
3. **The judge is itself a model.** It was run blind and with matched controls, which
   guards against the main biases, but it is not human adjudication.
4. **A cross-lingual probe was inconclusive.** Testing Hebrew queries against English
   documents and vice versa was meant to isolate whether the query or the document side
   carries the loss. mE5's cross-language matching is too weak to resolve it (both mixed
   conditions ≈39%, far below either same-language condition). The one usable read is
   that the two directions are **symmetric** — no evidence Hebrew documents are more
   damaged than Hebrew queries.

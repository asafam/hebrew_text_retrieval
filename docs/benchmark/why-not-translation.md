# Why Translation Is Not the Cause of Our Lower Hebrew Scores

Our Hebrew BeIR scores are lower than the published English ones. The obvious suspect is
the translation. This document shows it isn't.

**We took all 1,700 queries the model failed on and sorted them by cause:**

| Cause | Queries | Share |
|---|---:|---:|
| The English original fails too — nothing to do with Hebrew | 1,070 | 63% |
| Measurement noise — borderline cases that flip either way | 178 | 10% |
| The model is simply weaker in Hebrew | 416 | 24% |
| **A real translation defect** | **36** | **2%** |
| **Total** | **1,700** | |

The four counts add up to exactly 1,700. The percentages are each rounded to a whole
number, so reading down that column gives 99%.

**So: fix the encoder, not the translation.**

*mE5-base, 3,672 queries over the 5 translated datasets. Method and scripts:
[failure-analysis.md](failure-analysis.md).*

---

## Why this was worth measuring

Two explanations, two very different bills:

- **The Hebrew text is damaged** → re-translate 5 datasets, revise the pipeline, redo QA.
- **The model is weaker in Hebrew** → better encoder, which is work already planned.

It also decided whether to keep translating the remaining 10 BeIR datasets, so guessing
was not good enough.

## The trick that makes it answerable

When we translated each dataset we **kept the English source next to every Hebrew
record**. That means we can hand the *same model* the *same task* twice — once in Hebrew,
once in English — with identical IDs, identical documents, identical answer keys.

**The only thing that differs is the language.** So any difference in results is caused by
the language, and nothing else.

Now every failed query can be asked one question: *did it also fail in English?*

---

## Step 1 — Remove the failures that aren't about Hebrew

**The question:** if a query fails in Hebrew *and* in English, the Hebrew text can't be
the reason. How many are like that?

**The answer: 1,070 of 1,700 — 63%.**

These are hard queries, or queries whose answer key is wrong. They fail in any language.
Nothing about translation will fix them, and they're the single biggest chunk.

That leaves **630** failures where English succeeded and Hebrew didn't. Those are the
suspects.

## Step 2 — Remove the ones that are just measurement noise

**The question:** is a single failure meaningful? A query sitting right at the edge of the
top-10 cutoff can flip on a trivial score difference.

**How we measured it:** we counted the queries that flipped the *other* way — Hebrew got
them right and English got them wrong. Nobody thinks the English is broken, so those are
pure jitter. **There were 178 of them.**

The same jitter affects both directions about equally, so ~178 of our 630 suspects are
noise too.

**630 − 178 = 452 genuinely Hebrew-specific failures (27%).** Without this control we'd
have overstated the translation effect by nearly 40%.

> **Important: this is a headcount, not a list.** We know *how many* are noise. We have no
> way to know *which ones*. A noisy failure and a real one look identical from the outside
> — both are simply "Hebrew missed it, English found it." So the checks in Steps 3–5 had
> to run on all **630**, including the ~178 noisy ones.
>
> That matters, because noise makes the failure group look *more* like the control group
> and therefore hides real differences. Step 5 tests whether that hiding actually happened.

## Step 3 — Check whether the failures have bad translations

**Which queries:** all **630** English-recoverable failures — see the note above on why
the 452 cannot be isolated.

**The question:** are these failures actually badly translated?

**How we measured it:** we compared their translations against the translations of
queries Hebrew got *right*, on four automatic checks — text length lost, English words
left untranslated, share of Hebrew characters, and whether numbers survived.

**The answer: no difference at all.** Failing and succeeding queries have equally good
translations by every measure.

Two comparisons looked significant at first and both fell apart on inspection:

- *nfcorpus failures had "shorter" translations* — but the short ones are simply correct
  (`pineapples → אננסים`). Successful queries were equally short. Hebrew is a compact
  language; the signal was really about query length.
- *arguana documents differed in leftover English* — but backwards: the **failing** ones
  were more thoroughly translated.

## Step 4 — Check for problems a machine can't see

**The question:** those checks measure length and characters. They can't spot a
translation that is perfectly correct and still breaks search:

> `suppositories` → `נרות` — a correct translation. But נרות also means **"candles."**
> English rank 1, Hebrew rank 45.

**How we measured it:** we had an LLM read the Hebrew against the English and judge it,
on **626** of the 630 failures (4 dropped for missing gold documents). Crucially, we mixed
in **628 queries Hebrew answered correctly**, shuffled, with the labels hidden. Without
that control the result would be meaningless — some translation noise exists everywhere,
so what matters is whether failures have *more* of it.

**The answer: they don't.**

| | Failures | Controls | Difference |
|---|---:|---:|---|
| Any translation fault | 53.5% | 54.6% | **−1.1 points (p = 0.70)** |

Identical, and identical in every dataset taken separately. Even a careful semantic
reader cannot tell the failures from the successes by translation quality.

## Step 4b — Does the noise hide the answer?

**The objection:** Steps 3 and 4 ran on all 630, of which ~178 are noise. Noise makes the
two groups look alike, so "no difference" might just be dilution.

**How we tested it:** re-run the comparison on failures that *cannot* be borderline —
English found the answer at rank 1–3 while Hebrew missed it entirely, not even in the top
100. A 100-place gap is not measurement jitter.

| Failure subset | n | Translation fault | Controls | p |
|---|---:|---:|---:|---:|
| All English-recoverable | 626 | 53.5% | 54.6% | 0.70 |
| **English top-3, Hebrew nowhere** | **80** | **50.0%** | 54.6% | **0.44** |
| English rank 1, Hebrew nowhere | 42 | 59.5% | 54.6% | 0.54 |

**Still no difference.** Removing the noise did not reveal a hidden translation problem,
so the Step 3–4 conclusion is not a dilution artifact. If anything the clearest failures
have *slightly better* translations than average.

## Step 5 — One thing the judge did find, worth 2%

The judge flagged failures as twice as likely to have an issue that would specifically
break *search*, even though overall quality was the same:

| Failure subset | Likely to break retrieval | Controls | p |
|---|---:|---:|---:|
| All English-recoverable (n=626) | **11.3%** | 5.6% | 0.0002 |
| **English top-3, Hebrew nowhere (n=80)** | **25.0%** | 5.6% | **<0.00001** |
| English rank 1, Hebrew nowhere (n=42) | **33.3%** | 5.6% | **<0.00001** |

This is the mirror image of Step 4b. General translation quality is flat across every
subset, but this specific defect **concentrates sharply in the clearest failures** — a
third of the queries English nails and Hebrew misses completely carry it. That pattern is
what you would expect if the defect is genuinely causing failures rather than just
co-occurring with them.

Across all failures it accounts for **36 queries — 2%.** (The absolute count is unaffected
by the noise dilution: noise lowers the measured *rate* but does not remove affected
queries from the pile.)

The cause wasn't bad translation. It was the **same English word translated differently
in the query and in the document**, which breaks the word-match that search depends on:

| English | In the query | In the document |
|---|---|---|
| ECMO | `אקמו` | `ECMO` |
| short sale | `שורט` | `מכירה בחסר` |
| evolvability | `יכולת התפתחות` | `יכולת אבולוציונית` |
| debit / credit | `חיוב` / `זיכוי` | `חובה` / `זכות` |

Each one is a *fine* translation on its own. That's exactly why every quality check missed
it — the problem only exists when you compare the two.

**Why it happened:** queries and documents were translated in separate runs, at
temperature 0.7, using two prompts that differed by one word (`...English query` vs
`...English document`). Nothing kept the two in agreement. We tested that word directly:
**38% of texts got different Hebrew from those two words alone.**

**Now fixed** — one prompt for both, temperature 0.0, shared translation cache.
Query/document agreement went from 47% to 100%, with no loss of quality (p = 0.79).

The 2% above still describes the 5 datasets we have, since nothing was re-translated.
New translations shouldn't carry it. Settings: [../translation/ledger.md](../translation/ledger.md).

---

## What to do

1. **Don't re-translate.** It would address 2% of failures — and that 2% was two config
   settings, now changed.
2. **Keep translating the remaining 10 datasets.** Quality was never the limiter.
3. **Put the effort into the Hebrew encoder.** That's where the 416 queries are.

## What this doesn't prove

1. **One model only.** All of this is mE5-base. "Weaker in Hebrew" is a claim about a
   multilingual model; a Hebrew-specific encoder might behave differently.
2. **English is a ceiling we can't reach anyway.** Part of the gap is mE5 having seen far
   more English than Hebrew in training. No translation work touches that.
3. **The judge is itself a model.** Run blind with matched controls, which handles the
   obvious biases — but it isn't a human expert.
4. **One test was inconclusive.** Pairing Hebrew queries with English documents (and vice
   versa) was meant to show whether queries or documents carry the damage. mE5 is too weak
   at cross-language matching for the result to mean anything (both ≈39%). The only usable
   read: the two directions are symmetric, so no sign that documents are worse than
   queries.

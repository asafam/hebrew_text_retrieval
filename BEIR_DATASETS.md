# BeIR → Hebrew: Per-Dataset Retrieval Task Reference

What each translated BeIR dataset actually asks a retriever to do, what counts as relevant, and the traps specific to each one.

All statistics below were measured directly on the exported Hebrew data in
`outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus/BeIR_<name>/beir/`
(not copied from the BeIR paper). Companion docs: `BEIR_TRANSLATION.md` (translation status/ledger), `EVALUATION.md` (retrieval results).

---

## At a glance — the 5 translated datasets

| Dataset | Query is… | Document is… | Relevance means | Scale | Pos/query | Corpus | Splits |
|---|---|---|---|---|---|---:|---|
| **nfcorpus** | a 2-word health topic/video title | a PubMed abstract | the abstract is cited by, or topical to, that health claim | **graded 0/1/2** | **38.2** (median 16) | 3,633 | train/val/test |
| **fiqa** | a natural-language personal-finance question | a StackExchange/forum answer | the answer answers the question | binary | 2.6 | 57,600 | train/val/test |
| **scifact** | a one-sentence scientific claim | a paper abstract | the abstract contains evidence supporting/refuting the claim | binary | **1.1** | 5,183 | train/test |
| **arguana** | a **~140-word argument** | a counter-argument | the doc is *the designated rebuttal* to that argument | binary | **1.0** | 8,674 | test only |
| **scidocs** | a paper **title** | a paper abstract | the doc is cited by the query paper | binary + **explicit 0-pool** | 4.9 | 25,313 | test only |

The three axes that actually differ between them: **query length** (2 words → 140 words), **positives per query** (1.0 → 38.2), and **whether the qrels file contains explicit negatives**. Every one of those changes what a metric number means.

---

## 1. nfcorpus — biomedical claim → evidence abstract

**Task.** Given a short consumer-health topic in Hebrew (median **2 words**, e.g. `תאי סרטן השד ניזונים מכולסטרול` / "Breast Cancer Cells Feed on Cholesterol"), retrieve PubMed abstracts (median **193 words**, 100% have titles) that back it.

Queries originate as NutritionFacts.org video titles; documents are the studies those videos cite.

**Relevance is graded, and this is the one dataset where that matters.**

| Split | Judgments | Queries | Score 1 | Score 2 | Pos/query (mean / median) |
|---|---:|---:|---:|---:|---|
| train | 110,575 | 2,590 | 110,575 | **0** | 42.7 / 18 |
| validation | 11,385 | 324 | 10,864 | 521 | 35.1 / 18.5 |
| test | 12,334 | 323 | 11,758 | 576 | 38.2 / 16 |

- **Score 2** = the abstract is directly cited as evidence for the claim. **Score 1** = merely topically linked.
- The distribution is ~95% / 5% skewed toward the weak tier — a random qrels pair is almost certainly a *loose* topical match, not a tight one. This is what caused the sample-spreadsheet confusion documented in `BEIR_TRANSLATION.md`.
- **The train split has no grade-2 judgments at all.** Training treats "topically related" as positive; test rewards graded ranking. Don't assume train and test define relevance the same way.
- ~38 relevant docs per query out of a 3,633-doc corpus means **>1% of the entire corpus is relevant to any given query**. Recall@100 is close to meaningless here, and NDCG is the only metric worth reading.

**Hard negatives:** `hard_negatives_train.jsonl`, 110,575 rows (one per training qrel), schema `{query, positive, hard_negs}`. With 42 positives per query, negative mining on nfcorpus is unusually likely to sample a false negative — a doc that is genuinely relevant but unjudged for that query.

---

## 2. fiqa — financial question → forum answer

**Task.** Given a Hebrew personal-finance question (median **9 words**, e.g. "מה נחשב להוצאה עסקית בנסיעת עסקים?"), retrieve the forum answer that answers it. Documents are StackExchange/Reddit answer bodies, median **74 words**, and **0% have titles** — the only one of the five that is body-text-only, so any title-concatenation logic is a no-op here.

**Relevance:** binary (all judgments score 1), ~2.6 answers per question.

| Split | Judgments | Queries |
|---|---:|---:|
| train | 14,166 | 5,500 |
| validation | 1,238 | 500 |
| test | 1,706 | 648 |

This is the most "normal" retrieval task of the five — short question, medium answer, few positives, three clean splits. It's the best default for sanity-checking a model.

**Hard negatives:** `hard_negatives_train.jsonl`, 14,131 rows.

**Note, not a bug:** 621 query IDs numerically collide with corpus IDs (both are plain integers in separate namespaces). Verified: **0 of them share text.** There is no query-in-corpus leakage in fiqa.

---

## 3. scifact — scientific claim verification as retrieval

**Task.** Given a one-sentence scientific claim in Hebrew (median **11 words**, e.g. "לחומרים ביולוגיים 0-ממדיים חסרות תכונות השראתיות"), retrieve the paper abstract containing the evidence that supports or refutes it. Corpus is 5,183 abstracts, median 165 words, all titled.

**Relevance:** binary, **1.1 positives per query** — the sparsest of the five. Nearly every query has exactly one correct answer, which makes MRR and NDCG@10 nearly interchangeable.

| Split | Judgments | Queries |
|---|---:|---:|
| train | 919 | 809 |
| test | 339 | 300 |

**No validation split.** Carve one out of train if you need one.

**On the "different query-doc score" you were remembering:** scifact's qrels here are *not* graded — every judgment is score 1. What makes scifact *look* loosely coupled is different: the qrels link a claim to a whole cited **abstract ID**, never to the specific supporting sentence inside it. So a correct pair often reads as only vaguely on-topic — a 11-word claim against a 165-word abstract where one sentence does the work. That's the dataset's design, not a translation or join error. (The upstream SciFact dataset carries SUPPORT/CONTRADICT labels and sentence-level rationales; the BeIR conversion drops both.) **The retriever is not asked to judge whether the claim is true — only to find the abstract that adjudicates it.** A supporting and a refuting abstract are equally correct answers.

**Hard negatives:** `hard_negatives_train.jsonl`, 919 rows, schema `{query, positive, hard_negs}` — verified example carries 2 negatives.

---

## 4. arguana — argument → counter-argument (the odd one out)

**Task.** Given a **~140-word argument** in Hebrew, retrieve the argument that **rebuts** it. This is the only dataset here where the query is long-form — median **140 words**, p95 282 — and the only one where query and document are the *same kind of object*. It is a symmetric long-to-long matching task, not asymmetric short-query retrieval.

Corpus: 8,674 debate passages, median 117 words, only **31% have titles**.

**Relevance:** binary, exactly **1.0 positive per query** — 1,406 judgments over 1,406 distinct queries. Verified: **all 1,406 pairs follow the pattern `<debate-id>a` → `<debate-id>b`** — the ID scheme itself encodes the pairing. Test split only.

### ⚠️ Two arguana-specific issues, both verified on our export

**(a) Self-retrieval leakage.** The `a`-side arguments are themselves in the corpus: **1,294 of 1,401 query IDs also exist as corpus documents, and 954 of them share the identical English source text.** Standard BeIR practice is to **drop the query's own document from the ranked list before scoring**. `src/model/eval/eval_beir_retrieval_zeroshot.py` does not do this — there is no self-exclusion anywhere in `compute_metrics`. A good model will rank the query's own near-duplicate at position 1 and push the real counterargument to position 2, roughly halving reciprocal rank on 68% of queries. **ArguAna numbers from the current eval script are not comparable to published BeIR ArguAna scores.**

**(b) The leak is partially masked by translation nondeterminism.** Queries and documents were translated in separate passes at `temperature 0.7`, so the same English string produced *different Hebrew* on each pass — of the 954 self-present queries, only **125 have byte-identical Hebrew** in both places. Example:

```
query: להיות צמחוני עוזר לסביבה הפיכה לצמחוני היא דבר ידידותי לסביבה לעשות...
doc  : להיות צמחוני עוזר לסביבה. להפוך לצמחוני זה דבר ידידותי לסביבה לעשות...
```

So the leakage shows up as *near*-duplicates rather than exact ones. That makes it harder to spot and makes ArguAna scores partly a measure of paraphrase robustness. Worth deciding deliberately whether to re-export arguana with a shared translation cache keyed on source text.

**Minor:** 5 qrels query-ids have no record in `queries.jsonl`, and 5 qrels corpus-ids are missing from `corpus.jsonl`.

---

## 5. scidocs — citation prediction, and it ships its own negatives

**Task.** Given a paper **title** in Hebrew (median **9 words**), retrieve the abstracts of papers that the query paper **cites**. Corpus: 25,313 abstracts, median 129 words, all titled. Test split only.

**This is the dataset whose qrels file contains explicit negatives**, and it's the one that already caused a real bug:

| Score | Judgments | Per query |
|---|---:|---:|
| **0 (explicit negatives)** | 25,000 | exactly 25 |
| 1 (positives) | 4,928 | ~5 |

SciDocs was built as a **reranking** benchmark: each query comes with a fixed candidate pool of ~5 cited papers plus 25 deliberately-chosen non-cited ones. The score-0 rows are that negative pool — **they are not relevance judgments, and treating them as pairs is wrong.** This is exactly the trap that put 21 of 25 scidocs rows in the review spreadsheet on non-relevant pairs (see `BEIR_TRANSLATION.md`, "Fix applied (pass 1)").

Two consequences:

- **Anything that reads the qrels file must filter `score > 0`** before using rows as positives. The eval script's metric code happens to be safe (it treats unretrieved/0-score docs as gain 0), but any sampling, training-pair construction, or spreadsheet tooling must filter explicitly.
- **The score-0 pool is free, high-quality hard negatives** — human-curated same-field non-citations. Notably, scidocs has *no* `hard_negatives_train.jsonl` (it's test-only), yet it's the one dataset that ships curated negatives in-band. If you want scidocs negatives, read them out of the qrels rather than mining them.

**Coverage gap:** 4,928 positives instead of the expected ~5,000, and **115 positive corpus-ids are missing from the translated `corpus.jsonl`** (25,313 docs exported vs 25,657 upstream). Those queries are scored against an incomplete gold set — a small pessimistic bias in scidocs recall.

---

## Cross-cutting evaluation caveats

Status: **fixed** in `src/model/eval/eval_beir_retrieval_zeroshot.py`. The metric implementation now matches `pytrec_eval` (the reference behind published BeIR numbers) to <1e-12 on all 5 datasets × all splits.

1. **No arguana self-exclusion** (§4a) — *was a real bug, now fixed.* The query's own document was never removed from the ranked list. New `--exclude_self {auto,always,never}`, default `auto`: drops the same-id document unless qrels judge it relevant (so fiqa's coincidental id collisions are unaffected). Verified end-to-end on an arguana-shaped case: MRR 0.50 → 1.00.

2. **Truncated ideal DCG** — *was a real bug, now fixed.* `compute_metrics` built the relevance vector from only the **top-`k` retrieved** docs and let `sklearn.ndcg_score` derive the ideal from that slice. If a query has 38 relevant docs (nfcorpus) and only 4 land in the top-100, IDCG was normalized against those 4 — **inflating NDCG**. NDCG is now computed by hand with the ideal ranking taken from the full qrels. Impact was severe for nfcorpus, mild for fiqa/scidocs, negligible for scifact/arguana.

3. **Linear vs. exponential gain** — *I was wrong; there was no bug here.* I initially claimed `pytrec_eval` uses `2^rel − 1` and that sklearn's linear default therefore deviated. Cross-checking against `pytrec_eval` on graded nfcorpus qrels showed the opposite: `trec_eval`'s `ndcg_cut` uses **linear** gain (`gain = rel`), so sklearn's default already matched published BeIR. The default is `linear`; `--ndcg_gain exponential` exists as an option but is **not** BeIR-comparable.

**Bonus fix:** the metric previously reported as `recall_at_100` was actually *hit rate* — `any(relevant in top-100)`, which saturates near 1.0 on nfcorpus (38 positives/query) and carries almost no signal. It is now true recall (`|relevant ∩ top-k| / |all relevant|`), with the old figure retained as `hit_rate_at_100` so previously-saved `results.json` files remain interpretable.

Net effect on existing numbers: **nfcorpus NDCG was inflated and will drop; arguana was depressed and will rise; recall_at_100 changes meaning on every dataset.** All results in `EVALUATION.md` / `MODEL_EVALUATIONS.md` predate these fixes and need regenerating before they can be compared against published BeIR figures. scidocs remains slightly pessimistic due to its 115 missing gold documents — that's a translation-coverage gap, not a metric bug.

---

## Not yet translated (10 / 15)

From upstream BeIR — task definitions only, to be re-verified against our export when each lands. Blocked on the approval gate in `BEIR_TRANSLATION.md`.

| Dataset | Query → Document | Notes |
|---|---|---|
| msmarco | web query → passage | binary, ~1.1 pos/q; ~8.8M docs, the long pole |
| nq | natural question → Wikipedia passage | binary |
| hotpotqa | multi-hop question → Wikipedia passage | **2 gold passages per query**, both needed |
| fever | claim → Wikipedia evidence | binary |
| climate-fever | climate claim → Wikipedia evidence | binary; deliberately noisy/disputed claims |
| quora | question → duplicate question | **symmetric** short-to-short, like arguana in shape |
| trec-covid | COVID topic → CORD-19 abstract | **graded 0/1/2**, very deep pools (~493 judgments/query) |
| dbpedia-entity | entity query → DBpedia abstract | **graded 0/1/2** |
| cqadupstack | question → duplicate question | 12 StackExchange subsets, scored separately then averaged |
| webis-touche2020 | controversial question → argument | **graded**; long documents |

When these land, the two things to check first are the ones that bit us above: **does the qrels file contain score-0 rows** (scidocs pattern), and **are queries present in the corpus** (arguana pattern). trec-covid, dbpedia-entity and webis-touche2020 are graded, so caveat (3) applies to them too.

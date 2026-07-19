# BeIR → Hebrew Translation Ledger

Running ledger of translation progress for the [BeIR](https://github.com/beir-cellar/beir) benchmark → Hebrew. Update this file as datasets move through the pipeline. See `README.md` ("Experiment 4") for the full experiment writeup and `EVALUATION.md` for downstream retrieval results on the translated data.

## Pipeline

- **Orchestrator:** `src/translation/api/run_beir_ladder_pipeline.py` — resumable "shard ladder": each dataset is split into growing shards (500 rows → 100K rows depending on dataset size), translated via Gemini batch (Vertex AI/GCS), then QA-judged (`gemini-3.1-pro-preview`, gate: mean score ≥ 3.5/5) before advancing to the next shard. A dataset that fails the gate stops automatically; others continue unaffected.
- **Model / prompt (current run):** `gemini-3.1-flash-lite`, prompt `prompts/translation/api/translation/translation_prompts_zeroshot_v20260531.yaml` (adds transliteration handling for foreign/Latin terms vs. the earlier v20250220 prompt).
- **Config:** `config/translation/full_corpus.yaml` (all 15 datasets, per-dataset shard sizes).
- **Run scripts:** `bash scripts/translation/translate.sh` (`--pilot` for 100-row samples, `--dry-run` to inspect the shard plan, `--resume` to continue, `--dataset BeIR/<name>` to scope to one).
- **Export:** `python scripts/translation/build_hf_dataset.py --run-dir outputs/translation/runs/<run_id> --dataset <name>` — produces eval-ready BeIR-format JSONL (`corpus.jsonl`, `queries.jsonl`, qrels).
- **Current run directory:** `outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/` (gitignored — lives only on the compute host). Cost so far: ~$32.61.

## Prompt & settings used (current run)

**Prompt file:** `prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20260531.yaml`

Two variants (query / document), same instructions:

```
system_prompt:
  You are a precise and concise translation assistant. Your task is to translate
  sentences from English to Hebrew, providing accurate translations without
  unnecessary explanations.

user_prompt_prefix:
  Translate the following English {document|query} into Hebrew, faithfully and accurately.
  Render only what the source says — add, omit, or explain nothing (no words or
  articles not in the source; never answer or fill in blanks). Translate foreign
  and Latin terms and proper nouns into Hebrew, transliterating when there is no
  Hebrew equivalent (e.g. Dendrobium → דנדרוביום, Python → פייתון). Reproduce
  non-linguistic elements exactly: markup (e.g. >, **), citations (e.g. [1]),
  blanks (e.g. ___), URLs, code, abbreviations/identifiers (e.g. PrP, 6MP),
  punctuation (keep -- not —), and numbers as digits (1, not "one"/"אחד").

user_prompt_template:
  {english_key}: {segment_text | text}
  {hebrew_key}:
```

Zero-shot, no context passed to the model (`context_col` unused; `include_context: false` on export) — hence "nocontext" in the run/prompt name. This is the successor to prompt v20250220; v20260531's addition is the explicit transliteration instruction (fixes foreign/Latin terms being left untranslated or garbled).

**Model / generation settings** (`config/translation/full_corpus.yaml`, applies identically to queries, documents, and titles):

| Setting | Value |
|---|---|
| Model | `gemini-3.1-flash-lite` (Vertex AI batch) |
| Temperature | 0.7 |
| Max document segment length | 512 tokens (tokenized with `gpt-4o-mini-2024-07-18` for length-counting only) |
| Repair pass (failed/truncated rows) | up to 3 attempts, temperature 0.3, sentence-by-sentence re-translation, triggered when `len(Hebrew) < 0.5 × len(English)` |
| QA judge | `gemini-3.1-pro-preview`, gate: ladder score must be within 1.5σ of that dataset's pilot mean (absolute floor 2.5, fallback 3.5 if no pilot baseline) |
| Random seed | 42 (dataset sampling / shard construction) |
| Cost guardrail | $2,500 max; $0.25 / 1M input tokens, $1.50 / 1M output tokens (gemini-3.1-flash-lite batch pricing) |

Full config: `config/translation/full_corpus.yaml`. Orchestrated via `src/translation/api/run_beir_ladder_pipeline.py` / `run_beir_batch_gcs.py` (see Pipeline section above).

## Status as of 2026-07-19

### ✅ Done — full corpus translated (5 / 15)

| Dataset | Corpus | Queries | Qrels | Notes |
|---|---:|---:|---|---|
| nfcorpus | 3,633 | 3,237 | train 110,575 / val 11,385 / test 12,334 | stage 8/8 |
| fiqa | 57,600 | 6,648 | train 14,166 / val 1,238 / test 1,706 | stage 6/6, includes `hard_negatives_train.jsonl` |
| scifact | 5,183 | 1,109 | train 919 / test 339 | stage 6/6 |
| arguana | 8,674 | 1,401 | test 1,406 | stage 9/9, test-only |
| scidocs | 25,313 | 1,000 | test 29,928 | stage 6/6, test-only |

Confirmed via `progress.json` (`ladder_all_done: true` for these 5 only). Output path: `outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus/BeIR_<name>/beir/`.

### ⏳ Pending — not yet started on full corpus (10 / 15)

msmarco, nq, hotpotqa, fever, climate-fever, quora, trec-covid, dbpedia-entity, cqadupstack, webis-touche2020.

`progress.json` shows `ladder_all_done: false` (stage 0) for all ten. A 100-row **pilot** sample exists for every one of the 15 datasets under `.../pilot/BeIR_*/{documents,queries}_translated.csv` (earlier all-15 pilot run, commit `aaee44d`), scored 3.72–5.00/5 by the QA judge — so quality is validated at small scale, but no full-corpus run has been executed. `msmarco` is the long pole at ~8.8M documents.

### Known issues / repairs

- Long-document truncation required "repair" rounds (whole-document batch re-translation). E.g. scidocs shard 4 had 68 failed docs → 63 repaired, 5 unresolved; shard 5 needed 3 repair rounds to fully resolve.
- Garbled source-text ligatures (e.g. `¤`, `\x92`) in some scidocs source docs (PDF-derived) caused malformed JSON lines during batch-prediction parsing — silently skipped.
- Several correctness fixes landed along the way: NaN crashes in export joins, qrels format switch (TSV → JSONL), id-matching tolerance, ladder resume/repair crash fixes, and the v20260531 prompt update specifically to fix transliteration of foreign/Latin terms.

### Recency

Last activity on the translation pipeline itself: `outputs/` last modified 2026-06-02 (`progress.json`). No translation-pipeline commits since early/mid June 2026 — the most recent repo commit (`8f13b85`, 2026-06-29) is unrelated (eval-cache invalidation). The effort is effectively **paused** after finishing the 5 smallest datasets.

## 🛑 Milestone — pending approval before continuing

A sample spreadsheet for manual review was generated from the 5 completed datasets:

- **Content:** 5 datasets × 25 query/document sample pairs each = 125 rows, Hebrew + English side by side, for spot-checking translation quality before committing to the remaining 10 (much larger) datasets.
- **Files:** `outputs/translation/advisor_sample_pairs.xlsx` (one sheet per dataset: `nfcorpus`, `fiqa`, `scifact`, `arguana`, `scidocs`) / `.csv` (flat, with a leading `dataset` column). Originally generated 2026-06-07, regenerated 2026-07-19 (see below); gitignored, host-local only.
- **Header (per-sheet, xlsx):** `query_id, query_he, query_en, doc_id, doc_text_he, doc_text_en, doc_title_he, doc_title_en`

### Sampling methodology fix (2026-07-19)

Manual review of the first few rows (nfcorpus, scifact) raised concerns that some query→document pairs didn't look topically related. Investigation (cross-checked against the original English `BeIR/nfcorpus` / `BeIR/scifact` HF datasets) found:

- Every checked pair **is a genuine qrels entry** — not a translation or join bug. Text matched the English source exactly.
- **nfcorpus has a graded 0–2 relevance scale** (score 2 = direct citation for the claim, score 1 = only topically related), and it's heavily skewed: 11,758 score-1 vs. only 576 score-2 judgments (~95%/5%) in the test split. The original sampler picked randomly per query with no score awareness, so 24/25 nfcorpus rows landed on the weak (score-1) tier — consistent with, not worse than, the base rate.
- **scifact/arguana/fiqa are binary** (all qrels entries are score 1) — there's no "stronger" pair to prefer; SciFact's apparent looseness comes from the qrels only linking a claim to a cited abstract ID, not to the specific supporting sentence.
- **scidocs was the real bug**: its qrels file mixes real positive judgments (score 1, 5 per query) with an explicit negative candidate pool (score 0, 25 per query, used for reranking eval). **21 of the original 25 scidocs sample rows had been drawn from the score-0 (non-relevant) pool** — i.e. those "pairs" were never valid query↔document matches at all.

**Fix applied (pass 1):** for every dataset with more than one qrels grade available per query (in practice: nfcorpus for the 1-vs-2 grade, and scidocs to exclude the score-0 negative pool), the sample selects the **highest-scoring** available document per query, restricted to documents that exist in the translated corpus export. 29 of 125 rows changed (8 nfcorpus, 21 scidocs); fiqa/scifact/arguana unchanged since all their qrels are single-tier. 3 scidocs rows had a qrels-positive doc ID missing from the translated `corpus.jsonl` entirely (translation/export gap, not a sampling issue) — next-available scored doc substituted.

**Fix applied (pass 2 — tie-breaking within a score tier):** manual spot-check of `PLAIN-2590` and `PLAIN-613` after pass 1 found the *chosen* doc still didn't obviously match the query (e.g. "ascorbic acid" → a document that never mentions ascorbic acid). Root cause: **it's common for one query to have several documents at the same top score** — 90 of 323 nfcorpus test queries (28%) have 2+ score-2 documents (one has 21) — because nfcorpus queries are health claims typically backed by multiple on-topic studies, not one. Pass 1 broke these ties arbitrarily (lowest corpus ID). Pass 2 re-ranks candidates *within* the max-score tier by lexical overlap between query and candidate doc (English side), and picks the best-overlapping one. This changed 60 of 125 rows (22 nfcorpus, 17 fiqa, 5 scifact, 16 scidocs; arguana unaffected — typically 1 candidate per query). Confirmed by manual read: `PLAIN-2590` → `MED-2294` ("Comparison of Nutritional Quality of the Vegan, Vegetarian, Semi-Vegetarian..."), `PLAIN-613` → `MED-1198` ("High-dose ascorbic acid increases intercourse frequency and improves mood"). Lexical overlap is a heuristic, not a semantic judgment — it can still miss cases where the best doc uses different wording than the query.

**Fix applied (pass 3 — distinct-query coverage):** verifying fiqa/arguana/scidocs confirmed they were already correctly configured (25/25 distinct queries, spot-checked coherent matches). But pass 2's re-ranking had a side effect on nfcorpus and scifact: several queries that appeared more than once in the original (randomly-sampled, sometimes-repeated) query list now resolved to the *same* best-matching doc each time, collapsing into literal duplicate rows — nfcorpus dropped to 21/25 distinct queries, scifact to 23/25. Backfilled with additional distinct queries (same max-score + lexical-overlap selection) to restore 25/25 distinct queries in every dataset.

**Known residual limitation:** 2 of the nfcorpus backfill queries ("deafness", "Dr. Walter Willett") have *no* lexically-overlapping candidate at all — checked manually, none of their 5–9 score-1 candidates mention the query term. This is a genuine nfcorpus trait, not a pipeline bug: some queries are named-entity/single-word video titles from NutritionFacts.org, and their cited evidence is thematically/causally linked (e.g. heavy-metal-contaminated herbal products → ototoxicity/deafness) rather than lexically linked. Flagged for reviewer awareness rather than swapped out for an easier example.

**Status:** finalized with all three fixes applied (2026-07-19) — awaiting sign-off on this spreadsheet. Do not resume the ladder for the remaining 10 datasets (including the large `msmarco` run) until this sample is reviewed and approved — treat approval of this sheet as the gate for continuing.

## Next steps

- [ ] Get approval on `advisor_sample_pairs.xlsx` (blocking).
- [ ] Once approved: resume the ladder for the next-smallest pending dataset (of the ten, likely nq/quora/trec-covid-sized ones before msmarco) via `bash scripts/translation/translate.sh --dataset BeIR/<name>`.
- [ ] Budget/plan specifically for `msmarco` (~8.8M docs) — by far the largest remaining corpus and likely the dominant cost driver of the remaining 10.

---

*Update this ledger whenever a dataset's `ladder_all_done` flips to `true`, a new pilot is run, or the pipeline/prompt version changes.*

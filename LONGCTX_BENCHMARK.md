# Hebrew Long-Context Retrieval Benchmark

A "needle in a haystack" retrieval benchmark built from the translated BeIR corpora, for
measuring how Hebrew retrieval models behave as documents grow from ~1K to ~27K characters.

Companion to [EVALUATION.md](EVALUATION.md) (standard short-context retrieval).
Supersedes the HeQ-based `LONG_CONTEXT_EVALUATION.md` design — see *Why this replaces the old
benchmark* below.

---

## The task

Given a Hebrew query, retrieve the document containing the answering passage. Every document
in the corpus is a **welded** document: one real BeIR passage surrounded by filler text.

The key mechanism: **welding changes document _text_, never document _identity_.** Each
document keeps its original `_id`, so the original `queries.jsonl` and `qrels/test.jsonl` stay
valid unmodified, and results are directly comparable to the unpadded baseline.

The needle stays constant. Only the haystack around it grows.

---

## The length ladder

Length is budgeted in **characters**, never in any one model's tokens — budgeting in tokens
silently calibrates the corpus to whichever tokenizer was used (the mistake the old benchmark
made with mE5).

| rung | chars | HMB (8192) | NeoDictaBERT (4096) | mE5 (512) |
|---|---:|---|---|---|
| `c0` | raw, unpadded | native | native | native |
| `c3700` | 3,700 | native | native | sees ~27% |
| `c7400` | 7,400 | native | native | sees ~14% |
| `c11800` | 11,800 | native | **last native rung** | sees ~9% |
| `c19000` | 19,000 | native | sees ~89% | sees ~7.5% |
| `c27000` | 27,000 | **native (at ceiling)** | sees ~62% | sees ~5.3% |

`c19000` and `c27000` are the **decisive rungs**: the only ones where a 8192-token model reads
the whole document and no baseline can.

Budgets were set from measured *welded* token counts (p99 ≤ 0.95 × each model's limit), not
from per-passage ratios — a welded document averages its chars/token over many passages, so its
distribution is 5.6× tighter than the per-passage one.

### Measured tokenizer efficiency (raw Hebrew, not detokenized text)

| model | chars/token | window | text the window holds |
|---|---|---|---|
| NeoDictaBERT | 4.11 | 4,096 | ~16,800 chars |
| HMB | 4.01 | 8,192 | ~32,800 chars |
| mE5 | 2.78 | 512 | ~1,425 chars |

Note NeoDictaBERT tokenizes Hebrew ~2.5% *better* than HMB — a wash. The 44% tokenizer
advantage is over **mE5 only**, and is not a general Hebrew finding.

---

## Datasets

| dataset | corpus | test golds | safe filler pool | on disk |
|---|---:|---:|---:|---:|
| `BeIR_scifact` | 5,183 | 283 | 4,516 (87%) | 1.4 GB |
| `BeIR_scidocs` | 25,313 | 3,905 | 21,367 (84%) | 8.2 GB |
| `BeIR_fiqa` | 57,600 | 1,705 | 40,273 (70%) | 15 GB |

**Excluded, with reasons:**
- `nfcorpus` — averages 38 positives per query, so 86% of the corpus is a positive for
  something, leaving only 505 usable filler passages for 3,633 documents (~165× reuse).
- `arguana` — its queries *are* corpus documents, so a query's own text can land inside another
  document as filler and match it for free.

---

## Filler conditions

- **`random`** — uniform draw from the safe pool. Isolates pure length dilution. **Primary.**
- **`bm25`** — top-200 BM25 neighbours of the seed passage. Justified by *realism* (real long
  documents are topically coherent), **not** by difficulty: filler similar to a gold is also
  similar to the query, so a coherent document can be *easier* to retrieve.

The BM25 selector is frozen and model-independent, and its neighbour lists are written to disk
for audit. The previous design picked neighbours using *each evaluated model's own embeddings*,
which silently gave every model a different corpus.

### Leakage constraints (both enforced and verified)

1. **No passage that is a positive for any query, in any split, may be used as filler.**
   Otherwise an irrelevant document genuinely contains a query's answer while the qrels call it
   irrelevant — a manufactured false negative that punishes correct retrieval.
2. **A document may not use its own passage as filler**, by id, by exact text, or by
   containment (for passages ≥64 chars). Containment is not enforced below that length because
   a 4-character common word like `מקור` matches most Hebrew text and would starve the tapes.

---

## Construction

Per document, per rung:

1. Draw a filler sequence from a per-record RNG seeded `sha256(doc_id + condition)` — not a
   global stream, so it cannot shift when other documents are processed.
2. Build two character *tapes*; the left tape's end and the right tape's start abut the passage.
3. Slice outward to the character budget, cutting at whitespace where the loss is ≤32 chars and
   falling back to a hard cut otherwise (fiqa contains URLs and figures with no whitespace for
   hundreds of characters).
4. Gold position is a continuous fraction over characters, binned `start` / `middle` / `end`.

**Guaranteed by construction and verified on every document:**

- The passage is **never truncated**; `text[gold_char_start:gold_char_end]` always equals it.
- **Infix nesting**: `doc(smaller rung)` is an exact contiguous substring of `doc(larger rung)`,
  so length is the only variable across the ladder.
- Corpus size is identical at every rung.
- Text is raw — no detokenize round-trip, real newlines preserved, explicit `\n\n` separators.

---

## Layout

```
data/retrieval/beir_longctx/v1/{dataset}/
├── manifest.json            seed, budgets, separator, pool + qrels stats
├── safe_filler_pool.jsonl   passages usable as padding
├── bm25_neighbours.jsonl    frozen, auditable neighbour lists
├── queries.jsonl            copied unchanged from source
├── qrels/test.jsonl         copied unchanged from source
└── {condition}/c{rung}/
    ├── nongold.jsonl        welded non-gold documents
    ├── gold_start.jsonl     welded golds, needle near the start
    ├── gold_middle.jsonl    ... middle
    └── gold_end.jsonl       ... end
```

One eval run loads `nongold.jsonl` **plus exactly one** `gold_{bin}.jsonl`, which together
reproduce the source corpus document-for-document. A non-gold is never a query target, so its
own position cannot affect any metric and it needs only one variant — which keeps the
position-bin cost at 1.14× rather than 3×.

---

## Building

CPU only — no GPU is needed to build; only the eval encode step uses one.

```bash
conda activate htr                      # only env with scipy + rank_bm25
export PYTHONPATH="./src:$PYTHONPATH"
python src/data/long_context/build_benchmark.py --out_root data/retrieval/beir_longctx/v1
# or: sbatch scripts/data/long_context/build_beir_longctx.sh
```

Measured: scifact 77s, scidocs 7.6 min, fiqa 16.3 min. ~24 GB total.

### Verification (gates the eval)

```bash
bash scripts/data/long_context/verify_all.sh     # non-zero exit on any failure
```

48 checks per dataset × condition, **288 total, all passing**. Each corresponds to a measured
defect in the previous builder, so a corpus reproducing one cannot be written:

| check | what it prevents |
|---|---|
| gold intact at recorded offsets | gold clipped by truncation (old builder: 71% of rows) |
| gold appears exactly once | ambiguous offsets from duplicated text |
| corpus size constant across rungs | length confounded with corpus size (old: 447K→556K) |
| infix nesting across rungs | rungs not comparable |
| no qrel positive used as filler | manufactured false negatives |
| gold position not collapsed to 0 | fake "random" placement (old: 53% at offset 0) |
| padded docs respect the budget | under-filled or empty documents |
| separator present / raw whitespace | tokenizer round-trip artifacts |
| `weld.py` imports no tokenizer | corpus becoming model-specific |

Module self-tests, both of which have caught real defects:

```bash
python src/data/long_context/bm25.py   --self-test   # exact equivalence with rank_bm25
python src/data/long_context/verify.py --self-test   # each check fires on its target defect
conda activate bert24 && python -m pytest tests/test_long_context_weld.py   # 59 invariants
```

---

## Running the evaluation

```bash
python src/model/eval/eval_longctx.py \
    --benchmark_dir data/retrieval/beir_longctx/v1/BeIR_scifact \
    --condition random --rung 27000 --position middle \
    --model_name_or_path <path> --model_label HMB-native \
    --pooling cls --strategy native --window 8192 \
    --output_file outputs/eval/longctx/HMB-native.json
```

### Encoding strategies

| strategy | behaviour |
|---|---|
| `native` | one pass over the whole document; refused if it exceeds the model's real limit |
| `truncate` | first `window` tokens only — **label honestly**, this is the truncation condition |
| `chunked` | fixed overlapping token windows, max-pooled over window scores |
| `chunked_para` | split on paragraph boundaries, pack to the window — **the primary chunked baseline** |

`chunked_para` exists because fixed windows quietly handicap the baseline: scifact's median gold
is 402 mE5 tokens against a 510-token window, so fixed chunking leaves the gold *complete in no
window* 92% of the time at stride 64. Paragraph-aware chunking raises that to 73% and costs
fewer windows than 50%-overlap fixed chunking.

The window limit is read from each model's config rather than passed by hand, and an impossible
configuration is **refused** rather than silently truncated — the previous harness pinned mE5 to
512 in a shell variable at every context size and reported the result as "degradation".

### The c0 sanity gate

`c0` is the unpadded corpus, so it has a known right answer and can falsify the harness before
any long-context number is interpreted.

```bash
sbatch scripts/model/eval/c0_sanity_gate.sh
```

**Status: PASS** — 12/12 cells, 11 exact:

| arm | scifact | scidocs | fiqa |
|---|---|---|---|
| mE5-large | 0.581 | 0.139 | 0.335 |
| mE5-base | 0.549 | 0.125 | 0.241 |
| NeoDictaBERT | 0.501 | 0.093 | 0.288 |
| HMB-base-final | 0.309 | 0.031 | 0.055 |

> ⚠️ **`EVALUATION.md` is inflated for multi-positive datasets.** Its numbers disagree with the
> machine-written `results.json` files, and the gap scales with positives-per-query: ~0.000 on
> scifact (1.1/query), ~0.045 on fiqa (2.6), ~0.093 on scidocs (4.9) — the signature of the
> IDCG-over-retrieved-slice bug that `compute_metrics` later fixed. **Trust `results.json`.**
> The ranking is unaffected; the magnitudes are not.

---

## Results (scifact, `random`, gold in middle)

NDCG@10:

| arm | c0 | c3.7k | c19k | c27k | retention |
|---|---|---|---|---|---|
| **mE5-large + para-chunking** | 0.581 | 0.446 | 0.400 | **0.403** | 69% |
| mE5-base + para-chunking | 0.549 | 0.443 | 0.383 | 0.392 | 71% |
| NeoDictaBERT + para-chunking | 0.501 | 0.363 | 0.239 | 0.236 | 47% |
| HMB + para-chunking | 0.309 | 0.209 | 0.131 | 0.129 | 42% |
| NeoDictaBERT native (4096) | 0.501 | 0.026 | 0.002 | **0.000** | 0% |
| HMB native (8192) | 0.309 | 0.019 | 0.001 | **0.000** | 0% |
| mE5-large truncate (512) | 0.581 | 0.086 | 0.000 | 0.000 | 0% |

**Native single-vector encoding of padded documents collapses to zero** — for *both*
independently fine-tuned Hebrew dual encoders. Chunk-and-max-pool does not. HMB *chunked*
(0.129) beats HMB *native* (0.000) by two orders of magnitude, so an 8192-token window is not
an asset here.

Three checks that this is not a harness artifact:
- **Window probe**: on *unpadded* documents, widening the window costs ~nothing
  (NDB 0.5008@512 → 0.4973@4096; HMB 0.3090@512 → 0.3030@8192).
- **Anchor**: mE5-base chunked reproduced 0.443 exactly from a prior validated run.
- `R@100 = 0.0067` for NDB-native @ c27k — the gold is at a random rank, not a near miss.

### Interpretation, and what it does *not* show

Both HMB and NeoDictaBERT were fine-tuned at `--max_length 512` on passages of median ~264
tokens. The collapse is therefore most likely a **train/inference mismatch**, not evidence that
long-context retrieval is impossible. mE5 never hits the mismatch because its window *is* 512
and it is only ever used chunked.

The defensible claim is: *an 8192-context model fine-tuned on short passages cannot exploit its
window; chunk-and-pool dominates in Hebrew.* Testing the window hypothesis properly requires
training on **welded long documents** — note that simply raising `--max_length` does nothing,
since no training example approaches the limit.

### Limitations to disclose

- Documents are synthetic — a real passage glued to unrelated filler, so topical discontinuity
  is a possible shortcut. Same-domain filler mitigates but does not remove this.
- `chunked_para` exploits paragraph structure that exists because documents were welded from
  passages. Real long documents do have paragraph breaks and every arm gets the same benefit,
  but it is a way the construction is exploitable.
- HMB and NeoDictaBERT are fine-tuned on translated BeIR; mE5 is zero-shot. This favours the
  fine-tuned models.
- Results above are scifact only, `random` condition, gold in the middle.

---

## Why this replaces the old benchmark

The HeQ corpora under `data/retrieval/heq/test/documents*_long_context_*.jsonl` (~46 GB) are
invalid and should not be used. Measured, not suspected:

1. The gold passage was clipped by a final truncation in **71%** of 512-token rows (52% at 1024,
   32% at 2048) — the needle was frequently not in the haystack. 4% had negative spans.
2. Gold position collapsed to offset 0 in **53%** of rows, because placement sampled over 1–3
   passage *slots* rather than character offsets.
3. Sizes were measured in **mE5's tokenizer**, silently calibrating every corpus to mE5.
4. Stored text was a **detokenize round-trip** through mE5 SentencePiece, destroying newlines.
5. Passages were joined with `""` — no separator, text running together mid-sentence.
6. A global RNG whose stream depended on per-record draw counts gave the same document different
   filler and a different gold position at every size.
7. Corpus size varied across sizes (447K→556K after dedup), confounding length with corpus size.
8. The eval pinned mE5 to `MAX_LEN=512` at every size, so its published "degradation" was a
   truncation artifact. 31 of 35 SLURM jobs failed; NeoDictaBERT never completed a single run.

---

## Code

| file | purpose |
|---|---|
| `src/data/long_context/weld.py` | welding core — no tokenizer, no I/O, 59 pytest invariants |
| `src/data/long_context/rng.py` | per-record deterministic seeding |
| `src/data/long_context/pool.py` | safe filler pools with leakage exclusion |
| `src/data/long_context/bm25.py` | frozen sparse BM25 selector (exact `rank_bm25` equivalence) |
| `src/data/long_context/calibrate.py` | tokenizer calibration, gold token offsets |
| `src/data/long_context/verify.py` | the assertion suite + its self-test |
| `src/data/long_context/build_benchmark.py` | staged, resumable orchestrator |
| `src/model/eval/eval_longctx.py` | eval driver |
| `src/model/eval/longctx_encoding.py` | strategies, capability guard, window aggregation |
| `src/model/eval/longctx_metrics.py` | position-binned metrics, cluster bootstrap |

### Environment notes

- `htr` — has scipy, rank_bm25, faiss, sentence_transformers. **No pytest.**
- `bert24` — has pytest. **No scipy.** Run `tests/test_long_context_weld.py` here.
- `biu` — NumPy ABI conflict breaks `import transformers`.

Tokenizer loads from the shared filesystem intermittently stall for 90s+; this looks like a hang
but is not. Load each tokenizer once and reuse it.

# Wiring `ladder_dedup` into the ladder

**Do NOT apply while a ladder run is live.** Each dataset spawns a fresh process
that re-imports `run_beir_ladder_pipeline.py`; an in-progress edit can crash the
next dataset at import time. Wire this in only after the loop drains, behind a
default-off config flag, then test on a small dataset first.

`ladder_dedup.py` is a tested, standalone module (orphan — nothing imports it).
This file is the plan to connect it.

## Config (default OFF)

```yaml
dedup:
  enabled: false                  # turn on only for the large tier
  cache_path: ""                  # default: <run_dir>/translation_cache.sqlite
```

`run_ladder` reads `config.get("dedup", {})`. If `not enabled`, **do nothing** —
behavior is byte-identical to today. One shared `SqliteTranslationCache` is
opened per `run_ladder` call (sequential loop ⇒ single writer ⇒ no race).

## Two hook points in `run_ladder` (file: run_beir_ladder_pipeline.py)

### A. Prefill — in the submit loop, ~line 1139 (before `_submit_shard_job`)
For each `(shard_idx, text_type)` that isn't already `appended`:

```python
if dedup_enabled:
    text_col   = type_cfg["prompt"]["text_col"]
    ctx_col    = type_cfg["prompt"].get("context_col") or None   # queries only
    stats = prefill_shard(shard_csv, cache, type_cfg["model"],
                          type_cfg["prompt"]["file"], text_col, ctx_col,
                          out_path=out_path)            # write filled rows to OUTPUT, not the candidate
    if stats["all_cached"]:
        # NO-JOB SHARD: output_path is fully written from cache. Do not submit.
        cached_shards[(shard_idx, text_type)] = {
            "output_path": out_path, "input_tokens": 0, "output_tokens": 0}
        shard_record[text_type] = {"job_name": None, "cached_only": True,
                                   "submitted_at": now_iso()}
        save_progress(run_dir, progress)
        continue        # skip _submit_shard_job
    # else: PARTIAL or MISS → fall through and submit the WHOLE shard as today
    #       (Phase 1 ignores partial fills; see Phasing below)
```

Key point: write prefilled rows to `out_path` (the `_translated.csv`), **never**
overwrite the candidate CSV. For Phase 1, only the `all_cached` branch writes;
partial shards are submitted whole (cache still gets populated at finalize).

### B. Merge no-job shards + finalize — after `_collect_shard_results`, ~line 1191
```python
shard_results = _collect_shard_results(pending_jobs, gcs_client, bucket, config)
if dedup_enabled:
    shard_results.update(cached_shards)     # no-job shards flow through repair/accumulate/QA
```
Then after the accumulate loop stores translations (~line 1250), for every shard
that was actually translated (not `cached_only`), populate the cache so later
datasets hit it:
```python
if dedup_enabled and not rec.get("cached_only"):
    finalize_shard(r["output_path"], cache, type_cfg["model"],
                   type_cfg["prompt"]["file"], text_col, ctx_col)
```

## The no-job-shard state path (where bugs live — test hardest)

A `cached_only` shard must traverse the SAME state machine as a normal shard, but
without a job:

| stage | normal shard | no-job (cached_only) shard |
|---|---|---|
| submit | `_submit_shard_job` → `job_name` | prefill writes `out_path`; record `job_name=None, cached_only=True`; **no** entry in `pending_jobs` |
| poll | polled to terminal | **not polled** (not in `pending_jobs`) |
| collect | in `shard_results` | injected via `shard_results.update(cached_shards)` |
| repair | `_repair_shard_csv` runs | runs (finds nothing — cached text was already good) — harmless, or skip on `cached_only` to save time |
| accumulate | append, `appended=True`, tokens | append, `appended=True`, **tokens=0** (no batch cost) |
| cost | `_compute_cost(tokens)` | adds **$0** (tokens 0) |
| QA gate | judged | judged (output exists in accumulated) |

### Resume idempotency (critical)
- `cached_shards` is rebuilt each run from prefill — fine, it's deterministic.
- The existing `if existing.get("appended"): continue` (line 1131) already short-
  circuits a `cached_only` shard that finished on a prior run → it is NOT
  re-prefilled, NOT re-submitted. ✓
- A `cached_only` shard interrupted **after** prefill wrote `out_path` but
  **before** `appended=True`: on resume it has `job_name=None, cached_only=True,
  appended` absent → it is NOT in the `existing.appended` skip, has no
  `job_name` to reuse, so it re-prefills (idempotent — overwrites the same
  `out_path`) and re-injects. ✓  (Verify: the `existing.get("job_name")` branch
  at 1140 must not fire for `job_name=None`.)
- `finalize_shard` is `INSERT OR IGNORE` → re-storing is a no-op. ✓

## Phasing
- **Phase 1 (recommended first):** handle only `all_cached` shards. Captures
  fever≡climate (~5.49M segments, ~$960 batch) fully, since every climate shard
  is 100% cache hits once fever is done. Partial shards translate whole as today.
  Lowest risk.
- **Phase 2 (optional, ~$108):** partial-shard merge — submit only the uncached
  rows of a partial shard, then merge cached rows back by `_id` on collect. This
  captures hotpotqa∩dbpedia (~1.16M scattered ⇒ partial shards). More complex
  (split + merge + token accounting on a subset); defer unless the $108 matters.

## Ordering requirement (already satisfied)
Cache must be populated before the consumer runs. The config dataset order has
fever before climate-fever and hotpotqa before dbpedia-entity, so the producer
always precedes the consumer. If that order ever changes, dedup silently does
nothing (correctness preserved, savings lost) — log cache size per dataset.

## Scale
~25M unique segments. SQLite keeps the index on disk; point lookups are batched
(`lookup_keys`, 900-var chunks). Expect the cache file to reach a few GB. The
JSONL `TranslationCache` would re-parse that per process — don't use it here.
`migrate_jsonl` can import any existing JSONL cache once.

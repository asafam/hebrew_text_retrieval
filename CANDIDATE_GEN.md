# Candidate Generation

Builds sharded translation candidates for all 15 BeIR datasets.
Each dataset is split into fixed-size shards; a `shard_manifest.json` records
every shard's filename and row count so the ladder pipeline can process them
in deterministic order and apply QA gating after each shard.

---

## Configuration

All settings live in **`config/translation/candidates.yaml`** — the single
source of truth for dataset names, shard sizes, tokenizer, and output paths.

| Setting | Value |
|---------|-------|
| Datasets | 15 BeIR datasets, ordered small → large |
| Shard sizes | Per-dataset: 500 rows (nfcorpus) → 100,000 rows (msmarco) |
| Tokenizer | `gpt-4o-mini-2024-07-18` |
| Max segment tokens | 512 |
| Output | `outputs/translation/candidates/` |
| Logs | `outputs/translation/candidates/logs/` |

> **Note:** Three datasets (`fiqa`, `webis-touche2020`, `cqadupstack`) use
> `fastparquet` to work around a PyArrow 19 incompatibility. This is handled
> transparently by the loader — no action needed.

---

## Output structure

```
outputs/translation/candidates/
  BeIR_nfcorpus/
    queries_shard_000.csv       # ≤ shard_size rows
    documents_shard_000.csv
    documents_shard_001.csv
    ...
    shard_manifest.json         # index of all shards + row counts
  BeIR_scifact/
    ...
  logs/
    BeIR_nfcorpus.log
    BeIR_scifact.log
    ...
```

`shard_manifest.json` example:
```json
{
  "shard_size": 500,
  "types": {
    "queries": [
      {"index": 0, "file": "queries_shard_000.csv", "rows": 323}
    ],
    "documents": [
      {"index": 0, "file": "documents_shard_000.csv", "rows": 500},
      {"index": 1, "file": "documents_shard_001.csv", "rows": 500},
      {"index": 2, "file": "documents_shard_002.csv", "rows": 500},
      {"index": 3, "file": "documents_shard_003.csv", "rows": 133}
    ]
  }
}
```

---

## Step-by-step

### 1. Environment

```bash
cd ~/Workspace/biu/hebrew_text_retrieval
conda activate htr          # or: conda activate biu
export PYTHONPATH="./src:$PYTHONPATH"
```

### 2. Pilot — run one small dataset first

Validates the environment without launching all 15 jobs in parallel.

```bash
python scripts/translation/build_ladder_candidates.py \
    --config config/translation/candidates.yaml \
    --dataset nfcorpus
```

nfcorpus has ~3,600 documents and finishes in under a minute.
Check the manifest to confirm the shard counts look right:

```bash
cat outputs/translation/candidates/BeIR_nfcorpus/shard_manifest.json
```

Spot-check a shard file:

```bash
head -3 outputs/translation/candidates/BeIR_nfcorpus/queries_shard_000.csv
```

### 3. Run all 15 datasets in parallel

```bash
bash scripts/translation/build_ladder_candidates.sh
```

Or equivalently:

```bash
python scripts/translation/build_ladder_candidates.py \
    --config config/translation/candidates.yaml
```

A live table updates in place while all datasets run in parallel:

```
Building ladder candidates  (42s)
Dataset                    Status      Shard size  Q shards  D shards  Queries  Documents  Time
BeIR/nfcorpus              ✓ done         500          1        8      323      3,633      38s
BeIR/scifact               ⟳ running     1000          -        -        -          -      12s
BeIR/arguana               · pending        -          -        -        -          -       -
...
```

Datasets already completed (manifest exists) are shown as `✓ skipped` and
not rebuilt. Failed datasets print the last 5 lines of their log.

### 4. Resume after a failure

Already-completed datasets are skipped automatically — just rerun the same
command. No `--force` needed.

```bash
bash scripts/translation/build_ladder_candidates.sh
```

### 5. Force rebuild everything

```bash
bash scripts/translation/build_ladder_candidates.sh --force
```

---

## Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config (default: `config/translation/candidates.yaml`) |
| `--dataset NAME` | Partial name filter, e.g. `nfcorpus` or `scifact` |
| `--force` | Rebuild even if `shard_manifest.json` already exists |

---

## Next step

Once all candidates are built, run the ladder translation pipeline:

```bash
# Dry run — prints shard plan, no translation
python -m translation.api.run_beir_ladder_pipeline \
    --config config/translation/full_corpus.yaml --dry-run

# Full run
python -m translation.api.run_beir_ladder_pipeline \
    --config config/translation/full_corpus.yaml
```

See `README.md` for the full pipeline description and kill-safety notes.

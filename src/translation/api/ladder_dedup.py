"""
Content-addressed translation dedup for the BeIR ladder pipeline.

STANDALONE / ORPHAN MODULE — intentionally imports nothing from the rest of the
pipeline (only stdlib + pandas) so it can be developed and tested while a ladder
run is live, without risk of breaking a fresh process's import of the ladder.
The ladder wires it in via a single import *after* a run drains.

Why this exists
---------------
The ladder translates every shard of every dataset independently and never
dedups. But across the BeIR corpora a large fraction of document segments are
identical text:

  * fever ≡ climate-fever      ~5.49M shared segments (the whole corpus)
  * hotpotqa ∩ dbpedia-entity  ~1.16M shared segments
  * plus smaller cross-corpus overlaps and within-shard duplicates

Deduping removes ~6.7M of ~32.4M large-tier segments (~21%). The win is realized
by translating each unique (model, prompt, text, context) once and reusing it.

Why SQLite (not the JSONL TranslationCache)
-------------------------------------------
The existing `translation.beir.cache.TranslationCache` is JSONL and, by its own
docstring, is meant for <=500K entries — it loads the entire file into a dict at
construction. The large tier reaches ~25M unique segments, and the ladder spawns
one process per dataset, so a JSONL cache would re-parse a multi-GB file at every
process start. SQLite keeps the index on disk, supports batched point lookups,
and — because the ladder loop is sequential — never has a write race.

Key compatibility
------------------
The key scheme is identical to `TranslationCache._make_key`:
    sha256(f"{model}\\x00{prompt_file}\\x00{text}\\x00{context}")
so a JSONL cache can be migrated in (see `migrate_jsonl`) and the two stay
interchangeable.

The no-job-shard path
---------------------
A shard that is 100% cache hits needs NO Vertex batch job. `prefill_shard`
writes the fully-translated output CSV directly and reports `all_cached=True`;
the ladder must then route that shard through its collect/append/resume state
machine as "output already written, done" without ever submitting or polling.
That state path — not the helpers here — is where integration bugs live; see
INTEGRATION.md / the wiring notes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Iterable, Optional

import pandas as pd

TRANSLATION_COL = "translation"


def make_key(model_name: str, prompt_file: str, text: str, context: str = "") -> str:
    """sha256 key — byte-identical to translation.beir.cache.TranslationCache."""
    content = f"{model_name}\x00{prompt_file}\x00{text}\x00{context}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SqliteTranslationCache:
    """On-disk content-addressed cache. Same key scheme as the JSONL cache.

    Safe for a sequential ladder loop (one writer at a time). Uses WAL so a
    read-only status process can inspect it concurrently.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS translations ("
            "  key TEXT PRIMARY KEY,"
            "  translation TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    # ── reads ────────────────────────────────────────────────────────────────
    def lookup(self, model_name: str, prompt_file: str, text: str, context: str = "") -> Optional[str]:
        key = make_key(model_name, prompt_file, text, context)
        cur = self._conn.execute("SELECT translation FROM translations WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def lookup_keys(self, keys: Iterable[str]) -> dict:
        """Batch point-lookup by precomputed key. Returns {key: translation}."""
        keys = list(keys)
        out: dict[str, str] = {}
        CHUNK = 900  # under SQLite's default 999 variable limit
        for i in range(0, len(keys), CHUNK):
            chunk = keys[i:i + CHUNK]
            q = "SELECT key, translation FROM translations WHERE key IN (%s)" % ",".join("?" * len(chunk))
            for k, t in self._conn.execute(q, chunk):
                out[k] = t
        return out

    # ── writes ───────────────────────────────────────────────────────────────
    def store(self, model_name: str, prompt_file: str, text: str, context: str, translation: str) -> None:
        self.store_keyed([(make_key(model_name, prompt_file, text, context), translation)])

    def store_keyed(self, items: Iterable[tuple[str, str]]) -> int:
        """Insert (key, translation) pairs; ignore keys already present.

        Returns the number of newly inserted rows.
        """
        items = [(k, t) for (k, t) in items if t is not None]
        if not items:
            return 0
        before = len(self)
        self._conn.executemany(
            "INSERT OR IGNORE INTO translations(key, translation) VALUES (?, ?)", items
        )
        self._conn.commit()
        return len(self) - before

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _row_keys(df: pd.DataFrame, model_name: str, prompt_file: str,
              text_col: str, context_col: Optional[str]) -> pd.Series:
    """Vectorized per-row cache keys for a dataframe."""
    text = df[text_col].fillna("").astype(str) if text_col in df.columns else pd.Series([""] * len(df))
    if context_col and context_col in df.columns:
        ctx = df[context_col].fillna("").astype(str)
    else:
        ctx = pd.Series([""] * len(df), index=df.index)
    prefix = f"{model_name}\x00{prompt_file}\x00"

    def _k(t, c):
        return hashlib.sha256(f"{prefix}{t}\x00{c}".encode("utf-8")).hexdigest()

    return pd.Series([_k(t, c) for t, c in zip(text, ctx)], index=df.index)


def prefill_shard(
    shard_csv: str,
    cache: SqliteTranslationCache,
    model_name: str,
    prompt_file: str,
    text_col: str,
    context_col: Optional[str] = None,
    out_csv: Optional[str] = None,
) -> dict:
    """Pre-fill a shard's `translation` column from the cache + within-shard dups.

    Two layers, mirroring the legacy `_apply_cache_and_dedup`:
      1. cache hits: rows whose key is already in the cache are filled.
      2. within-shard dedup: among rows sharing identical (text, context), one
         filled value propagates to the rest.

    Writes the (possibly partially filled) shard to `out_csv` (defaults to
    overwriting `shard_csv`) ONLY when something was filled. Returns stats incl.
    `all_cached` — True when no row needs a batch translation (the no-job path).
    """
    out_csv = out_csv or shard_csv
    df = pd.read_csv(shard_csv, encoding="utf-8")
    if TRANSLATION_COL not in df.columns:
        df[TRANSLATION_COL] = None

    total = len(df)
    already = int(df[TRANSLATION_COL].notna().sum())

    keys = _row_keys(df, model_name, prompt_file, text_col, context_col)
    need = df[TRANSLATION_COL].isna()
    found = cache.lookup_keys(keys[need].unique().tolist())

    cache_hits = 0
    if found:
        fill = keys.map(found)  # NaN where key absent
        mask = need & fill.notna()
        df.loc[mask, TRANSLATION_COL] = fill[mask]
        cache_hits = int(mask.sum())

    # within-shard dedup
    group_keys = [text_col] + ([context_col] if context_col and context_col in df.columns else [])
    dedup_fills = 0
    for _, g in df.groupby(group_keys, sort=False, dropna=False):
        filled = g[TRANSLATION_COL].notna()
        empty = g[TRANSLATION_COL].isna()
        if filled.any() and empty.any():
            df.loc[g.index[empty], TRANSLATION_COL] = g.loc[filled, TRANSLATION_COL].iloc[0]
            dedup_fills += int(empty.sum())

    remaining = int(df[TRANSLATION_COL].isna().sum())
    if cache_hits or dedup_fills:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")

    return {
        "total": total,
        "already_done": already,
        "cache_hits": cache_hits,
        "dedup_fills": dedup_fills,
        "remaining": remaining,
        "all_cached": remaining == 0,
    }


def finalize_shard(
    translated_csv: str,
    cache: SqliteTranslationCache,
    model_name: str,
    prompt_file: str,
    text_col: str,
    context_col: Optional[str] = None,
) -> dict:
    """After a shard is translated: propagate within-shard dups, store new keys.

    Mirrors `_expand_dedup_and_update_cache`. Idempotent — storing the same key
    twice is a no-op (INSERT OR IGNORE), so re-running on `--resume` is safe.
    """
    if not os.path.exists(translated_csv):
        return {"dedup_fills": 0, "new_entries": 0}
    df = pd.read_csv(translated_csv, encoding="utf-8")
    if TRANSLATION_COL not in df.columns:
        return {"dedup_fills": 0, "new_entries": 0}

    group_keys = [text_col] + ([context_col] if context_col and context_col in df.columns else [])
    dedup_fills = 0
    for _, g in df.groupby(group_keys, sort=False, dropna=False):
        filled = g[TRANSLATION_COL].notna()
        empty = g[TRANSLATION_COL].isna()
        if filled.any() and empty.any():
            df.loc[g.index[empty], TRANSLATION_COL] = g.loc[filled, TRANSLATION_COL].iloc[0]
            dedup_fills += int(empty.sum())
    if dedup_fills:
        df.to_csv(translated_csv, index=False, encoding="utf-8")

    keys = _row_keys(df, model_name, prompt_file, text_col, context_col)
    have = df[TRANSLATION_COL].notna()
    items = list(zip(keys[have].tolist(), df.loc[have, TRANSLATION_COL].astype(str).tolist()))
    # de-dup keys within this batch before insert
    seen, uniq = set(), []
    for k, t in items:
        if k not in seen:
            seen.add(k)
            uniq.append((k, t))
    new_entries = cache.store_keyed(uniq)
    return {"dedup_fills": dedup_fills, "new_entries": new_entries}


def migrate_jsonl(jsonl_path: str, cache: SqliteTranslationCache) -> int:
    """Load an existing JSONL TranslationCache into SQLite. Returns rows added."""
    if not os.path.exists(jsonl_path):
        return 0
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                items.append((e["key"], e["translation"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return cache.store_keyed(items)

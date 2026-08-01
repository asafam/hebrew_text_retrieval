"""
Build the translated-BeIR long-context benchmark.

Usage (CPU only -- no GPU is needed to build; only the eval encode step uses one)::

    conda activate htr
    export PYTHONPATH="./src:$PYTHONPATH"
    python src/data/long_context/build_benchmark.py --datasets BeIR_scifact
    python src/data/long_context/build_benchmark.py            # all three, cheapest first

Layout written per dataset::

    data/retrieval/beir_longctx/v1/{dataset}/
      manifest.json            provenance: seed, budgets, separator, pool + qrels stats
      safe_filler_pool.jsonl   passages usable as padding (no qrel positive, any split)
      queries.jsonl            copied unchanged from source
      qrels/test.jsonl         copied unchanged from source
      {condition}/c{rung}/
        nongold.jsonl          welded corpus docs that are NOT test golds
        gold_start.jsonl       welded test-gold docs, gold near the start
        gold_middle.jsonl      ... middle
        gold_end.jsonl         ... end

Two naming distinctions worth keeping straight, because "filler" is easy to overload:

* the **filler pool** is the set of passages used as *padding*, and it excludes every qrel
  positive in every split -- padding an irrelevant document with another query's gold would
  make that document genuinely contain the answer while the qrels call it irrelevant.
* **nongold.jsonl** holds welded corpus documents that are not test golds. They are welded by
  exactly the same rule as golds (otherwise a gold would be identifiable by its padding), but
  they need only one variant: a non-gold is never a query target, so where its own passage
  sits cannot affect any metric. That is what keeps the position-bin cost at 1.14x rather
  than 3x, for both disk and encode time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Sequence

from data.long_context.bm25 import BM25Index, bm25_tokenize
from data.long_context.calibrate import LADDER_CHARS
from data.long_context.pool import (
    ALL_QREL_SPLITS,
    BENCHMARK_DATASETS,
    Passage,
    assert_pool_is_leakage_free,
    build_safe_filler_pool,
    find_corpus_dirs,
    load_corpus,
    load_qrels,
    positive_doc_ids,
    save_pool,
)
from data.long_context.rng import GLOBAL_SEED, record_rng, stable_bin
from data.long_context.weld import SEPARATOR, build_tapes, sample_position_frac, weld

POSITION_BINS = ("start", "middle", "end")
CONDITIONS = ("random", "bm25")

#: How many BM25 neighbours to retain per seed.
#:
#: This must be large enough to fill the largest rung's tapes entirely from BM25-selected
#: passages, or the `bm25` condition silently becomes a blend of BM25 and random filler. The
#: tapes need ~2 x 27,000 = 54,000 characters, and fiqa's mean passage is only 414 characters,
#: so 64 neighbours would cover just 26,500 -- under half. 200 covers ~82,800 and keeps the
#: condition meaning what it says. The extra cost is negligible: top-k is an argpartition.
N_NEIGHBOURS = 200

#: Filler passages sampled per document, instead of shuffling the whole pool.
#:
#: The pool shuffle was the dominant build cost and almost all of it was waste: welding one
#: document consumes ~50-130 passages, but the old code shuffled every passage in the pool
#: (40,273 for fiqa) once per document *per rung*. Measured build time scaled as
#: docs x pool_size -- 115s for scifact, 2,435s for scidocs, and a projected 174 minutes for
#: fiqa. Sampling k instead is O(k).
#:
#: k is chosen so the tapes fill even in the worst case: they need ~54,000 characters and the
#: pool's minimum passage length is 100 characters, so 542 passages suffice; 900 leaves margin.
#: Under-filling would be caught by the verifier's fill-fraction check regardless.
FILLER_SAMPLE_K = 900

#: Minimum seed-passage length, in characters, for the containment-based filler filter.
#:
#: Excluding filler that *contains* the seed passage prevents substring-form leakage, but it is
#: only meaningful for a substantive span. fiqa contains a document that is the single word
#: "מקור" ("source", 4 characters) and 281 documents under 64 characters; for those, containment
#: matches ordinary language rather than a duplicated answer, so the filter excluded nearly all
#: 900 sampled filler passages and produced empty tapes -- one document came out with zero
#: characters and no separator.
#:
#: Below this length only exact-text and same-id exclusion apply. A 4-character common word
#: appearing in filler is not leakage; leakage means an irrelevant document genuinely carries
#: the answer, which needs more than a word.
MIN_CONTAINMENT_CHARS = 64


def _write_jsonl(path: str, records: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a killed job cannot leave a half-written shard


def build_dataset(
    dataset: str,
    beir_dir: str,
    out_root: str,
    *,
    conditions: Sequence[str] = CONDITIONS,
    rungs: Sequence[int] = LADDER_CHARS,
    separator: str = SEPARATOR,
    force: bool = False,
) -> dict:
    t0 = time.time()
    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    corpus = load_corpus(beir_dir)
    test_qrels = load_qrels(beir_dir, "test")
    test_golds = {c for docs in test_qrels.values() for c in docs} & set(corpus)
    pool, stats = build_safe_filler_pool(beir_dir, dataset=dataset, splits=ALL_QREL_SPLITS)
    assert_pool_is_leakage_free(pool, beir_dir, ALL_QREL_SPLITS)
    save_pool(pool, os.path.join(out_dir, "safe_filler_pool.jsonl"))
    print(f"  {stats}", flush=True)
    print(f"  test golds in corpus: {len(test_golds):,} / {len(corpus):,}", flush=True)

    # Copy queries and qrels verbatim: welding changes document text, never identity, so the
    # original relevance judgements stay valid without any remapping.
    shutil.copyfile(
        os.path.join(beir_dir, "queries.jsonl"), os.path.join(out_dir, "queries.jsonl")
    )
    src_qrels = os.path.join(beir_dir, "qrels", "test.jsonl")
    if os.path.exists(src_qrels):
        os.makedirs(os.path.join(out_dir, "qrels"), exist_ok=True)
        shutil.copyfile(src_qrels, os.path.join(out_dir, "qrels", "test.jsonl"))

    pool_ids = [p.pid for p in pool]
    pool_pairs = [(p.pid, p.text) for p in pool]
    pool_by_id = {p.pid: p for p in pool}
    neighbours: dict[str, list[str]] = {}

    if "bm25" in conditions:
        print(f"  building BM25 index over {len(pool):,} filler passages...", flush=True)
        index = BM25Index.build(pool_ids, [p.text for p in pool])
        index.save(os.path.join(out_dir, "bm25_index"))
        seed_ids = sorted(corpus)
        print(f"  BM25 top-{N_NEIGHBOURS} for {len(seed_ids):,} seeds...", flush=True)
        hits = index.top_k(
            [corpus[s] for s in seed_ids], k=N_NEIGHBOURS, block=1000, progress=False
        )
        neighbours = {
            sid: [pool_ids[i] for i, _ in h] for sid, h in zip(seed_ids, hits)
        }
        _write_jsonl(
            os.path.join(out_dir, "bm25_neighbours.jsonl"),
            [{"seed_id": s, "neighbours": n} for s, n in neighbours.items()],
        )

    max_rung = max(rungs)
    counts: dict[str, dict[int, int]] = {}

    # KNOWN INEFFICIENCY (correctness unaffected): the pool shuffle and tape construction
    # below sit inside the rung loop, so both repeat once per rung even though both are
    # rung-independent. Measured 115s for scifact (5,183 docs), and it scales with document
    # count: ~9 min for scidocs, ~21 min for fiqa.
    #
    # Fixing it is more than a hoist, for two reasons:
    #
    #  1. Memory. Emitting all five rungs for one document before moving on means holding
    #     ~68,900 characters per document. For fiqa's 57,600 documents that is roughly 8GB
    #     of Python strings, so the fix needs to batch (~1,000 documents at a time, writing
    #     incrementally to per-rung handles) rather than simply reorder the loops.
    #  2. RNG semantics. Today each (condition, rung, doc) creates a *fresh* record_rng and
    #     consumes it in the same order, so position fractions come out identical at every
    #     rung -- the required behaviour, but by accident rather than by construction.
    #     Draw the fraction from its own generator seeded on (doc_id, condition, bin) to make
    #     rung-independence explicit. That changes the drawn fractions, so it needs a full
    #     rebuild and re-verify of all three datasets, not just the new ones.
    #
    # Not worth doing mid-flight: a ~25 minute one-off cost against a refactor that would
    # invalidate an already-verified corpus.

    for condition in conditions:
        counts[condition] = {}
        for rung in rungs:
            rung_dir = os.path.join(out_dir, condition, f"c{rung}")
            done = os.path.join(rung_dir, "gold_end.jsonl")
            if os.path.exists(done) and not force:
                print(f"  [skip] {condition}/c{rung} already built", flush=True)
                continue

            n_starved = 0
            shards: dict[str, list[dict]] = {"nongold": []}
            for b in POSITION_BINS:
                shards[f"gold_{b}"] = []

            for did, passage in corpus.items():
                rng = record_rng(did, condition)
                k = min(len(pool), FILLER_SAMPLE_K)
                if condition == "bm25" and neighbours.get(did):
                    seq = [
                        (pid, pool_by_id[pid].text)
                        for pid in neighbours[did]
                        if pid in pool_by_id
                    ]
                    # Top up if the neighbour list still cannot fill the tapes. Any top-up
                    # dilutes the condition with random filler, so N_NEIGHBOURS is sized to
                    # make this rare rather than routine.
                    if sum(len(t) for _, t in seq) < 2 * max_rung:
                        seq += [(p.pid, p.text) for p in rng.sample(pool, k)]
                else:
                    # Sample rather than shuffle the whole pool: welding needs ~50-130
                    # passages, and shuffling 40k to use 130 was the build's dominant cost.
                    seq = [(p.pid, p.text) for p in rng.sample(pool, k)]

                # A non-gold corpus document is itself in the filler pool (the pool only
                # excludes qrel positives), so a document can otherwise draw its own passage
                # as its own filler -- duplicating the seed text and making the recorded gold
                # offsets ambiguous. The rate grows with the budget as more filler is drawn.
                #
                # Exclude by id, by exact text, and by *containment*: fiqa's passages are short
                # and repetitive, so a filler passage can contain the seed passage verbatim as a
                # substring (measured 1-2 documents per rung). That is substring-form leakage --
                # an irrelevant document would genuinely hold the answer -- so it has to go too.
                # `in` on short strings is cheap enough at this scale.
                check_containment = len(passage) >= MIN_CONTAINMENT_CHARS
                seq = [
                    (pid, t)
                    for pid, t in seq
                    if pid != did
                    and t != passage
                    and not (check_containment and passage in t)
                ]

                # Safety net: never let filtering starve the tapes. A filter that removes too
                # much yields an under-filled or even empty document -- a 4-character seed
                # passage once produced a zero-character document this way, and 26 fiqa
                # documents came out at 91-95% fill.
                #
                # Test the tapes themselves rather than a proxy for them. An earlier version
                # checked whether the *sequence* held more than 2 * max_rung characters, but
                # each tape must reach max_rung individually and passages are dealt alternately,
                # so a sequence can clear that bar and still leave one side short. Checking the
                # built tapes is exact and needs no tuning constant.
                tapes = build_tapes(seq, min_tape_chars=max_rung + 64, separator=separator)
                if len(tapes.left) < max_rung or len(tapes.right) < max_rung:
                    n_starved += 1
                    seq = [
                        (pid, t)
                        for pid, t in rng.sample(pool_pairs, k)
                        if pid != did and t != passage
                    ]
                    tapes = build_tapes(
                        seq, min_tape_chars=max_rung + 64, separator=separator
                    )
                is_gold = did in test_golds
                bins = POSITION_BINS if is_gold else (stable_bin(did, POSITION_BINS),)

                for b in bins:
                    r = weld(
                        passage,
                        tapes,
                        rung,
                        sample_position_frac(rng, b),
                        separator=separator,
                    )
                    rec = {
                        "_id": did,
                        "doc_id_base": did,
                        "seed_doc_id": did,
                        "title": "",
                        "text": r.text,
                        "is_gold": is_gold,
                        "gold_char_start": r.gold_char_start,
                        "gold_char_end": r.gold_char_end,
                        "position_bin": b,
                        "position_frac": round(r.position_frac, 6),
                        "padded": r.padded,
                        "n_chars": r.n_chars,
                        "condition": condition,
                        "rung_chars": rung,
                        "distractor_ids": list(r.distractor_ids[:16]),
                    }
                    shards["gold_" + b if is_gold else "nongold"].append(rec)

            for name, recs in shards.items():
                _write_jsonl(os.path.join(rung_dir, f"{name}.jsonl"), recs)
            if n_starved:
                print(f"    note: {n_starved:,} documents fell back to exact-match-only "
                      f"filler filtering (filter would have starved the tapes)", flush=True)
            n_total = len(shards["nongold"]) + len(shards["gold_start"])
            counts[condition][rung] = n_total
            print(
                f"  built {condition}/c{rung}: {len(shards['nongold']):,} nongold + "
                f"{len(shards['gold_start']):,} gold x3 bins = {n_total:,} docs/eval-run",
                flush=True,
            )

    manifest = {
        "dataset": dataset,
        "source_beir_dir": beir_dir,
        "global_seed": GLOBAL_SEED,
        "separator": separator,
        "rungs_chars": list(rungs),
        "conditions": list(conditions),
        "n_corpus": len(corpus),
        "n_test_golds": len(test_golds),
        "n_filler_pool": len(pool),
        "filler_excludes_positives_from_splits": list(ALL_QREL_SPLITS),
        "n_neighbours_per_seed": N_NEIGHBOURS if "bm25" in conditions else 0,
        "docs_per_eval_run": counts,
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs_root", default="outputs/translation/runs")
    ap.add_argument("--out_root", default="data/retrieval/beir_longctx/v1")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--rungs", nargs="*", type=int, default=list(LADDER_CHARS))
    ap.add_argument("--force", action="store_true", help="rebuild shards that already exist")
    args = ap.parse_args()

    dirs = find_corpus_dirs(args.runs_root, args.datasets or BENCHMARK_DATASETS)
    # cheapest first, so a mistake surfaces on the smallest corpus
    order = sorted(dirs, key=lambda d: os.path.getsize(os.path.join(dirs[d], "corpus.jsonl")))
    for dataset in order:
        print(f"\n=== {dataset} ===", flush=True)
        m = build_dataset(
            dataset,
            dirs[dataset],
            args.out_root,
            conditions=args.conditions,
            rungs=args.rungs,
            force=args.force,
        )
        print(f"  manifest written ({m['build_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

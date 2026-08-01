"""
Build a *welded* training set, so a long-context model can learn to use its window.

Motivation. HMB and NeoDictaBERT both collapse to ~0.000 NDCG@10 when asked to encode a padded
document natively, while chunk-and-max-pool over the same content holds up. The most likely
cause is a train/inference mismatch: both were fine-tuned on passages of median ~264 tokens and
have never seen a document where the answer is a small span inside mostly-irrelevant text.

Simply raising ``--max_length`` does **not** fix this. That flag is only a truncation ceiling,
and no training example approaches even 512 tokens, let alone 4096. The training *documents*
have to become long. This module makes them long, using the same welding procedure the
benchmark uses, so training and inference finally see the same kind of input.

What gets welded, and why all of it:

* the positive passage -- the thing the model must learn to find inside a haystack;
* **every hard negative too**. If only positives were welded, length and formatting alone would
  separate positive from negative and the model would learn that shortcut instead of retrieval.

Filler source. Training filler is drawn from a **neutral external corpus** (Hebrew Wikipedia),
not from the BeIR corpora, for two reasons.

*Leakage.* Using a BeIR corpus's own safe passages as filler would show the model 40K of fiqa's
57.6K documents during training, labelled implicitly as "never an answer". Even though those
passages are by construction never golds, a model that learns to downrank them makes the
remaining documents -- which include every gold -- easier to rank at eval time. That is a subtle
inflation of our own benchmark. External filler cannot do this.

*Availability.* nfcorpus has **zero** usable filler of its own: every one of its 3,633 documents
is a qrel positive for some query, so a per-dataset safe pool is empty. It is 88% of the
training data, so without external filler most of the training set cannot be welded at all.

The tradeoff is a distribution difference -- training filler is Wikipedia, eval filler is
in-corpus -- but the skill being taught is "find the needle regardless of what surrounds it",
which should if anything generalise better when the surroundings vary.

Length. Each example samples a rung from a weighted distribution rather than using one fixed
length. Mixed-length training is both cheaper and more robust than training at the maximum:
attention cost grows with sequence length, so a corpus welded entirely at 27,000 characters
would be dominated by its longest examples while teaching nothing about shorter ones.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Sequence

from data.long_context.pool import (
    ALL_QREL_SPLITS,
    build_safe_filler_pool,
    find_corpus_dirs,
)
from data.long_context.rng import record_rng, seed64
from data.long_context.weld import SEPARATOR, build_tapes, sample_position_frac, weld

#: Neutral filler corpus -- Hebrew Wikipedia, disjoint from every BeIR eval corpus.
DEFAULT_FILLER_SOURCE = "data/retrieval/heq/test/documents.jsonl"

#: Rung (characters) -> sampling weight. Weighted toward shorter lengths because cost grows with
#: sequence length, while still exposing the model to the decisive 19k/27k rungs it must handle
#: at eval time. Mean of this distribution is ~10,700 chars (~2,670 HMB tokens).
DEFAULT_RUNG_WEIGHTS: dict[int, float] = {
    3700: 0.30,
    7400: 0.25,
    11800: 0.20,
    19000: 0.15,
    27000: 0.10,
}

#: Filler passages sampled per welded document. Well above the ~130 needed to fill the largest
#: rung's tapes; sampling rather than shuffling the pool keeps this O(k). See build_benchmark.py.
FILLER_SAMPLE_K = 900

#: Below this length a seed passage is only excluded from its own filler by exact match, not by
#: containment -- a short common phrase matches most Hebrew text and would starve the tapes.
MIN_CONTAINMENT_CHARS = 64


def load_external_filler(
    path: str = DEFAULT_FILLER_SOURCE,
    *,
    min_chars: int = 200,
    max_chars: int = 3000,
    limit: int = 60000,
) -> list[tuple[str, str]]:
    """Load neutral filler passages from a corpus disjoint from the BeIR eval sets."""
    out: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if len(out) >= limit:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("text") or ""
            if min_chars <= len(t) <= max_chars:
                out.append((f"ext{i}", t))
    return out


def sample_rung(rng: random.Random, weights: dict[int, float]) -> int:
    rungs = sorted(weights)
    return rng.choices(rungs, weights=[weights[r] for r in rungs], k=1)[0]


class InsufficientFiller(RuntimeError):
    """The filler pool cannot fill the tapes. Never silently emit a short document."""


def weld_one(
    passage: str,
    pool_pairs: Sequence[tuple[str, str]],
    rng: random.Random,
    budget: int,
    *,
    max_rung: int,
    position_bin: str,
    separator: str = SEPARATOR,
) -> tuple[str, int, int]:
    """Weld a single passage to ``budget`` characters. Returns (text, gold_start, gold_end).

    Falls back to exact-match-only filtering if containment filtering would starve the tapes,
    matching the benchmark builder -- emitting a degenerate document is strictly worse than
    relaxing the filter.
    """
    k = min(len(pool_pairs), FILLER_SAMPLE_K)
    check_containment = len(passage) >= MIN_CONTAINMENT_CHARS
    seq = [
        (pid, t)
        for pid, t in rng.sample(pool_pairs, k)
        if t != passage and not (check_containment and passage in t)
    ]
    tapes = build_tapes(seq, min_tape_chars=max_rung + 64, separator=separator)
    if len(tapes.left) < budget or len(tapes.right) < budget:
        seq = [(pid, t) for pid, t in rng.sample(pool_pairs, k) if t != passage]
        tapes = build_tapes(seq, min_tape_chars=max_rung + 64, separator=separator)
    # Never emit an unwelded training document. nfcorpus's own filler pool is empty (every
    # document is a qrel positive), and an earlier version silently wrote 110,545 examples at
    # 12.6% of target length -- training data that teaches the exact opposite of the intent.
    if len(tapes.left) < budget or len(tapes.right) < budget:
        raise InsufficientFiller(
            f"filler pool yields tapes of {len(tapes.left)}/{len(tapes.right)} chars, "
            f"need {budget} each"
        )

    r = weld(passage, tapes, budget, sample_position_frac(rng, position_bin),
             separator=separator)
    return r.text, r.gold_char_start, r.gold_char_end


def build_for_dataset(
    dataset: str,
    beir_dir: str,
    out_path: str,
    *,
    rung_weights: dict[int, float],
    pool_pairs: Sequence[tuple[str, str]],
    limit: int | None = None,
    position_bin: str = "uniform",
) -> dict:
    t0 = time.time()

    src = os.path.join(beir_dir, "hard_negatives_train.jsonl")
    if not os.path.exists(src):
        print(f"  !! no hard_negatives_train.jsonl -- skipping {dataset}", flush=True)
        return {}

    max_rung = max(rung_weights)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = f"{out_path}.tmp"

    n_written = 0
    n_skipped_filler = 0
    rung_counts: dict[int, int] = {}
    total_chars = 0

    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if limit is not None and n_written >= limit:
                break
            d = json.loads(line)
            query = d.get("query")
            positive = d.get("positive")
            hard_negs = d.get("hard_negs") or []
            if not query or not positive:
                continue

            # Seed on the example, not a global stream, so the set is reproducible and adding
            # examples does not perturb earlier ones.
            rng = record_rng(f"{dataset}:{i}", "train")
            budget = sample_rung(rng, rung_weights)

            # Skip examples whose passage cannot fit -- never truncate a positive.
            if len(positive) + 2 * len(SEPARATOR) > budget:
                continue

            try:
                pos_text, gs, ge = weld_one(
                    positive, pool_pairs, rng, budget,
                    max_rung=max_rung, position_bin=position_bin,
                )
            except InsufficientFiller as e:
                if n_skipped_filler == 0:
                    print(f"    !! insufficient filler: {e}", flush=True)
                n_skipped_filler += 1
                continue
            welded_negs = []
            for neg in hard_negs:
                if not neg or len(neg) + 2 * len(SEPARATOR) > budget:
                    continue
                try:
                    nt, _, _ = weld_one(
                        neg, pool_pairs, rng, budget,
                        max_rung=max_rung, position_bin=position_bin,
                    )
                except InsufficientFiller:
                    continue
                welded_negs.append(nt)

            if not welded_negs:
                continue  # an example with no negatives teaches nothing about discrimination

            fout.write(json.dumps({
                "query": query,
                "positive": pos_text,
                "hard_negs": welded_negs,
                "rung_chars": budget,
                "gold_char_start": gs,
                "gold_char_end": ge,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            rung_counts[budget] = rung_counts.get(budget, 0) + 1
            total_chars += len(pos_text) + sum(len(t) for t in welded_negs)

            if n_written % 5000 == 0:
                print(f"    {n_written:,} examples...", flush=True)

    os.replace(tmp, out_path)
    meta = {
        "dataset": dataset,
        "source": src,
        "n_examples": n_written,
        "rung_distribution": {str(k): v for k, v in sorted(rung_counts.items())},
        "filler_pool": len(pool_pairs),
        "gb_written": round(total_chars * 2.8 / 1e9, 2),
        "n_skipped_insufficient_filler": n_skipped_filler,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  wrote {n_written:,} examples -> {out_path} "
          f"({meta['gb_written']} GB, {meta['seconds']}s)", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a welded long-context training set.")
    ap.add_argument("--runs_root", default="outputs/translation/runs")
    ap.add_argument("--out_root", default="data/retrieval/beir_longctx_train/v1")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="default: every corpus with a hard_negatives_train.jsonl")
    ap.add_argument("--limit_per_dataset", type=int, default=None,
                    help="cap examples per dataset (for a quick pilot)")
    ap.add_argument("--position_bin", default="uniform",
                    choices=["uniform", "start", "middle", "end"],
                    help="uniform spreads the answer across the whole document")
    ap.add_argument("--filler_source", default=DEFAULT_FILLER_SOURCE,
                    help="neutral corpus for filler; must be disjoint from the eval sets")
    ap.add_argument("--max_rung", type=int, default=None,
                    help="drop rungs above this (cheaper training)")
    args = ap.parse_args()

    weights = dict(DEFAULT_RUNG_WEIGHTS)
    if args.max_rung:
        weights = {r: w for r, w in weights.items() if r <= args.max_rung}
        total = sum(weights.values())
        weights = {r: w / total for r, w in weights.items()}

    # Any corpus with training triples is fair game -- including nfcorpus, which is excluded
    # from the *eval* benchmark only because its filler pool is too small to build a corpus
    # from, a constraint that does not apply to training.
    candidates = args.datasets or ["BeIR_scifact", "BeIR_fiqa", "BeIR_nfcorpus", "BeIR_scidocs"]
    dirs = {}
    for ds in candidates:
        try:
            dirs.update(find_corpus_dirs(args.runs_root, [ds]))
        except FileNotFoundError:
            print(f"[skip] {ds}: no corpus found")

    print(f"rung distribution: { {k: round(v, 2) for k, v in sorted(weights.items())} }")
    print(f"mean length: {sum(r * w for r, w in weights.items()):,.0f} chars\n")

    pool_pairs = load_external_filler(args.filler_source)
    total_chars = sum(len(t) for _, t in pool_pairs)
    print(f"filler pool: {len(pool_pairs):,} passages, {total_chars:,} chars "
          f"from {args.filler_source}\n")
    if total_chars < 3 * max(weights) * 4:
        raise SystemExit("filler source too small to weld the largest rung")

    metas = []
    for ds, beir_dir in sorted(dirs.items(), key=lambda kv: os.path.getsize(
            os.path.join(kv[1], "corpus.jsonl"))):
        print(f"=== {ds} ===", flush=True)
        m = build_for_dataset(
            ds, beir_dir,
            os.path.join(args.out_root, ds, "welded_train.jsonl"),
            rung_weights=weights,
            pool_pairs=pool_pairs,
            limit=args.limit_per_dataset,
            position_bin=args.position_bin,
        )
        if m:
            metas.append(m)

    manifest = os.path.join(args.out_root, "manifest.json")
    os.makedirs(args.out_root, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"rung_weights": {str(k): v for k, v in weights.items()},
                   "position_bin": args.position_bin,
                   "datasets": metas}, fh, indent=2, ensure_ascii=False)
    print(f"\ntotal: {sum(m['n_examples'] for m in metas):,} examples, "
          f"{sum(m['gb_written'] for m in metas):.1f} GB -> {manifest}")


if __name__ == "__main__":
    main()

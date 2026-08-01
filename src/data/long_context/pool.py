"""
Safe-filler pools for the translated-BeIR long-context benchmark.

A welded document is one real passage surrounded by filler. Choosing that filler is not
free: **no passage that is a positive for any query may be used as filler anywhere.**

Why this is a hard constraint rather than a nicety. Suppose passage ``B`` is the gold for
query ``q``, and ``B`` gets welded inside document ``C``, which is not relevant to ``q``.
Then ``C`` now literally contains ``q``'s answer, but the qrels score it irrelevant. Every
model that correctly surfaces ``C`` is punished, and the metric measures noise.

This is what disqualified nfcorpus: it averages 38.2 positives per query, so 86% of its
corpus is a positive for something, leaving only 505 usable filler passages for 3,633
documents — roughly 165x reuse of the same text.

Availability after excluding positives from *all* qrels splits (measured):

    fiqa      57,600 docs -> 40,490 safe (70.3%)
    scidocs   25,313 docs -> 21,293 safe (84.1%)
    scifact    5,183 docs ->  4,516 safe (87.1%)

Excluding only the evaluated (test) split would leave more, but all-splits is affordable
here and stays correct if a dev/validation run is added later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Iterable, Sequence

#: Order matters only for reproducibility of the on-disk pool, not for correctness.
ALL_QREL_SPLITS: tuple[str, ...] = ("test", "train", "validation", "dev")

#: Datasets that can be welded cleanly. nfcorpus and arguana are excluded on purpose --
#: see the module docstring for nfcorpus; arguana's queries *are* corpus documents, so a
#: query's own text can land inside another document as filler and match it for free.
BENCHMARK_DATASETS: tuple[str, ...] = ("BeIR_scifact", "BeIR_scidocs", "BeIR_fiqa")


@dataclass(frozen=True)
class Passage:
    """A corpus passage, with text joined exactly as the eval joins it."""

    pid: str
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


def join_title_text(title: str | None, text: str | None) -> str:
    """Join title and body the same way the eval does.

    Mirrors ``_join_title_text`` in ``src/model/eval/eval_beir_retrieval_zeroshot.py``.
    Keeping these identical matters: the unpadded rung must reproduce the existing BeIR
    numbers exactly, which it cannot do if the document text differs by even a space.
    """
    parts = [p.strip() for p in [title or "", text or ""] if p and p.strip()]
    return " ".join(parts)


def find_corpus_dirs(
    runs_root: str = "outputs/translation/runs",
    datasets: Sequence[str] = BENCHMARK_DATASETS,
) -> dict[str, str]:
    """Locate each dataset's BeIR directory under the translation runs tree.

    Returns ``{dataset_name: beir_dir}``. If a dataset appears in more than one run, the
    most recently modified ``corpus.jsonl`` wins, and the choice is reported so it can be
    pinned in the manifest.
    """
    found: dict[str, tuple[float, str]] = {}
    pattern = os.path.join(runs_root, "*", "corpus", "*", "beir", "corpus.jsonl")
    for path in glob(pattern):
        beir_dir = os.path.dirname(path)
        name = os.path.basename(os.path.dirname(beir_dir))
        if name not in datasets:
            continue
        mtime = os.path.getmtime(path)
        if name not in found or mtime > found[name][0]:
            found[name] = (mtime, beir_dir)

    missing = [d for d in datasets if d not in found]
    if missing:
        raise FileNotFoundError(
            f"no BeIR corpus found for {missing} under {runs_root!r}"
        )
    return {name: d for name, (_, d) in sorted(found.items())}


def load_corpus(beir_dir: str) -> dict[str, str]:
    """Load ``corpus.jsonl`` as ``{doc_id: joined_text}``."""
    corpus: dict[str, str] = {}
    with open(os.path.join(beir_dir, "corpus.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            corpus[str(doc["_id"])] = join_title_text(doc.get("title"), doc.get("text"))
    return corpus


def load_queries(beir_dir: str) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(os.path.join(beir_dir, "queries.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    return queries


def load_qrels(beir_dir: str, split: str) -> dict[str, set[str]]:
    """Load one qrels split as ``{query_id: {positive_doc_id, ...}}``.

    Handles both the JSONL form (``query-id`` / ``corpus-id`` / ``score``) and the TSV
    form, since the translated corpora carry JSONL but upstream BeIR ships TSV.
    """
    out: dict[str, set[str]] = {}
    for ext, parse in (("jsonl", None), ("tsv", True)):
        path = os.path.join(beir_dir, "qrels", f"{split}.{ext}")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if parse:
                    if i == 0 and "query" in line.lower():
                        continue  # header
                    cols = line.split()
                    if len(cols) < 3:
                        continue
                    qid, cid, score = cols[0], cols[1], int(cols[2])
                else:
                    rec = json.loads(line)
                    qid = str(rec["query-id"])
                    cid = str(rec["corpus-id"])
                    score = int(rec.get("score", 0))
                if score > 0:
                    out.setdefault(qid, set()).add(cid)
        return out
    return out


def positive_doc_ids(
    beir_dir: str,
    splits: Iterable[str] = ALL_QREL_SPLITS,
) -> set[str]:
    """Every document that is a positive for any query in any of ``splits``.

    These are exactly the documents that must never appear as filler.
    """
    positives: set[str] = set()
    for split in splits:
        for docs in load_qrels(beir_dir, split).values():
            positives |= docs
    return positives


@dataclass
class PoolStats:
    dataset: str
    n_corpus: int
    n_positive: int
    n_too_short: int
    n_pool: int

    @property
    def pct_safe(self) -> float:
        return 100.0 * self.n_pool / max(1, self.n_corpus)

    def __str__(self) -> str:
        return (
            f"{self.dataset:16s} corpus={self.n_corpus:6,d} "
            f"positives={self.n_positive:6,d} short={self.n_too_short:5,d} "
            f"-> pool={self.n_pool:6,d} ({self.pct_safe:4.1f}%)"
        )


def build_safe_filler_pool(
    beir_dir: str,
    *,
    dataset: str = "",
    splits: Iterable[str] = ALL_QREL_SPLITS,
    min_chars: int = 100,
    max_chars: int | None = None,
) -> tuple[list[Passage], PoolStats]:
    """Build the filler pool for one dataset.

    ``max_chars`` is off by default: tape slicing already truncates long filler at a word
    boundary, so an over-long passage costs nothing but diversity. Set it if you want
    finer packing granularity.

    The pool is sorted by ``pid`` so the on-disk artifact is byte-stable across runs --
    the old builder collected pool entries in ``as_completed`` order, which made
    ``random.choice(pool)`` irreproducible even with a fixed seed.
    """
    corpus = load_corpus(beir_dir)
    positives = positive_doc_ids(beir_dir, splits)

    pool: list[Passage] = []
    n_short = 0
    for pid, text in corpus.items():
        if pid in positives:
            continue
        if len(text) < min_chars:
            n_short += 1
            continue
        if max_chars is not None and len(text) > max_chars:
            continue
        pool.append(Passage(pid=pid, text=text))

    pool.sort(key=lambda p: p.pid)
    stats = PoolStats(
        dataset=dataset or os.path.basename(os.path.dirname(beir_dir)),
        n_corpus=len(corpus),
        n_positive=len(positives),
        n_too_short=n_short,
        n_pool=len(pool),
    )
    return pool, stats


def save_pool(pool: Sequence[Passage], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for p in pool:
            fh.write(json.dumps({"pid": p.pid, "text": p.text}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic, so an interrupted write cannot leave a partial pool


def load_pool(path: str) -> list[Passage]:
    out: list[Passage] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out.append(Passage(pid=rec["pid"], text=rec["text"]))
    return out


def assert_pool_is_leakage_free(
    pool: Sequence[Passage],
    beir_dir: str,
    splits: Iterable[str] = ALL_QREL_SPLITS,
) -> None:
    """Fail loudly if any pooled passage is a positive for any query."""
    positives = positive_doc_ids(beir_dir, splits)
    offenders = [p.pid for p in pool if p.pid in positives]
    if offenders:
        raise AssertionError(
            f"{len(offenders)} filler passages are qrel positives "
            f"(e.g. {offenders[:5]}) -- this manufactures false negatives"
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build safe filler pools for BeIR long-context.")
    ap.add_argument("--runs_root", default="outputs/translation/runs")
    ap.add_argument("--out_root", default="data/retrieval/beir_longctx/v1")
    ap.add_argument("--datasets", nargs="*", default=list(BENCHMARK_DATASETS))
    ap.add_argument("--min_chars", type=int, default=100)
    ap.add_argument("--max_chars", type=int, default=None)
    ap.add_argument(
        "--splits",
        nargs="*",
        default=list(ALL_QREL_SPLITS),
        help="qrels splits whose positives are excluded from filler (default: all)",
    )
    args = ap.parse_args()

    dirs = find_corpus_dirs(args.runs_root, args.datasets)
    print(f"Excluding positives from splits: {', '.join(args.splits)}\n")
    for dataset, beir_dir in dirs.items():
        pool, stats = build_safe_filler_pool(
            beir_dir,
            dataset=dataset,
            splits=args.splits,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        assert_pool_is_leakage_free(pool, beir_dir, args.splits)
        out = os.path.join(args.out_root, dataset, "safe_filler_pool.jsonl")
        save_pool(pool, out)
        print(f"{stats}  src={beir_dir}")
        print(f"{'':16s} -> {out}")


if __name__ == "__main__":
    main()

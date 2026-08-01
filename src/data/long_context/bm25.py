"""
Frozen, model-independent BM25 for selecting welding filler.

Two things this module exists to guarantee.

**Model independence.** The earlier long-context construction
(``src/data/long_context/patch_documents.py``) selected each document's neighbours using
*the evaluated model's own embeddings*. That silently gave every model a different corpus,
so none of those numbers were comparable across models. BM25 depends on no model, and the
resulting neighbour lists are written to disk so a reviewer can audit exactly which
passages were used.

**Scale.** Every corpus document is welded, not just golds, so the ``bm25-coherent``
condition needs top-k neighbours for all documents -- up to 57,600 x 40,273 for fiqa.
``rank_bm25.BM25Okapi`` keeps a Python dict of term frequencies per document and scores one
query at a time; at this scale that is far too slow and memory-hungry. ``pyserini`` needs a
JVM and an Anserini index, and ``bm25s`` is not installed. So this is a direct sparse
implementation on ``scipy.sparse``: build the BM25 document-weight matrix once, then score
in row blocks with one sparse matmul per block.

Tokenization is deliberately shallow -- niqqud stripped, split on Hebrew/Latin/digit runs,
**no stemming and no prefix stripping**. Hebrew stemmers are lossy, and a naive
ו/ה/ב/כ/ל/מ/ש stripper corrupts real words. Either would bake an unauditable linguistic
choice into a selector whose whole purpose is to be neutral and inspectable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

#: Hebrew niqqud / cantillation. Stripped so vocalised and unvocalised spellings match.
NIQQUD_RE = re.compile(r"[֑-ׇ]")
#: Geresh / gershayim, used in Hebrew acronyms and abbreviations.
PUNCT_RE = re.compile(r"[׳״'\"]")
#: Runs of Hebrew letters, Latin letters, or digits.
TOKEN_RE = re.compile(r"[א-ת]+|[A-Za-z]+|\d+")

K1_DEFAULT = 1.5
B_DEFAULT = 0.75
#: Floor applied to non-positive IDF, as a multiple of the mean IDF. Matches
#: ``rank_bm25.BM25Okapi``'s default so this implementation is a drop-in equivalent.
EPSILON_DEFAULT = 0.25


def bm25_tokenize(text: str) -> list[str]:
    """Shallow Hebrew-aware tokenization. See the module docstring for why it is shallow."""
    text = NIQQUD_RE.sub("", text)
    text = PUNCT_RE.sub("", text)
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    """Sparse Okapi BM25 over a fixed document set.

    ``weights`` is the (n_docs x vocab) CSR matrix of precomputed BM25 document weights,
    so scoring a query is a single sparse matmul against a term-count vector.
    """

    doc_ids: tuple[str, ...]
    vocab: dict[str, int]
    weights: sp.csr_matrix
    k1: float = K1_DEFAULT
    b: float = B_DEFAULT

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    # ---------------------------------------------------------------- build --

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = K1_DEFAULT,
        b: float = B_DEFAULT,
        min_df: int = 2,
    ) -> "BM25Index":
        """Build the index. ``min_df=2`` drops hapax terms, which are pure noise for
        neighbour selection and account for a large share of a Hebrew vocabulary."""
        vocab: dict[str, int] = {}
        indptr = [0]
        indices: list[int] = []
        data: list[int] = []
        df_counter: dict[int, int] = {}

        for text in texts:
            counts: dict[int, int] = {}
            for tok in bm25_tokenize(text):
                tid = vocab.setdefault(tok, len(vocab))
                counts[tid] = counts.get(tid, 0) + 1
            for tid in counts:
                df_counter[tid] = df_counter.get(tid, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))

        n_docs, n_vocab = len(texts), len(vocab)
        tf = sp.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=(n_docs, n_vocab),
        )

        if min_df > 1:
            df = np.zeros(n_vocab, dtype=np.int64)
            for tid, c in df_counter.items():
                df[tid] = c
            keep = df >= min_df
            tf = tf[:, keep]
            remap = np.cumsum(keep) - 1
            vocab = {t: int(remap[i]) for t, i in vocab.items() if keep[i]}

        weights = cls._bm25_weights(tf, k1=k1, b=b)
        return cls(
            doc_ids=tuple(str(d) for d in doc_ids),
            vocab=vocab,
            weights=weights,
            k1=k1,
            b=b,
        )

    @staticmethod
    def _bm25_weights(
        tf: sp.csr_matrix, *, k1: float, b: float, epsilon: float = EPSILON_DEFAULT
    ) -> sp.csr_matrix:
        """Okapi BM25 document weights, computed on the CSR data array in place.

        w[d,t] = idf(t) * tf(d,t)*(k1+1) / (tf(d,t) + k1*(1 - b + b*len_d/avg_len))

        IDF is standard Robertson, ``ln((N - df + 0.5)/(df + 0.5))``, with non-positive
        values floored at ``epsilon * mean(idf)``. This reproduces
        ``rank_bm25.BM25Okapi`` exactly (verified to Spearman rho = 1.0 and identical
        top-k), which matters because the whole point of this selector is that it is
        standard, frozen and auditable rather than a bespoke variant.
        """
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = max(doc_len.mean(), 1e-9)

        df = np.diff(tf.tocsc().indptr)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # rank_bm25 averages over *all* terms (including the negative ones) before
        # flooring; reproduce that ordering rather than averaging the survivors.
        floor = np.float32(epsilon * float(idf.mean()))
        idf[idf <= 0] = floor

        w = tf.tocsr(copy=True)
        # per-row denominator component, broadcast onto the CSR data array
        row_norm = (k1 * (1.0 - b + b * doc_len / avg_len)).astype(np.float32)
        rows = np.repeat(np.arange(n_docs), np.diff(w.indptr))
        w.data = w.data * (k1 + 1.0) / (w.data + row_norm[rows])
        w.data = w.data * idf[w.indices]
        return w

    # ---------------------------------------------------------------- score --

    def _query_matrix(self, texts: Sequence[str]) -> sp.csr_matrix:
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for text in texts:
            counts: dict[int, int] = {}
            for tok in bm25_tokenize(text):
                tid = self.vocab.get(tok)
                if tid is not None:
                    counts[tid] = counts.get(tid, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))
        return sp.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(texts), self.weights.shape[1]),
        )

    def top_k(
        self,
        query_texts: Sequence[str],
        k: int,
        *,
        block: int = 2000,
        exclude: Sequence[set[int]] | None = None,
        progress: bool = False,
    ) -> list[list[tuple[int, float]]]:
        """Top-k documents per query, as ``[(doc_index, score), ...]`` descending.

        Scored in row blocks so peak memory is ``block x n_docs`` floats rather than
        ``n_queries x n_docs``. ``exclude[i]`` is a set of document indices barred from
        query ``i``'s results -- used to drop the document itself and any qrel positive.
        """
        results: list[list[tuple[int, float]]] = []
        n = len(query_texts)
        for start in range(0, n, block):
            stop = min(start + block, n)
            qm = self._query_matrix(query_texts[start:stop])
            scores = (qm @ self.weights.T).toarray()  # (block, n_docs)

            for r in range(scores.shape[0]):
                row = scores[r]
                banned = exclude[start + r] if exclude is not None else ()
                if banned:
                    idx = np.fromiter(banned, dtype=np.int64, count=len(banned))
                    row[idx] = -np.inf
                take = min(k, row.size)
                # argpartition is O(n); only the retained slice is sorted
                cand = np.argpartition(-row, take - 1)[:take]
                cand = cand[np.argsort(-row[cand])]
                results.append([(int(i), float(row[i])) for i in cand if row[i] > -np.inf])

            if progress:
                print(f"  bm25 top-{k}: {stop:,}/{n:,}", flush=True)
        return results

    # ----------------------------------------------------------------- i/o --

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.npz"
        sp.save_npz(tmp, self.weights)
        os.replace(tmp, f"{path}.npz")
        meta = {
            "doc_ids": list(self.doc_ids),
            "vocab": self.vocab,
            "k1": self.k1,
            "b": self.b,
        }
        tmp_meta = f"{path}.tmp.json"
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
        os.replace(tmp_meta, f"{path}.meta.json")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        weights = sp.load_npz(f"{path}.npz")
        with open(f"{path}.meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(
            doc_ids=tuple(meta["doc_ids"]),
            vocab=meta["vocab"],
            weights=weights,
            k1=meta["k1"],
            b=meta["b"],
        )


def self_test(n_docs: int = 800, n_probe: int = 40, verbose: bool = True) -> dict:
    """Assert this implementation reproduces ``rank_bm25.BM25Okapi`` exactly.

    Lives here rather than in ``tests/`` because the conda environments are split: only
    ``bert24`` has pytest, and only ``htr`` has scipy and rank_bm25, so a pytest module
    covering this would skip in both. Run with::

        conda activate htr && PYTHONPATH=./src python src/data/long_context/bm25.py --self-test

    Returns the measured agreement statistics so a caller can record them in a manifest.
    """
    import numpy as np
    from rank_bm25 import BM25Okapi
    from scipy.stats import spearmanr

    from data.long_context.pool import find_corpus_dirs, load_corpus

    beir_dir = find_corpus_dirs(datasets=["BeIR_scifact"])["BeIR_scifact"]
    corpus = load_corpus(beir_dir)
    ids = sorted(corpus)[:n_docs]
    texts = [corpus[i] for i in ids]

    ref = BM25Okapi([bm25_tokenize(t) for t in texts])
    mine = BM25Index.build(ids, texts, min_df=1)
    ours = mine.top_k(texts[:n_probe], k=10)

    overlaps, rhos, max_abs = [], [], 0.0
    for qi, q in enumerate(texts[:n_probe]):
        ref_scores = np.asarray(ref.get_scores(bm25_tokenize(q)))
        our_scores = (mine._query_matrix([q]) @ mine.weights.T).toarray().ravel()
        overlaps.append(
            len(set(np.argsort(-ref_scores)[:10]) & {i for i, _ in ours[qi]}) / 10
        )
        rhos.append(spearmanr(ref_scores, our_scores).correlation)
        max_abs = max(max_abs, float(np.abs(ref_scores - our_scores).max()))

    stats = {
        "top10_overlap_mean": float(np.mean(overlaps)),
        "top10_overlap_min": float(np.min(overlaps)),
        "spearman_mean": float(np.mean(rhos)),
        "spearman_min": float(np.min(rhos)),
        "max_abs_score_diff": max_abs,
    }
    if verbose:
        for k, v in stats.items():
            print(f"  {k:22s} {v:.6f}")

    assert stats["top10_overlap_min"] == 1.0, f"top-k disagrees with rank_bm25: {stats}"
    assert stats["spearman_min"] > 0.999999, f"ranking disagrees with rank_bm25: {stats}"
    assert max_abs < 1e-2, f"scores drift from rank_bm25: {stats}"
    return stats


def write_neighbours(
    path: str,
    seed_ids: Sequence[str],
    neighbours: Iterable[list[tuple[int, float]]],
    pool_ids: Sequence[str],
) -> None:
    """Persist the frozen neighbour lists so the selection is auditable."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for sid, hits in zip(seed_ids, neighbours):
            fh.write(
                json.dumps(
                    {
                        "seed_id": sid,
                        "neighbours": [
                            {"pid": pool_ids[i], "score": round(s, 4)} for i, s in hits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(tmp, path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="BM25 selector utilities.")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="assert equivalence with rank_bm25.BM25Okapi (needs the htr env)",
    )
    args = ap.parse_args()
    if args.self_test:
        print("BM25 equivalence vs rank_bm25.BM25Okapi:")
        self_test()
        print("OK - exact match")

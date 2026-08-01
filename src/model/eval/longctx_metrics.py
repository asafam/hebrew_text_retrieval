"""
Metrics for long-context retrieval, broken out by gold position.

Two requirements shape this module.

**Position bins are never averaged away.** A single number hides lost-in-the-middle, which is
one of the main effects the benchmark exists to measure, so ``by_position`` is a non-optional
key in the result rather than something a caller can forget to ask for.

**Confidence intervals are cluster bootstrap over gold documents, not per-query.** Queries
share gold documents (scidocs averages 4.9 positives per query, fiqa 2.6), so treating
queries as independent overstates precision. Model-vs-model deltas are paired on the same
resampled clusters, because the arms are evaluated on identical corpora and the paired
variance is much smaller than the difference of two independent intervals.

A third quantity is reported alongside the usual ones: ``gold_visible``-conditioned scores.
A model whose window cannot reach the gold passage scores zero for reasons of capacity, not
ranking quality. Reporting the score restricted to the subset a model can actually see is
what distinguishes an honest capacity finding from a strawman.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

DEFAULT_KS = (1, 5, 10, 100)


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(ranked_rel: Sequence[float], all_rel: Sequence[float], k: int) -> float:
    """nDCG@k with the ideal DCG taken over the **full** qrels, not the retrieved slice.

    Normalising against the retrieved slice inflates the score for a system that retrieved
    few relevant documents, which is the kind of quiet bias that makes numbers
    non-comparable with published BeIR results.
    """
    dcg = _dcg(list(ranked_rel)[:k])
    ideal = _dcg(sorted(all_rel, reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0


@dataclass
class QueryResult:
    """Per-query outcome, retained so metrics can be re-averaged over any subset."""

    query_id: str
    cluster_id: str
    rank_of_first_relevant: int | None
    ranked_relevance: list[float]
    all_relevance: list[float]
    gold_visible: bool = True
    position_bin: str = "unknown"

    def metrics(self, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
        r = self.rank_of_first_relevant
        out: dict[str, float] = {
            "success_at_1": 1.0 if r == 1 else 0.0,
            "mrr_at_10": 1.0 / r if r is not None and r <= 10 else 0.0,
        }
        n_rel = sum(1 for g in self.all_relevance if g > 0)
        for k in ks:
            found = sum(1 for g in self.ranked_relevance[:k] if g > 0)
            out[f"recall_at_{k}"] = found / n_rel if n_rel else 0.0
        for k in (10, 100):
            out[f"ndcg_at_{k}"] = _ndcg_at_k(self.ranked_relevance, self.all_relevance, k)
        return out


def _mean_metrics(results: Sequence[QueryResult], ks: Sequence[int]) -> dict[str, float]:
    if not results:
        return {"num_queries_evaluated": 0}
    acc: dict[str, float] = {}
    for r in results:
        for key, val in r.metrics(ks).items():
            acc[key] = acc.get(key, 0.0) + val
    out = {k: v / len(results) for k, v in acc.items()}
    out["num_queries_evaluated"] = len(results)
    return out


def cluster_bootstrap_ci(
    results: Sequence[QueryResult],
    metric: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    ks: Sequence[int] = DEFAULT_KS,
) -> tuple[float, float]:
    """95% CI for ``metric``, resampling **gold-document clusters** with replacement.

    Resampling queries instead would understate the interval, because several queries can
    share a gold document and therefore are not independent observations.
    """
    import random

    by_cluster: dict[str, list[float]] = {}
    for r in results:
        by_cluster.setdefault(r.cluster_id, []).append(r.metrics(ks)[metric])
    clusters = list(by_cluster.values())
    if len(clusters) < 2:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    means = []
    n = len(clusters)
    for _ in range(n_boot):
        vals: list[float] = []
        for _ in range(n):
            vals.extend(clusters[rng.randrange(n)])
        means.append(sum(vals) / len(vals))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_cluster_bootstrap_delta(
    results_a: Sequence[QueryResult],
    results_b: Sequence[QueryResult],
    metric: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """CI for ``mean(a) - mean(b)``, resampling the same clusters for both arms.

    Pairing matters: both arms see an identical corpus, so most of the cluster-to-cluster
    variance is shared and cancels. An unpaired comparison would call a real difference
    insignificant.
    """
    import random

    a_by: dict[str, list[float]] = {}
    b_by: dict[str, list[float]] = {}
    for r in results_a:
        a_by.setdefault(r.cluster_id, []).append(r.metrics(ks)[metric])
    for r in results_b:
        b_by.setdefault(r.cluster_id, []).append(r.metrics(ks)[metric])
    shared = sorted(set(a_by) & set(b_by))
    if len(shared) < 2:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = random.Random(seed)
    deltas = []
    n = len(shared)
    for _ in range(n_boot):
        av: list[float] = []
        bv: list[float] = []
        for _ in range(n):
            c = shared[rng.randrange(n)]
            av.extend(a_by[c])
            bv.extend(b_by[c])
        deltas.append(sum(av) / len(av) - sum(bv) / len(bv))
    deltas.sort()
    mean_a = sum(v for vs in a_by.values() for v in vs) / sum(len(v) for v in a_by.values())
    mean_b = sum(v for vs in b_by.values() for v in vs) / sum(len(v) for v in b_by.values())
    return {
        "delta": mean_a - mean_b,
        "ci_low": deltas[int((alpha / 2) * n_boot)],
        "ci_high": deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))],
        "n_clusters": float(n),
    }


@dataclass
class MetricReport:
    overall: dict[str, float] = field(default_factory=dict)
    by_position: dict[str, dict[str, float]] = field(default_factory=dict)
    gold_visible_only: dict[str, float] = field(default_factory=dict)
    gold_hidden_only: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "by_position": self.by_position,
            "gold_visible_only": self.gold_visible_only,
            "gold_hidden_only": self.gold_hidden_only,
            "uncertainty": self.uncertainty,
        }


def compute_report(
    results: Sequence[QueryResult],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    ci_metric: str = "ndcg_at_10",
    n_boot: int = 1000,
    seed: int = 0,
) -> MetricReport:
    """Overall + per-bin + visibility-conditioned metrics, with a cluster-bootstrap CI."""
    rep = MetricReport()
    rep.overall = _mean_metrics(results, ks)

    bins = sorted({r.position_bin for r in results})
    for b in bins:
        rep.by_position[b] = _mean_metrics([r for r in results if r.position_bin == b], ks)

    visible = [r for r in results if r.gold_visible]
    hidden = [r for r in results if not r.gold_visible]
    rep.gold_visible_only = _mean_metrics(visible, ks)
    rep.gold_hidden_only = _mean_metrics(hidden, ks)

    lo, hi = cluster_bootstrap_ci(results, ci_metric, n_boot=n_boot, seed=seed, ks=ks)
    rep.uncertainty = {
        "method": "cluster_bootstrap_over_gold_docs",
        "metric": ci_metric,
        "n_boot": n_boot,
        "seed": seed,
        "n_clusters": len({r.cluster_id for r in results}),
        "n_queries": len(results),
        "ci95": [lo, hi],
    }
    return rep


def build_query_results(
    query_ids: Sequence[str],
    ranked_doc_ids: Sequence[Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    gold_visible: Mapping[str, bool] | None = None,
    position_bin: Mapping[str, str] | None = None,
) -> list[QueryResult]:
    """Turn ranked doc-id lists into per-query results.

    ``cluster_id`` is the query's lowest-sorted positive document id: queries sharing a gold
    land in the same bootstrap cluster, which is what makes the CI honest.
    """
    out: list[QueryResult] = []
    for qid, ranked in zip(query_ids, ranked_doc_ids):
        rel_map = qrels.get(qid, {})
        if not rel_map:
            continue
        ranked_rel = [float(rel_map.get(d, 0)) for d in ranked]
        first = next((i + 1 for i, g in enumerate(ranked_rel) if g > 0), None)
        cluster = min(rel_map)
        out.append(
            QueryResult(
                query_id=qid,
                cluster_id=cluster,
                rank_of_first_relevant=first,
                ranked_relevance=ranked_rel,
                all_relevance=[float(v) for v in rel_map.values()],
                gold_visible=True if gold_visible is None else gold_visible.get(qid, True),
                position_bin="unknown" if position_bin is None else position_bin.get(qid, "unknown"),
            )
        )
    return out

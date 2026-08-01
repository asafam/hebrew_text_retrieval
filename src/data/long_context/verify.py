"""
Build-time verification for the long-context benchmark.

Every check here corresponds to a specific, measured defect in the previous builder. The
point is that a corpus which would reproduce one of those defects cannot be written without
failing the build:

  #1 gold clipped by a final truncation (71% of 512-token rows, 52% at 1024, 32% at 2048)
  #2 unclamped offsets producing negative gold spans (4% of 512-rows)
  #3 gold position collapsed to offset 0 (53% of rows) while documented as "random"
  #4 text round-tripped through a tokenizer, destroying newlines
  #5 passages joined with "" -- no separator, text running together mid-sentence
  #6 length budgeted in one model's tokens, silently calibrating the corpus to mE5
  #7 corpus size varying across sizes (447K -> 556K), confounding length with corpus size
  #8 filler containing a qrel positive, manufacturing false negatives

Run after each dataset is built. Non-zero exit on any failure; nothing is a warning.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence


class VerificationError(AssertionError):
    """A built corpus violates an invariant. Never downgrade this to a warning."""


@dataclass
class Report:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    def __str__(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if self.failed:
            names = ", ".join(n for n, _, _ in self.failed)
            raise VerificationError(f"{len(self.failed)} check(s) failed: {names}")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ------------------------------------------------------------------ document --


def check_gold_intact(records: Sequence[dict], source_text: dict[str, str], rep: Report) -> None:
    """#1, #2: the gold passage is present, complete, and at its recorded offsets.

    This is the check the previous builder failed on the majority of its rows.
    """
    bad_slice, bad_span, missing = 0, 0, 0
    for r in records:
        text = r["text"]
        a, b = r["gold_char_start"], r["gold_char_end"]
        gold = source_text.get(r["seed_doc_id"])
        if gold is None:
            missing += 1
            continue
        if not (0 <= a < b <= len(text)):
            bad_span += 1
            continue
        if text[a:b] != gold:
            bad_slice += 1
    n = max(1, len(records))
    rep.add(
        "gold intact at recorded offsets",
        bad_slice == 0 and bad_span == 0 and missing == 0,
        f"{bad_slice} mis-sliced, {bad_span} bad spans, {missing} unknown seeds of {n}",
    )


#: Minimum passage length for the uniqueness check. Mirrors ``MIN_CONTAINMENT_CHARS`` in
#: ``build_benchmark.py`` -- a check should assert exactly what the builder guarantees, and the
#: builder only excludes containment-duplicating filler for passages at or above this length.
#:
#: Below it, a repeat occurrence is ordinary language rather than leakage. Measured on fiqa, the
#: only two offenders in 57,600 documents are the passages ``'מקור'`` ("source", 4 chars) and
#: ``'בהצלחה!'`` ("good luck!", 7 chars); neither is a test gold, and neither is an answer, so a
#: filler passage containing them tells no query anything. Enforcing uniqueness there would mean
#: filtering out most Hebrew text as filler, which starves the tapes and produced a
#: zero-character document when it was tried.
GOLD_UNIQUE_MIN_CHARS = 64


def check_gold_unique(records: Sequence[dict], source_text: dict[str, str], rep: Report) -> None:
    """A duplicated gold string would make the recorded offsets ambiguous.

    Scoped to substantive passages; see ``GOLD_UNIQUE_MIN_CHARS``. Note that the offsets
    themselves are verified unconditionally by ``check_gold_intact`` -- that is the invariant
    the eval actually depends on, since retrieval is by document id and the offsets are used
    only to locate the gold's token span.
    """
    dups, short_dups = 0, 0
    for r in records:
        g = source_text.get(r["seed_doc_id"])
        if not g or r["text"].count(g) == 1:
            continue
        if len(g) >= GOLD_UNIQUE_MIN_CHARS:
            dups += 1
        else:
            short_dups += 1
    detail = f"{dups} documents with != 1 occurrence"
    if short_dups:
        detail += (
            f" (plus {short_dups} with a passage under {GOLD_UNIQUE_MIN_CHARS} chars, "
            f"not enforced -- ordinary language, not leakage)"
        )
    rep.add("gold appears exactly once", dups == 0, detail)


#: Minimum fill fraction for a padded document.
#:
#: Word-boundary snapping loses part of the final filler slice, and the loss is a larger
#: *fraction* of a small budget. Measured on scidocs: at the smallest rung (3,700 chars) 14 of
#: 25,140 padded documents (0.056%) land below 0.98, worst case 0.966 -- i.e. 124 characters
#: short -- while every larger rung has none. The gold is intact and the budget is never
#: exceeded, so this is a rounding artifact, not a welding defect. 0.96 still catches a real
#: regression (a half-filled document) while not failing on the artifact.
MIN_FILL_FRAC = 0.96


def check_lengths(records: Sequence[dict], budget: int, rep: Report) -> None:
    """Padded documents fill the budget; pass-through documents are the only exceptions."""
    padded = [r for r in records if r.get("padded", True)]
    over = [r for r in padded if len(r["text"]) > budget]
    under = [r for r in padded if len(r["text"]) < MIN_FILL_FRAC * budget]
    n_pt = len(records) - len(padded)
    worst = min((len(r["text"]) / budget for r in padded), default=1.0)
    rep.add(
        "padded docs respect the budget",
        not over and not under,
        f"{len(over)} over, {len(under)} below {MIN_FILL_FRAC:.2f} fill "
        f"(worst {worst:.3f}); {n_pt} pass-through "
        f"({100 * n_pt / max(1, len(records)):.2f}%)",
    )


def check_separator(records: Sequence[dict], separator: str, rep: Report) -> None:
    """#5: passages must be separated, not concatenated."""
    missing = sum(
        1 for r in records if r.get("padded", True) and separator not in r["text"]
    )
    rep.add(
        "separator present between passages",
        missing == 0,
        f"{missing} padded documents contain no separator",
    )


def check_raw_text_preserved(records: Sequence[dict], rep: Report) -> None:
    """#4: no tokenizer round-trip, so real whitespace survives."""
    with_newline = sum(1 for r in records if "\n" in r["text"])
    rep.add(
        "raw whitespace preserved",
        with_newline > 0,
        f"{with_newline}/{len(records)} documents contain a newline",
    )


def check_position_distribution(records: Sequence[dict], rep: Report) -> None:
    """#3: gold position must be genuinely spread, not collapsed to the start."""
    fracs = [
        r["gold_char_start"] / max(1, len(r["text"]) - (r["gold_char_end"] - r["gold_char_start"]))
        for r in records
        if r.get("padded", True)
    ]
    if not fracs:
        rep.add("gold position distribution", False, "no padded documents to check")
        return
    at_zero = sum(1 for f in fracs if f < 0.01) / len(fracs)
    quartiles = [
        sum(1 for f in fracs if lo <= f < lo + 0.25) / len(fracs)
        for lo in (0.0, 0.25, 0.5, 0.75)
    ]
    ok = at_zero < 0.10 and all(q > 0.10 for q in quartiles)
    rep.add(
        "gold position not collapsed to 0",
        ok,
        f"{at_zero:.1%} at offset 0 (old builder: 53%); "
        f"quartiles {'/'.join(f'{q:.0%}' for q in quartiles)}",
    )


def check_position_bins_balanced(records: Sequence[dict], rep: Report) -> None:
    counts = Counter(r.get("position_bin") for r in records)
    total = sum(counts.values())
    ok = total > 0 and all(0.28 < c / total < 0.39 for c in counts.values())
    rep.add(
        "position bins balanced",
        ok,
        ", ".join(f"{b}={c / max(1, total):.0%}" for b, c in sorted(counts.items())),
    )


# -------------------------------------------------------------------- corpus --


def check_corpus_size_constant(counts_by_rung: dict[int, int], rep: Report) -> None:
    """#7: document count must not vary across rungs."""
    uniq = set(counts_by_rung.values())
    rep.add(
        "corpus size constant across rungs",
        len(uniq) == 1,
        ", ".join(f"{r}={c:,}" for r, c in sorted(counts_by_rung.items())),
    )


def check_ids_unchanged(built_ids: Iterable[str], source_ids: Iterable[str], rep: Report) -> None:
    """Welding changes document *text*, never document *identity* -- that is what keeps the
    original qrels valid without modification."""
    b, s = set(built_ids), set(source_ids)
    rep.add(
        "document id set unchanged from source",
        b == s,
        f"{len(b - s)} added, {len(s - b)} dropped",
    )


def check_qrels_reachable(
    qrels: dict[str, set[str]], built_ids: Iterable[str], rep: Report,
    source_ids: Iterable[str] | None = None,
) -> None:
    """Welding must not make any query unanswerable that was answerable in the source.

    The absolute count is the wrong test: translated arguana ships 5 of 1,406 queries whose
    gold is absent from its own corpus, and failing the build for that would reject a dataset
    over a defect it inherited. What matters is the delta -- a query answerable in the source
    and unanswerable after welding means welding dropped a gold, which is a real bug.
    """
    ids = set(built_ids)
    dead = {q for q, docs in qrels.items() if not (docs & ids)}
    if source_ids is None:
        rep.add("every query has a reachable gold", not dead,
                f"{len(dead)} unanswerable queries of {len(qrels)}")
        return
    src = set(source_ids)
    dead_src = {q for q, docs in qrels.items() if not (docs & src)}
    introduced = dead - dead_src
    rep.add(
        "welding introduced no unanswerable queries",
        not introduced,
        f"{len(introduced)} introduced; {len(dead_src)} already unanswerable in source "
        f"(of {len(qrels)})",
    )


def check_filler_leakage_free(
    records: Sequence[dict], positives: set[str], rep: Report
) -> None:
    """#8: no qrel positive may be used as filler.

    A positive welded into an irrelevant document makes that document genuinely contain the
    answer while the qrels score it irrelevant -- a manufactured false negative that
    penalises correct retrieval.
    """
    offenders = 0
    for r in records:
        used = set(r.get("distractor_ids") or ())
        if used & positives:
            offenders += 1
    rep.add(
        "no qrel positive used as filler",
        offenders == 0,
        f"{offenders} documents contain a positive as filler",
    )


def check_nesting(
    docs_by_rung: dict[int, dict[str, str]], rep: Report, sample: int = 500
) -> None:
    """Infix nesting: doc(smaller rung) is an exact contiguous substring of doc(larger)."""
    rungs = sorted(docs_by_rung)
    if len(rungs) < 2:
        rep.add("infix nesting across rungs", True, "single rung, nothing to compare")
        return
    ids = sorted(docs_by_rung[rungs[0]])[:sample]
    failures = 0
    for small, large in zip(rungs, rungs[1:]):
        for did in ids:
            a, b = docs_by_rung[small].get(did), docs_by_rung[large].get(did)
            if a and b and a not in b:
                failures += 1
    rep.add(
        "infix nesting across rungs",
        failures == 0,
        f"{failures} violations over {len(ids)} docs x {len(rungs) - 1} rung pairs",
    )


def check_token_headroom(
    model: str,
    rung: int,
    p99_tokens: float,
    overflow_frac: float,
    limit: int,
    native_through: int,
    max_overflow: float,
    rep: Report,
) -> None:
    """#6 corollary: a model the design calls *native* at a rung must actually fit there.

    Beyond ``native_through`` the model is expected to chunk, so overflow is the measured
    condition rather than a failure.
    """
    if rung > native_through:
        rep.add(
            f"{model} @ {rung}ch (chunked by design)",
            True,
            f"p99={p99_tokens:.0f} limit={limit} overflow={overflow_frac:.2%} - expected",
        )
        return
    ok = p99_tokens <= 0.95 * limit and overflow_frac <= max_overflow
    rep.add(
        f"{model} @ {rung}ch native headroom",
        ok,
        f"p99={p99_tokens:.0f} vs 0.95*{limit}={0.95 * limit:.0f}, "
        f"overflow={overflow_frac:.2%} (max {max_overflow:.0%})",
    )


def check_weld_is_tokenizer_free(rep: Report) -> None:
    """#6: the welding core must not import a tokenizer, or length budgeting could drift
    back to being model-specific."""
    from data.long_context import weld as weld_mod

    with open(weld_mod.__file__, encoding="utf-8") as fh:
        src = fh.read()
    banned = [b for b in ("transformers", "AutoTokenizer", "sentencepiece", "tiktoken") if b in src]
    rep.add("weld.py imports no tokenizer", not banned, f"found: {banned}" if banned else "clean")


def self_test(verbose: bool = True) -> None:
    """Prove each check fires on the defect it targets.

    A verifier that has only ever been run against correct data is worth very little, so
    this builds a small valid corpus, confirms every check passes, then reintroduces each
    original bug one at a time and asserts the corresponding check fails. Runs on synthetic
    strings -- no data files, no models, no tokenizer.
    """
    import copy
    import random

    from data.long_context.rng import record_rng
    from data.long_context.weld import build_tapes, sample_position_frac, weld

    gold_texts = {f"d{i}": f"פסקת הזהב מספר {i} " + "מילה " * 30 for i in range(60)}
    filler = [(f"f{j}", " ".join(f"מלה{j}-{w}" for w in range(120))) for j in range(400)]
    positives = set(gold_texts)  # every seed here plays the role of a gold
    qrels = {f"q{i}": {f"d{i}"} for i in range(60)}
    budget = 4000

    def build(bins=("start", "middle", "end")) -> list[dict]:
        out = []
        for i, (did, gold) in enumerate(gold_texts.items()):
            rng = record_rng(did, "random")
            seq = list(filler)
            rng.shuffle(seq)
            tapes = build_tapes(seq, min_tape_chars=budget + 64)
            bin_name = bins[i % len(bins)]
            r = weld(gold, tapes, budget, sample_position_frac(rng, "uniform"))
            out.append(
                {
                    "_id": did,
                    "doc_id_base": did,
                    "seed_doc_id": did,
                    "text": r.text,
                    "gold_char_start": r.gold_char_start,
                    "gold_char_end": r.gold_char_end,
                    "padded": r.padded,
                    "position_bin": bin_name,
                    "distractor_ids": list(r.distractor_ids[:8]),
                }
            )
        return out

    def run(records) -> Report:
        rep = Report()
        check_gold_intact(records, gold_texts, rep)
        check_gold_unique(records, gold_texts, rep)
        check_lengths(records, budget, rep)
        check_separator(records, "\n\n", rep)
        check_raw_text_preserved(records, rep)
        check_position_distribution(records, rep)
        check_position_bins_balanced(records, rep)
        check_filler_leakage_free(records, positives, rep)
        check_ids_unchanged((r["_id"] for r in records), gold_texts.keys(), rep)
        check_qrels_reachable(qrels, (r["_id"] for r in records), rep,
                              source_ids=gold_texts.keys())
        return rep

    base = build()
    clean = run(base)
    if verbose:
        print("baseline (must be all PASS):")
        print(clean)
    clean.raise_if_failed()

    # Each mutation reintroduces one original defect; the named check must fail.
    def mutate_clip(recs):
        for r in recs:
            r["text"] = r["text"][: budget - 600]  # the #1 final-truncation clip
        return recs

    def mutate_no_sep(recs):
        for r in recs:
            r["text"] = r["text"].replace("\n\n", "")
        return recs

    def mutate_collapse_position(recs):
        for r in recs:
            g = gold_texts[r["seed_doc_id"]]
            r["text"] = g + "\n\n" + r["text"].replace(g, "", 1)
            r["gold_char_start"], r["gold_char_end"] = 0, len(g)
        return recs

    def mutate_positive_as_filler(recs):
        for r in recs:
            r["distractor_ids"] = list(r["distractor_ids"]) + ["d7"]
        return recs

    def mutate_drop_doc(recs):
        return recs[:-5]

    cases = [
        ("gold intact at recorded offsets", mutate_clip),
        ("separator present between passages", mutate_no_sep),
        ("gold position not collapsed to 0", mutate_collapse_position),
        ("no qrel positive used as filler", mutate_positive_as_filler),
        ("document id set unchanged from source", mutate_drop_doc),
    ]
    if verbose:
        print("\nmutations (each must trip its check):")
    for expected, fn in cases:
        rep = run(fn(copy.deepcopy(base)))
        failed = {n for n, _, _ in rep.failed}
        status = "caught" if expected in failed else "MISSED"
        if verbose:
            print(f"  [{status}] {expected:44s} (also tripped: "
                  f"{sorted(failed - {expected}) or 'none'})")
        assert expected in failed, f"{expected!r} did not fire; failures were {failed}"

    if verbose:
        print("\nOK - every check fires on the defect it targets")


def verify_dataset_dir(
    root: str,
    condition: str,
    source_text: dict[str, str],
    positives: set[str],
    qrels: dict[str, set[str]],
    *,
    separator: str = "\n\n",
    rungs: Sequence[int] = (),
) -> Report:
    """Verify one dataset/condition tree. Expects the layout written by build_benchmark.py."""
    rep = Report()
    check_weld_is_tokenizer_free(rep)

    counts: dict[int, int] = {}
    docs_by_rung: dict[int, dict[str, str]] = {}
    for rung in rungs:
        rung_dir = os.path.join(root, condition, f"c{rung}")
        # One eval run loads nongold.jsonl plus exactly ONE gold position shard, so the
        # document count per run equals the source corpus size. Verify that composition
        # rather than the union of all shards, which would triple-count the golds.
        shards = [
            os.path.join(rung_dir, f)
            for f in ("nongold.jsonl", "gold_start.jsonl")
            if os.path.exists(os.path.join(rung_dir, f))
        ]
        if not shards:
            rep.add(f"rung c{rung} present", False, f"no shards under {rung_dir}")
            continue
        records = [r for s in shards for r in _read_jsonl(s)]
        counts[rung] = len(records)
        docs_by_rung[rung] = {r["doc_id_base"]: r["text"] for r in records if "doc_id_base" in r}

        check_gold_intact(records, source_text, rep)
        check_gold_unique(records, source_text, rep)
        check_lengths(records, rung, rep)
        check_separator(records, separator, rep)
        check_raw_text_preserved(records, rep)
        check_position_distribution(records, rep)
        check_filler_leakage_free(records, positives, rep)
        check_ids_unchanged((r["_id"] for r in records), source_text.keys(), rep)
        check_qrels_reachable(qrels, (r["_id"] for r in records), rep,
                              source_ids=source_text.keys())

    if counts:
        check_corpus_size_constant(counts, rep)
    if docs_by_rung:
        check_nesting(docs_by_rung, rep)
    return rep


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Long-context benchmark verification.")
    ap.add_argument("--self-test", action="store_true",
                    help="prove each check fires on the defect it targets")
    ap.add_argument("--dataset_dir", default=None,
                    help="verify a built dataset tree, e.g. data/retrieval/beir_longctx/v1/BeIR_scifact")
    ap.add_argument("--condition", default="random", choices=["random", "bm25"])
    args = ap.parse_args()

    if args.self_test:
        self_test()

    if args.dataset_dir:
        from data.long_context.calibrate import LADDER_CHARS
        from data.long_context.pool import (ALL_QREL_SPLITS, load_corpus, load_qrels,
                                            positive_doc_ids)

        with open(os.path.join(args.dataset_dir, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        beir_dir = manifest["source_beir_dir"]
        rungs = manifest.get("rungs_chars", list(LADDER_CHARS))

        report = verify_dataset_dir(
            args.dataset_dir,
            args.condition,
            source_text=load_corpus(beir_dir),
            positives=positive_doc_ids(beir_dir, ALL_QREL_SPLITS),
            qrels=load_qrels(beir_dir, "test"),
            separator=manifest.get("separator", "\n\n"),
            rungs=rungs,
        )
        print(f"{args.dataset_dir}  condition={args.condition}")
        print(report)
        report.raise_if_failed()
        print(f"\nOK - {len(report.checks)} checks passed")

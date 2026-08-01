"""
Invariant tests for long-context document welding.

Each test corresponds to a specific defect in the previous builder
(``src/data/datasets/build_long_context_dataset.py``), measured on its output:

  * gold clipped by the final truncation in 71% of 512-token rows, 52% at 1024,
    32% at 2048; entirely absent (start >= window, negative span) in 4% of 512-rows
  * 53% of rows placed the gold at offset 0 while documenting the position as "random"
  * the same document received different distractors and a different gold position at
    every context size, so sizes were not comparable
  * passages were joined with "".join, running text together mid-sentence

No data files, no models, no network.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.long_context.rng import record_rng, seed64, stable_bin
from data.long_context.weld import (
    GoldTooLongError,
    POSITION_BINS,
    Tapes,
    build_tapes,
    sample_position_frac,
    snap_head,
    snap_tail,
    weld,
)

GOLD = "שאלה זו היא הפסקה הזהב שצריך לאחזר מתוך המסמך הארוך."
LADDER = (1850, 3700, 7400, 14800, 22200, 29600)


def _pool(n: int = 4000, seed: int = 7) -> list[tuple[str, str]]:
    """Synthetic distractor passages with realistic word-ish structure."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        n_words = rng.randint(40, 160)
        words = ["מילה%d" % rng.randint(0, 9999) for _ in range(n_words)]
        out.append((f"p{i:05d}", " ".join(words)))
    return out


def _tapes(doc_id: str = "doc-1", condition: str = "random") -> Tapes:
    rng = record_rng(doc_id, condition)
    pool = _pool()
    rng.shuffle(pool)
    return build_tapes(pool, min_tape_chars=max(LADDER) + 64)


# --------------------------------------------------------------------------- #
# Invariant 1: the gold is always present, complete, and at its recorded offsets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budget", LADDER)
@pytest.mark.parametrize("bin_name", ["start", "middle", "end", "uniform"])
def test_gold_intact_at_recorded_offsets(budget, bin_name):
    tapes = _tapes()
    rng = random.Random(seed64(budget, bin_name))
    frac = sample_position_frac(rng, bin_name)
    r = weld(GOLD, tapes, budget, frac)

    assert r.text[r.gold_char_start : r.gold_char_end] == GOLD
    assert 0 <= r.gold_char_start < r.gold_char_end <= r.n_chars
    assert r.gold_char_end - r.gold_char_start == len(GOLD)
    assert GOLD in r.text


@pytest.mark.parametrize("budget", LADDER)
def test_gold_appears_exactly_once(budget):
    """A duplicated gold string would make the recorded offsets ambiguous."""
    tapes = _tapes()
    r = weld(GOLD, tapes, budget, 0.5)
    assert r.text.count(GOLD) == 1


def test_budget_is_respected():
    tapes = _tapes()
    for budget in LADDER:
        r = weld(GOLD, tapes, budget, 0.5)
        assert r.n_chars <= budget, f"{r.n_chars} > {budget}"
        assert r.n_chars >= 0.98 * budget, f"{r.n_chars} under-fills {budget}"


def test_oversized_gold_raises_when_passthrough_disabled():
    """The old builder computed a negative slice bound and silently cut the gold."""
    tapes = _tapes()
    with pytest.raises(GoldTooLongError):
        weld("x" * 500, tapes, 100, 0.5, allow_passthrough=False)


def test_oversized_passage_passes_through_unpadded():
    """A passage already longer than the rung is emitted whole, never clipped.

    In translated BeIR, 3.7-6.0% of passages exceed 1,850 chars and fiqa's longest is
    13,748 — so this path must keep the document count constant across rungs.
    """
    tapes = _tapes()
    long_passage = " ".join(f"tok{i}" for i in range(400))  # ~2.7k chars
    r = weld(long_passage, tapes, 1000, 0.5)

    assert r.padded is False
    assert r.text == long_passage
    assert r.text[r.gold_char_start : r.gold_char_end] == long_passage
    assert r.n_chars > 1000  # deliberately over budget rather than truncated


def test_passthrough_never_loses_a_document():
    """Every document yields a result at every rung — corpus size cannot drift."""
    tapes = _tapes()
    lengths = [200, 900, 2600, 8000, 14000]
    for budget in LADDER:
        for n in lengths:
            passage = " ".join(f"w{i}" for i in range(n // 4))
            r = weld(passage, tapes, budget, 0.5)
            assert r.text[r.gold_char_start : r.gold_char_end] == passage
            assert r.padded == (len(passage) + 4 <= budget)


def test_padded_documents_still_respect_the_budget():
    tapes = _tapes()
    short = "קצר " * 20
    for budget in LADDER:
        r = weld(short, tapes, budget, 0.5)
        assert r.padded is True
        assert r.n_chars <= budget


def test_gold_never_pushed_past_the_end_cap():
    tapes = _tapes()
    for budget in LADDER:
        r = weld(GOLD, tapes, budget, 1.0)  # 'end' bin, maximum left fill
        assert r.gold_char_end <= budget
        assert r.gold_char_end <= 0.98 * budget + len(GOLD)


# --------------------------------------------------------------------------- #
# Invariant 2: infix nesting across the ladder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bin_name", ["start", "middle", "end"])
def test_infix_nesting_across_adjacent_rungs(bin_name):
    """doc(S-1) must be an exact contiguous substring of doc(S)."""
    tapes = _tapes()
    rng = random.Random(seed64("nest", bin_name))
    frac = sample_position_frac(rng, bin_name)

    docs = [weld(GOLD, tapes, b, frac).text for b in LADDER]
    for small, large in zip(docs, docs[1:]):
        assert small in large, (
            f"{bin_name}: {len(small)}-char doc is not an infix of the "
            f"{len(large)}-char doc"
        )


@pytest.mark.parametrize("bin_name", ["start", "middle", "end"])
def test_infix_nesting_on_a_dense_ladder(bin_name):
    """The six real rungs are far apart, so they can satisfy nesting by luck even when
    the underlying snap is not monotone. Check a dense sweep as well."""
    tapes = _tapes()
    rng = random.Random(seed64("dense", bin_name))
    frac = sample_position_frac(rng, bin_name)

    budgets = list(range(2000, 12000, 97))
    docs = [weld(GOLD, tapes, b, frac).text for b in budgets]
    for (bs, small), (bl, large) in zip(zip(budgets, docs), zip(budgets[1:], docs[1:])):
        assert small in large, f"{bin_name}: doc({bs}) is not an infix of doc({bl})"


def test_snap_falls_back_to_hard_cut_on_long_unbroken_tokens():
    """A long unbroken token must not cause most of the slice to be discarded.

    fiqa contains URLs and figures with no whitespace for hundreds of characters; snapping to
    the nearest earlier boundary there cost up to 12.5% of a document's length (worst measured
    fill 0.875 at the 3,700-char rung).
    """
    tape = "ab " + "X" * 800 + " tail"
    for n in (200, 400, 700):
        got = snap_head(tape, n)
        assert len(got) >= 0.9 * n, f"snap_head kept only {len(got)} of {n}"
        assert len(got) <= n

    tape2 = "head " + "Y" * 800 + " cd"
    for n in (200, 400, 700):
        got = snap_tail(tape2, n)
        assert len(got) >= 0.9 * n, f"snap_tail kept only {len(got)} of {n}"
        assert len(got) <= n


def test_monotonicity_survives_the_hard_cut_fallback():
    """Mixing snapped and hard cuts must not break the prefix/suffix ordering that nesting
    depends on."""
    tape = " ".join(
        ("w%d" % i) if i % 7 else ("Z" * 300) for i in range(600)
    )
    prev = ""
    for n in range(10, 5000, 23):
        cur = snap_head(tape, n)
        assert cur.startswith(prev), f"snap_head not monotone at n={n}"
        prev = cur
    prev = ""
    for n in range(10, 5000, 23):
        cur = snap_tail(tape, n)
        assert cur.endswith(prev), f"snap_tail not monotone at n={n}"
        prev = cur


def test_fill_fraction_with_pathological_filler():
    """End-to-end: documents stay near their budget even when filler has no whitespace."""
    rng = random.Random(3)
    pool = [
        (f"p{i}", ("Q" * rng.randint(200, 600)) if i % 3 == 0
         else " ".join(f"w{j}" for j in range(80)))
        for i in range(600)
    ]
    tapes = build_tapes(pool, min_tape_chars=max(LADDER) + 64)
    for budget in LADDER:
        r = weld(GOLD, tapes, budget, 0.5)
        assert r.n_chars >= 0.96 * budget, f"budget {budget}: filled only {r.n_chars / budget:.3f}"
        assert r.n_chars <= budget
        assert r.text[r.gold_char_start : r.gold_char_end] == GOLD


def test_snap_head_is_monotone():
    tape = " ".join(f"w{i}" for i in range(4000))
    prev = ""
    for n in range(0, 3000, 37):
        cur = snap_head(tape, n)
        assert cur.startswith(prev), f"snap_head not monotone at n={n}"
        assert len(cur) <= n or n <= 0
        prev = cur


def test_snap_tail_is_monotone():
    tape = " ".join(f"w{i}" for i in range(4000))
    prev = ""
    for n in range(0, 3000, 37):
        cur = snap_tail(tape, n)
        assert cur.endswith(prev), f"snap_tail not monotone at n={n}"
        assert len(cur) <= n or n <= 0
        prev = cur


def test_snapping_keeps_word_boundaries_on_ordinary_text():
    """On text with normal word spacing, slices must still cut cleanly at whitespace.

    The hard-cut fallback exists only for pathological filler (long unbroken tokens); it must
    not fire on ordinary prose, or every document would be cut mid-word for no reason.
    """
    tape = " ".join(f"word{i}" for i in range(2000))
    for n in range(50, 900, 13):
        head = snap_head(tape, n)
        if head and len(head) < len(tape):
            assert tape[len(head)].isspace(), f"snap_head cut mid-word at n={n}"
        tail = snap_tail(tape, n)
        if tail and len(tail) < len(tape):
            assert tape[len(tape) - len(tail) - 1].isspace(), f"snap_tail cut mid-word at n={n}"


def test_snap_loss_is_bounded_on_ordinary_text():
    """Snapping never discards more than its stated allowance."""
    from data.long_context.weld import _snap_allowance

    tape = " ".join(f"word{i}" for i in range(2000))
    for n in range(40, 3000, 29):
        assert n - len(snap_head(tape, n)) <= _snap_allowance(n)
        assert n - len(snap_tail(tape, n)) <= _snap_allowance(n)


# --------------------------------------------------------------------------- #
# Invariant 3: position control is genuinely uniform, and bins are balanced
# --------------------------------------------------------------------------- #


def test_position_fraction_lands_in_its_declared_bin():
    for bin_name, (lo, hi) in POSITION_BINS.items():
        for i in range(200):
            frac = sample_position_frac(random.Random(i), bin_name)
            assert lo <= frac <= hi


def test_gold_offset_is_not_collapsed_to_zero():
    """The old builder put 53% of documents at offset 0 for the 512 window."""
    tapes = _tapes()
    fracs = []
    for i in range(300):
        frac = sample_position_frac(random.Random(i), "uniform")
        r = weld(GOLD, tapes, 7400, frac)
        fracs.append(r.gold_char_start / max(1, r.n_chars - len(GOLD)))

    at_zero = sum(1 for f in fracs if f < 0.01) / len(fracs)
    assert at_zero < 0.10, f"{at_zero:.0%} of documents collapsed to offset 0"

    # roughly uniform: each quartile should be populated
    for lo in (0.0, 0.25, 0.5, 0.75):
        share = sum(1 for f in fracs if lo <= f < lo + 0.25) / len(fracs)
        assert share > 0.10, f"quartile [{lo}, {lo + 0.25}) holds only {share:.0%}"


def test_position_bins_are_balanced():
    bins = ("start", "middle", "end")
    counts = {b: 0 for b in bins}
    for i in range(3000):
        counts[stable_bin(f"ctx-{i}", bins)] += 1
    for b, c in counts.items():
        assert 0.29 < c / 3000 < 0.38, f"bin {b} holds {c / 3000:.0%}"


# --------------------------------------------------------------------------- #
# Invariant 4: determinism, independent of processing order
# --------------------------------------------------------------------------- #


def test_same_document_gets_same_layout_at_every_rung():
    """Distractor identity and relative gold position must not vary with size."""
    tapes = _tapes("doc-42")
    frac = sample_position_frac(record_rng("doc-42", "random"), "middle")

    results = [weld(GOLD, tapes, b, frac) for b in LADDER]
    for r in results:
        assert r.position_frac == frac
    for small, large in zip(results, results[1:]):
        assert set(small.distractor_ids) <= set(large.distractor_ids)


def test_record_rng_is_independent_of_other_records():
    """Draws for one record must not shift when other records are processed."""
    a = [record_rng("doc-A", "random").random() for _ in range(3)]
    for _ in range(500):
        record_rng(f"noise-{_}", "random").random()
    b = [record_rng("doc-A", "random").random() for _ in range(3)]
    assert a == b


def test_seed64_is_stable_not_hash_randomized():
    assert seed64("abc") == seed64("abc")
    assert seed64(20260716, "random", "doc-1") == seed64(20260716, "random", "doc-1")
    assert seed64("a", "b") != seed64("b", "a")


def test_condition_changes_the_distractor_stream():
    assert record_rng("doc-1", "random").random() != record_rng("doc-1", "bm25").random()


# --------------------------------------------------------------------------- #
# Invariant 5: separators, and no tokenizer dependency
# --------------------------------------------------------------------------- #


def test_passages_are_separated():
    """The old builder used "".join, running passages together mid-sentence."""
    tapes = _tapes()
    r = weld(GOLD, tapes, 7400, 0.5)
    assert "\n\n" in r.text
    assert r.text[: r.gold_char_start].endswith("\n\n")


def test_raw_whitespace_survives():
    """The old builder round-tripped text through mE5 SentencePiece, destroying newlines."""
    gold = "שורה ראשונה\nשורה שנייה\n\nפסקה חדשה"
    tapes = _tapes()
    r = weld(gold, tapes, 7400, 0.5)
    assert "\n" in r.text
    assert r.text[r.gold_char_start : r.gold_char_end] == gold


def test_weld_module_imports_no_tokenizer():
    """Structural guarantee that length is budgeted in model-neutral units."""
    import data.long_context.weld as w

    src = open(w.__file__, encoding="utf-8").read()
    for banned in ("transformers", "AutoTokenizer", "sentencepiece", "tiktoken"):
        assert banned not in src, f"weld.py must not depend on {banned}"

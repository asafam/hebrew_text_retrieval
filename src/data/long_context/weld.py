"""
Welding: place a gold passage inside a longer document of distractor text.

This module is the testable core of the long-context benchmark. It imports no
tokenizer and performs no I/O, which is a deliberate structural guarantee: document
length is budgeted in *characters*, so the corpus cannot become specific to any one
model's vocabulary. (The previous builder budgeted in ``intfloat/multilingual-e5-base``
tokens, which silently calibrated every corpus to mE5.)

Two invariants hold by construction and are asserted:

1. The gold passage appears verbatim and complete:
   ``text[gold_char_start:gold_char_end] == gold_text``. It is never truncated.
2. Infix nesting: for budgets b1 <= b2 over the same record, ``doc(b1)`` is an exact
   contiguous substring of ``doc(b2)``, with byte-identical gold and inner context.
   Growth happens by extending outward from the gold on both sides.

Note that (2) is *infix* nesting, not prefix nesting. Prefix nesting is impossible for
any gold position other than the very start: if the gold sits at 50% of both a short and
a long document, the short one cannot be a prefix of the long one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

SEPARATOR = "\n\n"

PositionBin = Literal["start", "middle", "end"]

#: Fraction of the distractor slack placed *before* the gold. A continuous range over
#: character offsets — not an index into a passage list. The old builder drew
#: ``randint(0, len(distractors))`` over a list that was often 1-3 long, which put 53%
#: of documents at offset 0 while claiming to be "random".
POSITION_BINS: dict[str, tuple[float, float]] = {
    "start": (0.00, 0.05),
    "middle": (0.45, 0.55),
    "end": (0.95, 1.00),
    "uniform": (0.00, 1.00),
}

#: Cap on where the gold may end, as a fraction of the budget. Leaves a tail of pure
#: distractor text so that a document whose tokenizer ratio runs below the calibrated
#: p05 cannot push the gold past a model's position limit.
MAX_GOLD_END_FRAC = 0.97

#: Characters that whitespace-snapping may discard from a slice before it is abandoned in
#: favour of a hard character cut.
#:
#: Why a fallback exists at all: snapping to a word boundary normally costs a few characters,
#: but fiqa contains long unbroken tokens (URLs, figures, table rows). When the nearest earlier
#: whitespace is far back, snapping discards most of the slice -- measured 61 fiqa documents
#: below 0.96 fill, worst case 0.875, a 463-character shortfall at the 3,700-char rung. Cutting
#: mid-word is harmless *here* because only filler is ever cut; the gold is never truncated.
#:
#: Why the allowance is a constant and not a fraction of the slice: a proportional allowance
#: lets the two slices together lose a proportion of the *slack*, and when the gold is small the
#: slack is most of the document. An 822-character gold in a 3,700-character budget leaves 78%
#: slack, so a 10% allowance still left documents at 91-95% of target. A flat 32 characters
#: bounds the whole document's shortfall to ~64 characters at any rung -- under 2% even at the
#: smallest. 32 is far more than any ordinary word needs, so normal prose snaps cleanly.
#:
#: Monotonicity, which infix nesting depends on, survives: if the slice for b2 snapped to cut2
#: and cut2 <= b1, then no whitespace lies in (cut2, b2], so b1's nearest boundary is that same
#: cut2 and loss1 = b1 - cut2 <= b2 - cut2 <= 32. b1 therefore also snaps, to the same point, so
#: no ordering is inverted. A constant allowance makes this argument hold with no case analysis.
MAX_SNAP_LOSS_CHARS = 32


class GoldTooLongError(ValueError):
    """The gold passage does not fit in the budget. Never silently truncate it."""


def _snap_allowance(n: int) -> float:
    """Characters that whitespace-snapping may discard from a slice of length ``n``.

    Deliberately a constant rather than a fraction of ``n``. A proportional allowance lets the
    two slices together lose a proportion of the *slack*, and when the gold is small the slack
    is most of the document: an 822-character gold in a 3,700-character budget leaves 78% slack,
    so a 10% allowance put documents at 91-95% of target length. A flat 32 characters bounds the
    whole document's shortfall to ~64 characters regardless of rung, which is under 2% even at
    the smallest one.

    32 characters is far more than any ordinary word needs, so normal prose still snaps cleanly;
    only pathological unbroken runs (URLs, figures) exceed it and fall back to a hard cut.
    """
    return float(MAX_SNAP_LOSS_CHARS)


def _rfind_ws(s: str, hi: int) -> int:
    """Index of the last whitespace character in ``s[:hi]``, or -1."""
    for i in range(min(hi, len(s)) - 1, -1, -1):
        if s[i].isspace():
            return i
    return -1


def _find_ws(s: str, lo: int) -> int:
    """Index of the first whitespace character at or after ``lo``, or -1."""
    for i in range(max(lo, 0), len(s)):
        if s[i].isspace():
            return i
    return -1


def snap_head(tape: str, n: int) -> str:
    """Longest prefix of ``tape`` of length <= n ending at a whitespace boundary.

    Monotone in ``n``: for n1 <= n2 the result for n1 is a prefix of the result for n2.
    That is what makes nesting exact rather than approximate, so no slop tolerance is
    needed when verifying it.
    """
    if n >= len(tape):
        return tape
    if n <= 0:
        return ""
    cut = _rfind_ws(tape, n + 1)
    if cut > 0 and (n - cut) <= _snap_allowance(n):
        return tape[:cut]
    return tape[:n]  # snapping would throw away too much; cut mid-word (filler only)


def snap_tail(tape: str, n: int) -> str:
    """Longest suffix of ``tape`` of length <= n starting at a whitespace boundary.

    Monotone in ``n``: for n1 <= n2 the result for n1 is a suffix of the result for n2.
    """
    if n >= len(tape):
        return tape
    if n <= 0:
        return ""
    cut = _find_ws(tape, len(tape) - n)
    if cut != -1 and (n - (len(tape) - cut - 1)) <= _snap_allowance(n):
        return tape[cut + 1 :]
    return tape[-n:]  # snapping would throw away too much; cut mid-word (filler only)


@dataclass(frozen=True)
class Tapes:
    """Size-independent character tapes flanking the gold.

    ``left`` is consumed as a *suffix* (its end abuts the gold) and ``right`` as a
    *prefix* (its start abuts the gold), so a larger budget extends the document
    outward while leaving the inner text untouched.
    """

    left: str
    right: str
    left_ids: tuple[str, ...] = ()
    right_ids: tuple[str, ...] = ()


def build_tapes(
    distractors: Iterable[tuple[str, str]],
    *,
    min_tape_chars: int,
    separator: str = SEPARATOR,
) -> Tapes:
    """Build the two tapes from a fixed, size-independent distractor sequence.

    ``distractors`` yields ``(passage_id, text)``. Passages are dealt alternately to
    the left and right sides. Each tape is filled to at least ``min_tape_chars`` so
    that the largest rung can be sliced from it without running dry.
    """
    left_parts: list[str] = []
    right_parts: list[str] = []
    left_ids: list[str] = []
    right_ids: list[str] = []
    left_len = right_len = 0

    for i, (pid, text) in enumerate(distractors):
        if left_len <= right_len:
            left_parts.append(text)
            left_ids.append(pid)
            left_len += len(text) + len(separator)
        else:
            right_parts.append(text)
            right_ids.append(pid)
            right_len += len(text) + len(separator)
        if left_len >= min_tape_chars and right_len >= min_tape_chars:
            break

    # The left tape's *end* abuts the gold, so the passage nearest the gold must be
    # last. Reversing puts the first-dealt (innermost) passage adjacent to the gold.
    left = separator.join(reversed(left_parts))
    right = separator.join(right_parts)
    return Tapes(
        left=left,
        right=right,
        left_ids=tuple(reversed(left_ids)),
        right_ids=tuple(right_ids),
    )


@dataclass
class WeldResult:
    text: str
    gold_char_start: int
    gold_char_end: int
    gold_char_frac: float
    n_chars: int
    n_words: int
    position_frac: float
    padded: bool = True
    tape_exhausted: bool = False
    distractor_ids: tuple[str, ...] = field(default_factory=tuple)


def weld(
    gold_text: str,
    tapes: Tapes,
    budget_chars: int,
    position_frac: float,
    *,
    separator: str = SEPARATOR,
    max_gold_end_frac: float = MAX_GOLD_END_FRAC,
    allow_passthrough: bool = True,
) -> WeldResult:
    """Place ``gold_text`` inside a document of ``budget_chars`` characters.

    ``position_frac`` is the fraction of the distractor slack placed before the gold.

    A passage already at or above the budget is emitted **unpadded** (``padded=False``)
    rather than clipped: the rung is a target length, and a document that already exceeds
    it is by definition already long. This keeps the document count exactly constant
    across every rung — a requirement, since the previous benchmark let corpus size vary
    from 447K to 556K and thereby confounded length with corpus size. The rule is applied
    without reference to whether the passage is a gold.

    In translated BeIR this affects <=0.7% of documents at the smallest padded rung and
    ~0.1% above it (fiqa's longest passage is 13,748 chars).
    """
    gold_len = len(gold_text)
    sep_len = len(separator)

    if gold_len + 2 * sep_len > budget_chars:
        if not allow_passthrough:
            raise GoldTooLongError(
                f"gold is {gold_len} chars but the budget is {budget_chars}; "
                "refusing to truncate the gold passage"
            )
        return WeldResult(
            text=gold_text,
            gold_char_start=0,
            gold_char_end=gold_len,
            gold_char_frac=0.0,
            n_chars=gold_len,
            n_words=len(gold_text.split()),
            position_frac=position_frac,
            padded=False,
        )

    slack = budget_chars - gold_len - 2 * sep_len
    left_budget = int(position_frac * slack)

    # Keep a tail of distractor text after the gold so a low tokenizer ratio cannot
    # push the gold beyond a model's position limit.
    max_gold_end = int(max_gold_end_frac * budget_chars)
    if gold_len + left_budget + sep_len > max_gold_end:
        left_budget = max(0, max_gold_end - gold_len - sep_len)
    right_budget = slack - left_budget

    # NOTE: do not "compensate" right_budget with slack - len(left). That looks like the
    # natural way to recover the left slice's snapping loss, but it makes right_budget depend
    # on len(left), which is not monotone in the budget -- for the `end` bin a small shift in
    # left's snap point flips the remainder and the right slice *shrinks* as the budget grows,
    # breaking infix nesting. The dense-ladder test catches this. Bounding the per-slice loss
    # absolutely (see _snap_allowance) achieves the same fill without touching monotonicity.
    left = snap_tail(tapes.left, left_budget)
    right = snap_head(tapes.right, right_budget)

    parts = [p for p in (left, gold_text, right) if p]
    text = separator.join(parts)

    gold_char_start = (len(left) + sep_len) if left else 0
    gold_char_end = gold_char_start + gold_len

    # The single assertion that makes the old builder's headline bug unrepresentable:
    # it clipped the gold in 71% of 512-token rows.
    assert text[gold_char_start:gold_char_end] == gold_text, (
        "gold passage is not intact at its recorded offsets"
    )

    n_chars = len(text)
    denom = max(1, n_chars - gold_len)
    exhausted = (len(left) < left_budget) or (len(right) < right_budget)

    return WeldResult(
        text=text,
        gold_char_start=gold_char_start,
        gold_char_end=gold_char_end,
        gold_char_frac=gold_char_start / denom,
        n_chars=n_chars,
        n_words=len(text.split()),
        position_frac=position_frac,
        tape_exhausted=exhausted,
        distractor_ids=tapes.left_ids + tapes.right_ids,
    )


def sample_position_frac(rng, bin_name: str) -> float:
    """Draw a position fraction from a named bin using a caller-supplied generator."""
    lo, hi = POSITION_BINS[bin_name]
    return rng.uniform(lo, hi)

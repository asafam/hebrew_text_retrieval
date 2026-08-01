"""
Tokenizer calibration for the long-context benchmark.

Document length is budgeted in **characters** so that no model's vocabulary defines the
task. That choice is what makes the benchmark fair, but it means the mapping from a
character budget to each model's token count has to be measured rather than assumed --
and reported, because "who can actually see this document" is itself a headline result.

Two jobs, run at different times:

*Before building* -- measure chars/token on the raw corpora and confirm the intended
character ladder keeps every model inside its position limit.

*After building* -- compute exact gold token offsets in the welded documents via
``return_offsets_mapping``, which yields ``gold_visible_at_maxlen`` per model: whether the
answering passage falls inside the window at all. That flag is the explanatory variable for
mE5's collapse, and reporting the score *conditioned* on it is what keeps the comparison
from reading as a strawman.

``load_tokenizer`` always overrides ``model_max_length`` before measuring, because a
tokenizer that silently truncates reports nonsense. Verified per checkpoint:
``HebrewModernBERT-base-{final,phase1}`` correctly declare 8192, but
``HebrewModernBERT-base-phase0.2`` declares 1024 -- so the override matters if the arm is
ever switched to an older checkpoint, even though it is a no-op for ``-final``.

Operational note: the model files live on a shared network filesystem, and tokenizer loads
from it intermittently stall for minutes before succeeding (a cold load measured 4.3s once
and >95s another time, for the same checkpoint). Encoding itself is instant -- 30k characters
in well under a second. Two consequences for anything running under a wall clock: treat a
slow tokenizer load as expected rather than a failure, and load each tokenizer **once** and
reuse it across datasets and rungs instead of reloading per cell.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Sequence

HMB_PATH = "/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf/HebrewModernBERT-base-final"

#: The arms under comparison, with the tokenizer used to measure them.
MODELS: dict[str, str] = {
    "HMB": HMB_PATH,
    "NDB": "dicta-il/NeoDictaBERT",
    "mE5-base": "intfloat/multilingual-e5-base",
    "mE5-large": "intfloat/multilingual-e5-large",
}

#: The character ladder. ``c0`` (raw, unpadded) is handled separately since it has no budget.
#:
#: Budgets are set so that p99 of the *welded* token distribution stays under 0.95x each
#: model's native limit on all three corpora. Two rungs were lowered from their first draft
#: after measurement: 14,800 -> 11,800 because scidocs pushed NeoDictaBERT past 4,096 at a
#: rung the design calls "NDB fully native", and 29,600 -> 27,000 because scifact (the densest
#: corpus, 3.90 chars/token) left HMB under 2% headroom at its own ceiling rung. A rung where
#: a supposedly-native arm silently truncates measures nothing.
LADDER_CHARS: tuple[int, ...] = (3700, 7400, 11800, 19000, 27000)

#: Rungs at which each model is expected to encode natively. Used by ``verify.py`` to decide
#: which (model, rung) pairs must satisfy the p99 headroom assertion; beyond these a model is
#: expected to chunk, so overflow is the measured condition rather than a failure.
NATIVE_THROUGH: dict[str, int] = {"HMB": 27000, "NDB": 11800, "mE5-base": 0, "mE5-large": 0}

#: Maximum tolerated fraction of documents exceeding a model's limit at a rung where it is
#: expected to be native. Measured worst case is 0.17% (scidocs); see the overflow policy in
#: the plan for why this cannot be driven to zero.
MAX_OVERFLOW_FRAC = 0.01


def model_native_limit(model_path: str) -> int:
    """Longest input the encoder accepts without a position-index error.

    Deliberately checks several config keys, because the three families disagree:
    ModernBERT uses ``max_position_embeddings`` (8192), NeoBERT publishes only
    ``max_length`` (4096) with no ``max_position_embeddings`` at all, and XLM-R reports 514
    positions of which 512 are usable. Getting this wrong is how the previous harness ended
    up pinning mE5 to 512 in a shell variable and calling the result "degradation".
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mpe = getattr(cfg, "max_position_embeddings", None)
    if mpe:
        # XLM-R reserves 2 positions (padding offset), so 514 -> 512 usable.
        return int(mpe) - 2 if int(mpe) in (514, 512 + 2) else int(mpe)

    for key in ("max_length", "n_positions", "max_seq_length"):
        val = getattr(cfg, key, None)
        if val:
            return int(val)

    raise ValueError(f"cannot determine native context limit for {model_path!r}")


def load_tokenizer(model_path: str):
    """Load a tokenizer with truncation disabled, so measurements are not silently capped."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.model_max_length = int(1e9)  # see the HMB=1024 trap in the module docstring
    return tok


@dataclass
class RatioStats:
    """chars/token distribution. ``p05`` is the pessimistic end (most tokens per char)."""

    mean: float
    p05: float
    p50: float
    p95: float
    n: int

    def as_dict(self) -> dict:
        return {"mean": self.mean, "p05": self.p05, "p50": self.p50, "p95": self.p95, "n": self.n}


def chars_per_token(texts: Sequence[str], tokenizer) -> RatioStats:
    ratios = []
    for t in texts:
        n_tok = len(tokenizer.encode(t, add_special_tokens=False))
        if n_tok:
            ratios.append(len(t) / n_tok)
    ratios.sort()
    n = len(ratios)
    return RatioStats(
        mean=statistics.mean(ratios),
        p05=ratios[max(0, int(0.05 * n))],
        p50=statistics.median(ratios),
        p95=ratios[min(n - 1, int(0.95 * n))],
        n=n,
    )


@dataclass
class TokenBudgetCheck:
    """Whether a character budget keeps a model inside its window."""

    model: str
    budget_chars: int
    limit: int
    p50_tokens: float
    p99_tokens: float
    max_tokens: int
    frac_over_limit: float

    @property
    def safe(self) -> bool:
        return self.p99_tokens <= self.limit

    def __str__(self) -> str:
        flag = "ok " if self.safe else "OVER"
        return (
            f"  [{flag}] {self.model:10s} budget={self.budget_chars:6,d}ch "
            f"p50={self.p50_tokens:7,.0f} p99={self.p99_tokens:7,.0f} "
            f"max={self.max_tokens:7,d} limit={self.limit:5,d} "
            f"over={self.frac_over_limit:5.2%}"
        )


def check_welded_budget(
    welded_texts: Sequence[str],
    tokenizer,
    *,
    model: str,
    budget_chars: int,
    limit: int,
) -> TokenBudgetCheck:
    """Token statistics for *welded* documents at one budget.

    This must be measured on welded text, not raw passages. A welded document concatenates
    many passages, so its chars/token ratio is an average over all of them and concentrates
    near the corpus mean, whereas a single passage can sit far out in the tail. Budgeting
    from the per-passage p05 would therefore be substantially over-conservative; the
    quantity that actually matters is the p99 of the welded distribution.
    """
    counts = sorted(len(tokenizer.encode(t, add_special_tokens=True)) for t in welded_texts)
    n = len(counts)
    return TokenBudgetCheck(
        model=model,
        budget_chars=budget_chars,
        limit=limit,
        p50_tokens=statistics.median(counts),
        p99_tokens=counts[min(n - 1, int(0.99 * n))],
        max_tokens=counts[-1],
        frac_over_limit=sum(1 for c in counts if c > limit) / n,
    )


@dataclass
class GoldTokenSpan:
    """Where the gold passage lands in token space, per model."""

    n_tokens: int
    gold_tok_start: int
    gold_tok_end: int
    limit: int

    @property
    def gold_visible(self) -> bool:
        """True if the whole gold span falls inside the model's window."""
        return self.gold_tok_end <= self.limit

    @property
    def frac_doc_seen(self) -> float:
        return min(1.0, self.limit / max(1, self.n_tokens))

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "gold_tok_start": self.gold_tok_start,
            "gold_tok_end": self.gold_tok_end,
            "gold_visible": self.gold_visible,
            "frac_doc_seen": round(self.frac_doc_seen, 4),
        }


def gold_token_span(
    text: str,
    gold_char_start: int,
    gold_char_end: int,
    tokenizer,
    limit: int,
) -> GoldTokenSpan:
    """Map the gold's character span to a token span via the fast tokenizer's offsets.

    The prefix is deliberately not tokenized separately: tokenizing ``text[:start]`` in
    isolation is not guaranteed to agree with tokenizing the whole string, so the offset
    map of the single full encoding is the only correct source.
    """
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]

    tok_start, tok_end = None, None
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue  # special tokens carry an empty span
        if tok_start is None and b > gold_char_start:
            tok_start = i
        if a < gold_char_end:
            tok_end = i + 1
    return GoldTokenSpan(
        n_tokens=len(offsets),
        gold_tok_start=tok_start if tok_start is not None else 0,
        gold_tok_end=tok_end if tok_end is not None else 0,
        limit=limit,
    )


@dataclass
class Calibration:
    """The manifest-bound calibration record."""

    raw_ratios: dict[str, dict] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)
    budget_checks: list[dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "raw_chars_per_token": self.raw_ratios,
                    "native_limits": self.limits,
                    "welded_budget_checks": self.budget_checks,
                },
                fh,
                indent=2,
            )
        os.replace(tmp, path)

"""
Deterministic per-record seeding for long-context benchmark construction.

The old builder used a single module-level ``random.seed(42)``. Because each record
consumed a variable number of draws (however many distractors it happened to need),
the RNG stream position depended on every record processed before it — so the same
document received different distractors and a different gold position at every
context size, and adding one record perturbed all subsequent ones.

Here every record derives its own generator from a stable hash of its identity.
Nothing in this module reads or writes global RNG state.
"""

from __future__ import annotations

import hashlib
import random

GLOBAL_SEED = 20260716


def seed64(*parts: object) -> int:
    """Stable 64-bit seed from arbitrary stringable parts.

    Uses SHA-256 rather than ``hash()``, which is randomized per interpreter run
    (PYTHONHASHSEED) and would silently break reproducibility across machines.
    """
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def record_rng(
    seed_doc_id: str,
    condition: str,
    *,
    global_seed: int = GLOBAL_SEED,
) -> random.Random:
    """Per-record generator.

    Deliberately does *not* depend on the context size or the position bin: the
    distractor sequence and the gold's relative position must be identical across
    every rung of the ladder, so that length is the only variable that changes.
    """
    return random.Random(seed64(global_seed, condition, seed_doc_id))


def stable_bin(seed_doc_id: str, bins: tuple[str, ...]) -> str:
    """Assign a record to one of ``bins`` by stable hash — deterministic
    round-robin, so bin membership is identical across rungs, conditions and models.
    """
    return bins[seed64(seed_doc_id) % len(bins)]

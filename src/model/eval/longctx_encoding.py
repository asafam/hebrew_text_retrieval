"""
Encoding strategies for long-context retrieval evaluation.

Three strategies, and the distinction between them is the whole fairness argument:

``native``
    Encode the document in one pass. Only legal if the document fits the model's real
    position limit; the driver refuses the config otherwise rather than truncating quietly.

``truncate``
    Encode the first ``window`` tokens and discard the rest. This is what the previous
    harness did to mE5 at *every* context size via a shell variable
    (``MAX_LEN="${MAX_LEN:-$CTX}"``), then reported the resulting collapse as
    "degradation". It is a legitimate condition to measure, but it must be *labelled* as
    truncation, and a win over it proves only that a 512-token model cannot see token 5,000.

``chunked``
    Split into fixed overlapping token windows, encode each, score the query against every
    window and keep the best.

``chunked_para``
    Split on paragraph boundaries first, sub-splitting only pieces that exceed the window.
    **This is the primary chunked baseline**, because fixed windows quietly handicap it.
    Measured on scifact at ``c27k``: the median gold passage is 402 mE5 tokens against a
    510-token window, so fixed-window chunking leaves the gold *whole* in only 20% of
    documents at stride 0, rising to just 41.7% at a 50% overlap that costs 38 windows per
    document. Splitting on the paragraph breaks that real documents (and these welded ones)
    actually contain recovers whole passages instead.

Truncated mE5 can see the gold in as little as 1.7% of documents at the top rung, so the
chunked arms are not optional -- a win over truncation alone would prove only that a
512-token model cannot see token 5,000. Giving the baseline its strongest honest
configuration is the point; ``chunked_para`` is that configuration.

One disclosure this requires: paragraph-aware chunking uses structure that exists because
the documents were welded from passages. Real long documents do have paragraph breaks, and
every arm gets the same benefit, but it is a way in which the synthetic construction is
exploitable and should be stated rather than hidden.

The same code path serves all three: ``native`` and ``truncate`` are just the degenerate
case where every document yields exactly one window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Strategy = Literal["native", "truncate", "chunked", "chunked_para"]

#: Overlap between adjacent windows, in tokens. Some overlap matters because a gold passage
#: that straddles a window boundary would otherwise be split across two windows and be fully
#: present in neither.
DEFAULT_STRIDE = 64


def model_native_limit(model_path: str) -> int:
    """Longest input the encoder accepts without a position-index error.

    Checks several config keys because the families disagree: ModernBERT publishes
    ``max_position_embeddings`` (8192), NeoBERT publishes only ``max_length`` (4096) and has
    no ``max_position_embeddings`` at all, and XLM-R reports 514 positions of which 512 are
    usable. Reading the wrong key is how a harness ends up hardcoding a limit in a shell
    variable.
    """
    import json
    import os

    from transformers import AutoConfig

    # An InfoNCE dual-encoder checkpoint's config.json describes the *wrapper*
    # ("info_nce_dual_encoder"), which Transformers does not recognise -- AutoConfig raises
    # KeyError on it. The context limit belongs to the base encoder the wrapper points at, so
    # follow that pointer. This is the same failure class that stopped every NeoDictaBERT run
    # in the previous harness from ever completing.
    local_cfg = os.path.join(model_path, "config.json")
    if os.path.isdir(model_path) and os.path.exists(local_cfg):
        with open(local_cfg, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("model_type") == "info_nce_dual_encoder":
            base = raw.get("doc_model_name") or raw.get("query_model_name")
            if not base:
                raise ValueError(
                    f"{model_path!r} is an InfoNCE checkpoint with no base model recorded"
                )
            model_path = base

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mpe = getattr(cfg, "max_position_embeddings", None)
    if mpe:
        mpe = int(mpe)
        # XLM-R reserves two positions for its padding offset: 514 -> 512 usable.
        return mpe - 2 if mpe in (514,) else mpe

    for key in ("max_length", "n_positions", "max_seq_length"):
        val = getattr(cfg, key, None)
        if val:
            return int(val)

    raise ValueError(f"cannot determine native context limit for {model_path!r}")


class IncompatibleStrategy(ValueError):
    """The requested strategy cannot be honoured for this model."""


def resolve_window(
    strategy: Strategy,
    requested_window: int | None,
    native_limit: int,
    *,
    model_label: str = "",
) -> int:
    """Validate a (strategy, window) pair against what the model can actually do.

    The point of failing loudly here is that ``sentence_transformers`` stores an explicit
    ``max_seq_length`` verbatim and passes it to the tokenizer, so asking mE5 for 8192 raises
    a CUDA position-index error deep inside a forward pass rather than truncating gracefully.
    Better to refuse the config up front and name the alternative.
    """
    if strategy == "native":
        window = requested_window or native_limit
        if window > native_limit:
            raise IncompatibleStrategy(
                f"{model_label or 'model'} supports {native_limit} tokens but native "
                f"encoding of {window} was requested; use --strategy chunked "
                f"--window {native_limit} instead"
            )
        return window

    if strategy in ("truncate", "chunked", "chunked_para"):
        window = requested_window or native_limit
        if window > native_limit:
            raise IncompatibleStrategy(
                f"{model_label or 'model'} window {window} exceeds its limit {native_limit}"
            )
        return window

    raise IncompatibleStrategy(f"unknown strategy {strategy!r}")


@dataclass
class WindowPlan:
    """Windows to encode, and the document each belongs to.

    ``window_to_doc[i]`` is the index of the document that window ``i`` came from, which is
    what lets the scorer aggregate window scores back up to documents.
    """

    texts: list[str]
    window_to_doc: list[int]
    n_docs: int

    @property
    def n_windows(self) -> int:
        return len(self.texts)

    @property
    def windows_per_doc(self) -> float:
        return self.n_windows / max(1, self.n_docs)


def plan_windows(
    doc_texts: Sequence[str],
    tokenizer,
    *,
    strategy: Strategy,
    window: int,
    stride: int = DEFAULT_STRIDE,
    doc_prefix: str = "",
    batch: int = 2000,
    separator: str = "\n\n",
) -> WindowPlan:
    """Split documents into windows according to ``strategy``.

    For ``native``/``truncate`` this returns one window per document. For ``chunked`` it uses
    the fast tokenizer's overflow machinery, which yields both the windows and the
    window->document map in a single call rather than a hand-rolled sliding window.

    ``doc_prefix`` (e.g. mE5's ``"passage: "``) is applied to **every** window, since that is
    how the model was trained to see a passage. Giving the baseline its intended
    configuration is part of not building a strawman.
    """
    if strategy in ("native", "truncate"):
        return WindowPlan(
            texts=[doc_prefix + t for t in doc_texts],
            window_to_doc=list(range(len(doc_texts))),
            n_docs=len(doc_texts),
        )

    if strategy == "chunked_para":
        return _plan_windows_paragraph(
            doc_texts,
            tokenizer,
            window=window,
            stride=stride,
            doc_prefix=doc_prefix,
            separator=separator,
        )

    # Reserve room for the special tokens the tokenizer will add back per window.
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    content = max(16, window - n_special)

    texts: list[str] = []
    window_to_doc: list[int] = []
    for start in range(0, len(doc_texts), batch):
        chunk = [doc_prefix + t for t in doc_texts[start : start + batch]]
        enc = tokenizer(
            chunk,
            max_length=content,
            truncation=True,
            stride=min(stride, content // 2),
            return_overflowing_tokens=True,
            add_special_tokens=False,
            padding=False,
        )
        mapping = enc["overflow_to_sample_mapping"]
        for ids, sample_idx in zip(enc["input_ids"], mapping):
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
            window_to_doc.append(start + int(sample_idx))

    return WindowPlan(texts=texts, window_to_doc=window_to_doc, n_docs=len(doc_texts))


def _plan_windows_paragraph(
    doc_texts: Sequence[str],
    tokenizer,
    *,
    window: int,
    stride: int,
    doc_prefix: str,
    separator: str,
) -> WindowPlan:
    """Paragraph-aware chunking: split on ``separator``, then pack greedily up to ``window``.

    Packing adjacent paragraphs together rather than emitting one window per paragraph keeps
    the window count near the fixed-window cost while still never splitting a paragraph that
    fits. A paragraph longer than the window is sub-split with the usual overlap, since
    nothing can keep it whole.
    """
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    content = max(16, window - n_special)

    texts: list[str] = []
    window_to_doc: list[int] = []

    for doc_idx, doc in enumerate(doc_texts):
        paras = [p for p in doc.split(separator) if p.strip()] or [doc]
        lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in paras]

        buf: list[str] = []
        buf_len = 0
        emitted = 0

        def flush() -> None:
            nonlocal buf, buf_len, emitted
            if buf:
                texts.append(doc_prefix + separator.join(buf))
                window_to_doc.append(doc_idx)
                buf, buf_len = [], 0
                emitted += 1

        for para, n_tok in zip(paras, lens):
            if n_tok > content:
                # Too long to keep whole: flush what we have, then sub-split this one.
                flush()
                ids = tokenizer.encode(para, add_special_tokens=False)
                step = max(1, content - min(stride, content // 2))
                for s in range(0, len(ids), step):
                    piece = ids[s : s + content]
                    if not piece:
                        break
                    texts.append(doc_prefix + tokenizer.decode(piece, skip_special_tokens=True))
                    window_to_doc.append(doc_idx)
                    emitted += 1
                    if s + content >= len(ids):
                        break
                continue
            if buf_len + n_tok > content:
                flush()
            buf.append(para)
            buf_len += n_tok
        flush()

        if emitted == 0:  # pathological: emit something so the document is never dropped
            texts.append(doc_prefix + doc[:1] if doc else doc_prefix)
            window_to_doc.append(doc_idx)

    return WindowPlan(texts=texts, window_to_doc=window_to_doc, n_docs=len(doc_texts))


def aggregate_window_scores(
    scores,
    window_to_doc,
    n_docs: int,
    *,
    how: Literal["max", "mean"] = "max",
):
    """Reduce a (queries x windows) score matrix to (queries x documents).

    Uses ``scatter_reduce_`` so the result is exact for both ``max`` and ``mean`` -- no
    top-k-windows approximation, which matters because the max-vs-mean comparison is itself
    something to report.
    """
    import torch

    doc_of = window_to_doc
    if not isinstance(doc_of, torch.Tensor):
        doc_of = torch.as_tensor(list(window_to_doc), dtype=torch.long)
    doc_of = doc_of.to(scores.device)

    n_q = scores.shape[0]
    idx = doc_of.unsqueeze(0).expand(n_q, -1)

    if how == "max":
        out = torch.full((n_q, n_docs), float("-inf"), device=scores.device, dtype=scores.dtype)
        out.scatter_reduce_(1, idx, scores, reduce="amax", include_self=True)
        return out

    total = torch.zeros((n_q, n_docs), device=scores.device, dtype=scores.dtype)
    total.scatter_reduce_(1, idx, scores, reduce="sum", include_self=True)
    counts = torch.zeros(n_docs, device=scores.device, dtype=scores.dtype)
    counts.scatter_reduce_(
        0, doc_of, torch.ones_like(doc_of, dtype=scores.dtype), reduce="sum", include_self=True
    )
    return total / counts.clamp(min=1).unsqueeze(0)

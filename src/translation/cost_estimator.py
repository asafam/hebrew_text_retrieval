"""
Pre-flight cost estimation for translation batch jobs.

Samples a subset of pending rows to estimate average token counts, then
projects total input/output tokens and USD cost at the configured pricing.
Uses Batch API pricing (50% cheaper than synchronous).
"""

import pandas as pd
from translation.utils import count_tokens


def estimate_batch_cost(
    csv_path: str,
    text_col: str,
    config: dict,
    n_sample: int = 300,
) -> dict:
    """
    Estimate the API cost of translating pending rows in a CSV file.

    Args:
        csv_path:  Path to source or partially-translated CSV.
        text_col:  Column containing the text to translate (e.g. 'text', 'segment_text').
        config:    Full pipeline config dict (uses guardrails and datasets sections).
        n_sample:  Number of rows to sample for average token estimation.

    Returns:
        dict with keys: n_pending, avg_text_tokens, estimated_input_tokens,
        estimated_output_tokens, estimated_cost_usd.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    pending = df[df["translation"].isna()] if "translation" in df.columns else df
    n_pending = len(pending)

    if n_pending == 0:
        return {
            "n_pending": 0,
            "avg_text_tokens": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    sample = pending.sample(min(n_sample, n_pending), random_state=42)
    tokenizer_model = config["datasets"].get("tokenizer_model", "gpt-4o-mini-2024-07-18")
    avg_text_tokens = (
        sample[text_col]
        .dropna()
        .apply(lambda x: count_tokens(str(x), tokenizer_model))
        .mean()
    )

    # Overhead per request: system prompt + user prompt template + keys (~120 tokens)
    PROMPT_OVERHEAD = 120
    avg_input_tokens = PROMPT_OVERHEAD + avg_text_tokens
    # Hebrew translations are roughly the same length in tokens as English source
    avg_output_tokens = avg_text_tokens * 0.95

    guardrails = config.get("guardrails", {})
    price_in = guardrails.get("cost_per_1m_input_tokens", 0.075)
    price_out = guardrails.get("cost_per_1m_output_tokens", 0.300)

    total_input = n_pending * avg_input_tokens
    total_output = n_pending * avg_output_tokens
    estimated_cost = (total_input / 1e6 * price_in) + (total_output / 1e6 * price_out)

    return {
        "n_pending": n_pending,
        "avg_text_tokens": int(avg_text_tokens),
        "estimated_input_tokens": int(total_input),
        "estimated_output_tokens": int(total_output),
        "estimated_cost_usd": round(estimated_cost, 4),
    }

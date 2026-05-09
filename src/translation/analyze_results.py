"""
Analyzes the master results CSV produced by collect_results.py.

Produces:
  - A markdown report (analysis_report.md) with ranked tables for each factor
  - PNG charts saved to the output directory

Analysis questions answered:
  1. Which translation model is best overall?
  2. Which prompt strategy is best overall?
  3. How does performance vary by dataset category?
  4. How does performance vary by text type (query vs document)?
  5. How does text length (short vs long documents) affect quality?
  6. How consistent are judge models with each other? (inter-judge agreement)
  7. Best (model × prompt) combination per dataset category — heatmaps
  8. Best (translation model × prompt strategy) combinations
  9. Does context help more for queries than documents? (text_type × prompt)
 10. Which model degrades least on long documents? (text_length × model)

Usage:
  python src/translation/analyze_results.py \
      --results_path outputs/translation/BeIR/results_master.csv \
      --output_dir  outputs/translation/BeIR/analysis
"""

import argparse
import os
import re
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def mean_score(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    return (
        df.groupby(groupby)["score"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values("mean", ascending=False)
        .round({"mean": 3, "std": 3})
    )


def mean_score_by_judge(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    """Per-judge mean scores plus an 'Average' row aggregated across all judges."""
    per_judge = (
        df.groupby(groupby + ["judge_model"])["score"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .round({"mean": 3, "std": 3})
    )
    avg = (
        df.groupby(groupby)["score"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .round({"mean": 3, "std": 3})
    )
    avg["judge_model"] = "Average"
    return pd.concat([per_judge, avg], ignore_index=True)


def _judge_slug(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '-', name).lower().strip('-')


def per_judge_variants(df: pd.DataFrame, judges: list) -> list:
    """Returns [(label, filtered_df, slug), ...] for each judge plus Average."""
    variants = [(j, df[df["judge_model"] == j], _judge_slug(j)) for j in judges]
    variants.append(("Average", df, "average"))
    return variants


def to_markdown_table(df: pd.DataFrame) -> str:
    lines = []
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def pivot_to_markdown(df: pd.DataFrame, index_col: str, columns_col: str,
                      values_col: str = "mean") -> str:
    """Pivot a flat mean_score table so comparisons sit side-by-side as columns."""
    pivot = df.pivot_table(index=index_col, columns=columns_col,
                           values=values_col, aggfunc="mean").round(3)
    pivot.columns.name = None
    pivot = pivot.reset_index()
    return to_markdown_table(pivot)


def bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: str,
              color: str = "#4C72B0"):
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 4))
    bars = ax.bar(df[x_col].astype(str), df[y_col], color=color, edgecolor="white")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean score (0–5)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, df[y_col]):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def grouped_bar_chart(df: pd.DataFrame, x_col: str, hue_col: str, y_col: str,
                      title: str, path: str, hue_label: str = None):
    """Side-by-side grouped bar chart: x_col on x-axis, one bar group per hue_col value."""
    groups  = list(df[x_col].unique())
    hues    = list(df[hue_col].unique())
    n_groups = len(groups)
    n_hues   = len(hues)
    if n_hues == 0 or n_groups == 0:
        return

    bar_w = min(0.8 / n_hues, 0.35)
    x = np.arange(n_groups)

    palette = cm.Set2(np.linspace(0, 0.75, n_hues))

    fig, ax = plt.subplots(figsize=(max(7, n_groups * n_hues * 0.7 + 2), 4.5))

    for i, hue in enumerate(hues):
        subset = df[df[hue_col] == hue].set_index(x_col)[y_col]
        vals = [subset.get(g, float("nan")) for g in groups]
        offset = (i - n_hues / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.95,
                      label=str(hue), color=palette[i], edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean score (0–5)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups], rotation=30, ha="right")
    ax.legend(title=hue_label or hue_col,
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmap(pivot: pd.DataFrame, title: str, path: str):
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4),
                                    max(4, len(pivot) * 0.7)))
    im = ax.imshow(pivot.values, vmin=0, vmax=5, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if 1.5 < val < 4 else "white")
    plt.colorbar(im, ax=ax, label="Mean score")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze(results_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df):,} rows from {results_path}")

    # Exclude prompt strategies that are not meaningful for documents
    _doc_excluded = {"fewshot_nocontext", "fewshot_searchopt"}
    _before = len(df)
    df = df[~((df["text_type"] == "document") & (df["prompt_slug"].isin(_doc_excluded)))]
    print(f"Excluded {_before - len(df):,} document rows with prompts {_doc_excluded} from analysis (data not deleted)")

    required = {"score", "translation_model", "prompt_slug", "category",
                "text_type", "text_length_bucket", "judge_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Master CSV missing columns: {missing}")

    n_models  = df["translation_model"].nunique()
    n_prompts = df["prompt_slug"].nunique()
    n_judges  = df["judge_model"].nunique()

    sections = []

    def section(title: str, body: str):
        sections.append(f"## {title}\n\n{body}\n")

    def chart(df_, groupby, title, path, color="#4C72B0", hue_label="judge"):
        """Bar chart split by judge when multiple judges exist, plain bar otherwise."""
        if n_judges > 1:
            t_ = mean_score_by_judge(df_, groupby)
            grouped_bar_chart(t_, groupby[0], "judge_model", "mean", title, path, hue_label=hue_label)
        else:
            t_ = mean_score(df_, groupby)
            bar_chart(t_, groupby[0], "mean", title, path, color=color)
        return mean_score(df_, groupby)  # always return flat table for markdown

    # ── 1. Overall by translation model ───────────────────────────────────────
    t = chart(df, ["translation_model"], "Mean score by translation model",
              f"{output_dir}/by_translation_model.png")
    section("1. Translation model comparison",
            to_markdown_table(t) +
            "\n\n![](by_translation_model.png)")

    # ── 2. Overall by prompt strategy (split by text type) ────────────────────
    df_queries = df[df["text_type"] == "query"]
    df_docs_analysis = df[df["text_type"] == "document"]

    tq = chart(df_queries, ["prompt_slug"], "Mean score by prompt strategy (queries)",
               f"{output_dir}/by_prompt_slug_query.png", color="#DD8452")

    td = chart(df_docs_analysis, ["prompt_slug"], "Mean score by prompt strategy (documents)",
               f"{output_dir}/by_prompt_slug_document.png", color="#4C72B0")

    section("2. Prompt strategy comparison",
            "**Queries**\n\n" +
            to_markdown_table(tq) +
            "\n\n![](by_prompt_slug_query.png)\n\n"
            "**Documents**\n\n" +
            to_markdown_table(td) +
            "\n\n![](by_prompt_slug_document.png)")

    # ── 3. By dataset category ─────────────────────────────────────────────────
    t = chart(df, ["category"], "Mean score by dataset category",
              f"{output_dir}/by_category.png", color="#55A868")
    section("3. Performance by dataset category",
            to_markdown_table(t) +
            "\n\n![](by_category.png)")

    # ── 4. Query vs document ───────────────────────────────────────────────────
    t = chart(df, ["text_type"], "Mean score: query vs document",
              f"{output_dir}/by_text_type.png", color="#C44E52")
    section("4. Query vs document",
            to_markdown_table(t) +
            "\n\n![](by_text_type.png)")

    # ── 5. Text length effect ──────────────────────────────────────────────────
    # Queries have no segment data → "unknown"; restrict length analysis to documents.
    df_docs = df[df["text_type"] == "document"].copy()
    t = chart(df_docs, ["text_length_bucket"], "Mean score by text length (documents only)",
              f"{output_dir}/by_text_length.png", color="#8172B2")
    section("5. Text length effect (documents only; short = 1 segment, long = multiple segments)",
            to_markdown_table(t) +
            "\n\n> Queries are excluded here — they have no segment structure.\n\n"
            "![](by_text_length.png)")

    # ── 6. Judge model agreement ───────────────────────────────────────────────
    t = mean_score(df, ["judge_model"])
    bar_chart(t, "judge_model", "mean", "Mean score awarded by each judge",
              f"{output_dir}/by_judge_model.png", color="#937860")
    section("6. Judge model comparison (calibration & discrimination)",
            to_markdown_table(t) +
            "\n\n![](by_judge_model.png)\n\n"
            "> Low std → judge tends to give similar scores regardless of quality (poorly discriminating).\n"
            "> High std → judge is more sensitive to quality differences.")

    # ── 7. Category × model and category × prompt heatmaps ────────────────────
    judges = sorted(df["judge_model"].unique())
    variants = per_judge_variants(df, judges)

    body7 = ""
    if n_models > 1:
        body7 += "**By translation model**\n\n"
        t7m_avg = mean_score(df, ["category", "translation_model"])
        body7 += pivot_to_markdown(t7m_avg, "category", "translation_model") + "\n\n"
        for label, df_v, slug in variants:
            pivot_cm = (
                df_v.groupby(["category", "translation_model"])["score"]
                .mean().unstack("translation_model").round(3)
            )
            heatmap(pivot_cm, f"Mean score: category × translation model ({label})",
                    f"{output_dir}/heatmap_category_model_{slug}.png")
            t7m = mean_score(df_v, ["category", "translation_model"])
            grouped_bar_chart(t7m, "category", "translation_model", "mean",
                              f"Mean score: category × translation model ({label})",
                              f"{output_dir}/grouped_category_model_{slug}.png",
                              hue_label="translation model")
            body7 += f"**{label}**\n\n"
            body7 += f"![](grouped_category_model_{slug}.png)\n\n"
            body7 += f"![](heatmap_category_model_{slug}.png)\n\n"
    else:
        for label, df_v, slug in variants:
            pivot_cm = (
                df_v.groupby(["category", "translation_model"])["score"]
                .mean().unstack("translation_model").round(3)
            )
            heatmap(pivot_cm, f"Mean score: category × translation model ({label})",
                    f"{output_dir}/heatmap_category_model_{slug}.png")
            body7 += f"![](heatmap_category_model_{slug}.png)\n\n"

    if n_prompts > 1:
        body7 += "**By prompt strategy**\n\n"
        for tt_label, tt_key, tt_df in [("Queries", "query", df_queries), ("Documents", "document", df_docs_analysis)]:
            body7 += f"*{tt_label}*\n\n"
            t7p_avg = mean_score(tt_df, ["category", "prompt_slug"])
            body7 += pivot_to_markdown(t7p_avg, "category", "prompt_slug") + "\n\n"
            tt_variants = per_judge_variants(tt_df, judges)
            for label, df_v, slug in tt_variants:
                pivot_cp = (
                    df_v.groupby(["category", "prompt_slug"])["score"]
                    .mean().unstack("prompt_slug").round(3)
                )
                fname = f"heatmap_category_prompt_{tt_key}_{slug}"
                heatmap(pivot_cp, f"Mean score: category × prompt strategy — {tt_label} ({label})",
                        f"{output_dir}/{fname}.png")
                t7p = mean_score(df_v, ["category", "prompt_slug"])
                gfname = f"grouped_category_prompt_{tt_key}_{slug}"
                grouped_bar_chart(t7p, "category", "prompt_slug", "mean",
                                  f"Mean score: category × prompt strategy — {tt_label} ({label})",
                                  f"{output_dir}/{gfname}.png",
                                  hue_label="prompt strategy")
                body7 += f"**{label}**\n\n"
                body7 += f"![]({gfname}.png)\n\n"
                body7 += f"![]({fname}.png)\n\n"
    else:
        for tt_label, tt_key, tt_df in [("Queries", "query", df_queries), ("Documents", "document", df_docs_analysis)]:
            tt_variants = per_judge_variants(tt_df, judges)
            for label, df_v, slug in tt_variants:
                pivot_cp = (
                    df_v.groupby(["category", "prompt_slug"])["score"]
                    .mean().unstack("prompt_slug").round(3)
                )
                fname = f"heatmap_category_prompt_{tt_key}_{slug}"
                heatmap(pivot_cp, f"Mean score: category × prompt strategy — {tt_label} ({label})",
                        f"{output_dir}/{fname}.png")
                body7 += f"![]({fname}.png)\n\n"

    section("7. Category × model and category × prompt", body7)

    # ── 8. Best overall (model × prompt) combination ──────────────────────────
    t8_avg = mean_score(df, ["translation_model", "prompt_slug"])
    body8 = pivot_to_markdown(t8_avg, "translation_model", "prompt_slug") + "\n\n"
    for label, df_v, slug in variants:
        t8 = mean_score(df_v, ["translation_model", "prompt_slug"])
        grouped_bar_chart(t8, "translation_model", "prompt_slug", "mean",
                          f"Mean score: translation model × prompt strategy ({label})",
                          f"{output_dir}/by_model_prompt_{slug}.png",
                          hue_label="prompt strategy")
        body8 += f"**{label}**\n\n![](by_model_prompt_{slug}.png)\n\n"
    section("8. Best (translation model × prompt strategy) combinations", body8)

    # ── 9. Text type × prompt strategy ────────────────────────────────────────
    body9 = "> Does context help more for queries than documents?\n\n"
    t9_avg = mean_score(df, ["text_type", "prompt_slug"])
    body9 += pivot_to_markdown(t9_avg, "prompt_slug", "text_type") + "\n\n"
    for label, df_v, slug in variants:
        t9 = mean_score(df_v, ["text_type", "prompt_slug"])
        grouped_bar_chart(t9, "prompt_slug", "text_type", "mean",
                          f"Mean score: query vs document, by prompt strategy ({label})",
                          f"{output_dir}/by_texttype_prompt_{slug}.png",
                          hue_label="text type")
        body9 += f"**{label}**\n\n![](by_texttype_prompt_{slug}.png)\n\n"
    section("9. Text type × prompt strategy\n\n", body9)

    # ── 10. Text length × translation model ───────────────────────────────────
    body10 = "> Which model degrades least on long (multi-segment) documents? (documents only)\n\n"
    t10_avg = mean_score(df_docs, ["text_length_bucket", "translation_model"])
    body10 += pivot_to_markdown(t10_avg, "text_length_bucket", "translation_model") + "\n\n"
    variants_docs = per_judge_variants(df_docs, judges)
    for label, df_v, slug in variants_docs:
        t10 = mean_score(df_v, ["text_length_bucket", "translation_model"])
        grouped_bar_chart(t10, "text_length_bucket", "translation_model", "mean",
                          f"Mean score by text length × translation model ({label}, documents only)",
                          f"{output_dir}/by_length_model_{slug}.png",
                          hue_label="translation model")
        body10 += f"**{label}**\n\n![](by_length_model_{slug}.png)\n\n"
    section("10. Text length × translation model\n\n", body10)

    # ── Write report ──────────────────────────────────────────────────────────
    n_datasets = df["dataset_slug"].nunique() if "dataset_slug" in df.columns else "?"

    header = textwrap.dedent(f"""\
        # Translation Experiment — Analysis Report

        **Data:** {len(df):,} evaluated translations across
        {n_datasets} datasets,
        {n_models} translation models,
        {n_prompts} prompt strategies,
        {n_judges} judge models.

        Generated from: `{results_path}`

    """)

    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(sections))

    print(f"Analysis report written to {report_path}")
    print(f"Charts saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze translation experiment results.")
    parser.add_argument("--results_path", required=True, help="Path to the master results CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for charts and report.")
    args = parser.parse_args()
    analyze(args.results_path, args.output_dir)


if __name__ == "__main__":
    main()

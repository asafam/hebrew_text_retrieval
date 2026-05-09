"""
Analyzes eval prompt calibration results to determine whether the category-specialized
evaluation prompts (technical, QA) score translations differently from the general prompt.

Requires the output of:
  collect_results.py --calibration

Analysis questions answered:
  1. Score distribution — do specialized prompts score systematically higher or lower?
  2. Discrimination — which prompt has higher variance? (higher std = more sensitive)
  3. Inter-prompt agreement — for the same translation, do general and specialized agree?
     (Pearson r and mean absolute score difference per dataset)
  4. Where do they disagree most? — top translations where scores diverge
  5. Are specialized prompts more informative for their intended domains?
     (Is the score gap between specialized and general larger on target datasets?)

Usage:
  python src/translation/analyze_eval_prompts.py \
      --results_path outputs/translation/BeIR/results_eval_calibration.csv \
      --output_dir  outputs/translation/BeIR/analysis_eval_calibration
"""

import argparse
import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def bar_chart(df, x_col, y_col, title, path, color="#4C72B0", hue_col=None):
    if hue_col:
        groups = df[hue_col].unique()
        x_vals = df[x_col].unique()
        x_idx = {v: i for i, v in enumerate(x_vals)}
        width = 0.8 / len(groups)
        fig, ax = plt.subplots(figsize=(max(6, len(x_vals) * 1.2), 4))
        for g_idx, group in enumerate(groups):
            sub = df[df[hue_col] == group]
            offsets = [x_idx[v] + g_idx * width - (len(groups) - 1) * width / 2
                       for v in sub[x_col]]
            ax.bar(offsets, sub[y_col], width=width * 0.9, label=str(group))
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=30, ha="right")
        ax.legend(title=hue_col, fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 4))
        ax.bar(df[x_col].astype(str), df[y_col], color=color, edgecolor="white")
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Mean score (0–5)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def scatter(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, alpha=0.4, s=15)
    lims = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="perfect agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze(results_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df):,} rows")

    required = {"score", "eval_prompt_slug", "dataset_slug", "category",
                "text_type", "judge_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sections = []

    def section(title, body):
        sections.append(f"## {title}\n\n{body}\n")

    # ── 1. Score distribution per eval prompt ──────────────────────────────────
    t = (df.groupby("eval_prompt_slug")["score"]
         .agg(mean="mean", std="std", n="count")
         .reset_index()
         .round(3)
         .sort_values("mean", ascending=False))
    bar_chart(t, "eval_prompt_slug", "mean",
              "Mean score awarded by each eval prompt",
              f"{output_dir}/mean_score_by_eval_prompt.png", color="#DD8452")
    section("1. Score distribution per eval prompt",
            to_md_table(t) +
            "\n\n> High mean + low std → prompt is lenient but consistent.\n"
            "> Low mean + high std → prompt is strict and discriminating.\n\n"
            "![](mean_score_by_eval_prompt.png)")

    # ── 2. Discrimination (std) per eval prompt ────────────────────────────────
    # Per-document score std across prompt types — high std = prompts disagree
    if "_id" in df.columns:
        doc_std = (df.groupby(["_id", "text_type", "dataset_slug"])["score"]
                   .std()
                   .reset_index()
                   .rename(columns={"score": "score_std_across_prompts"}))
        mean_doc_std = (doc_std.groupby("dataset_slug")["score_std_across_prompts"]
                        .mean()
                        .reset_index()
                        .sort_values("score_std_across_prompts", ascending=False)
                        .round(3))
        section("2. Inter-prompt disagreement per dataset\n\n"
                "> Mean std of scores across eval prompts for the same translation.\n"
                "> High value → prompts disagree more for this dataset's translations.",
                to_md_table(mean_doc_std))

    # ── 3. Score by (dataset × eval prompt) — does specialization help? ────────
    t = (df.groupby(["dataset_slug", "eval_prompt_slug"])["score"]
         .mean()
         .reset_index()
         .round(3))
    pivot = t.pivot(index="dataset_slug", columns="eval_prompt_slug", values="score").round(3)

    # Add a column showing score gap: specialized minus general
    if "general" in pivot.columns:
        for col in ["technical", "qa"]:
            if col in pivot.columns:
                pivot[f"{col}_vs_general"] = (pivot[col] - pivot["general"]).round(3)

    section("3. Score by dataset × eval prompt\n\n"
            "> Positive `_vs_general` → specialized prompt scores higher than general.\n"
            "> Focus on whether the gap is larger on the intended target datasets.",
            to_md_table(pivot.reset_index()))

    # ── 4. Per-prompt score distributions across datasets ─────────────────────
    t2 = (df.groupby(["dataset_slug", "eval_prompt_slug"])["score"]
          .mean()
          .reset_index())
    bar_chart(t2, "dataset_slug", "score",
              "Mean score per dataset, broken down by eval prompt",
              f"{output_dir}/score_by_dataset_eval_prompt.png",
              hue_col="eval_prompt_slug")
    section("4. Score distribution across datasets per eval prompt",
            "![](score_by_dataset_eval_prompt.png)")

    # ── 5. Pairwise agreement: general vs technical, general vs qa ─────────────
    if "_id" in df.columns:
        # Pivot to wide format: one column per eval_prompt_slug
        wide = (df.groupby(["_id", "text_type", "dataset_slug", "judge_model",
                             "eval_prompt_slug"])["score"]
                .mean()
                .unstack("eval_prompt_slug")
                .reset_index()
                .dropna())

        agreement_sections = []
        for pair in [("general", "technical"), ("general", "qa"), ("technical", "qa")]:
            a, b = pair
            if a not in wide.columns or b not in wide.columns:
                continue
            r = np.corrcoef(wide[a], wide[b])[0, 1]
            mad = (wide[a] - wide[b]).abs().mean()
            agreement_sections.append(
                f"**{a} vs {b}:** Pearson r = {r:.3f},  mean |score diff| = {mad:.3f}"
            )
            scatter(wide[a], wide[b], a, b,
                    f"Score agreement: {a} vs {b}",
                    f"{output_dir}/agreement_{a}_vs_{b}.png")

        section("5. Pairwise inter-prompt score agreement\n\n"
                "> r close to 1.0 → prompts agree on which translations are good/bad.\n"
                "> Low r or high mean |diff| → prompts are measuring different things.",
                "\n\n".join(agreement_sections) +
                "\n\n![](agreement_general_vs_technical.png)"
                "\n\n![](agreement_general_vs_qa.png)")

    # ── 6. Worst disagreements ─────────────────────────────────────────────────
    if "_id" in df.columns and "wide" in dir():
        if "general" in wide.columns and "technical" in wide.columns:
            wide["max_gap"] = wide[["general", "technical", "qa"]
                                    if "qa" in wide.columns else ["general", "technical"]].max(axis=1) - \
                               wide[["general", "technical", "qa"]
                                    if "qa" in wide.columns else ["general", "technical"]].min(axis=1)
            top_disagreements = (wide.nlargest(20, "max_gap")
                                 [["_id", "dataset_slug", "text_type"] +
                                  [c for c in ["general", "technical", "qa"] if c in wide.columns] +
                                  ["max_gap"]]
                                 .round(3))
            section("6. Top 20 translations with largest inter-prompt score gap\n\n"
                    "> These are the cases where specialized and general prompts disagree most.\n"
                    "> Review their critiques to understand what each prompt is sensitive to.",
                    to_md_table(top_disagreements))

    # ── 7. Query vs document breakdown per eval prompt ─────────────────────────
    t = (df.groupby(["text_type", "eval_prompt_slug"])["score"]
         .mean()
         .reset_index()
         .round(3))
    section("7. Text type × eval prompt",
            to_md_table(t))

    # ── Write report ───────────────────────────────────────────────────────────
    header = textwrap.dedent(f"""\
        # Eval Prompt Calibration Report

        Compares the three evaluation prompts (general, technical, QA) on the same
        translations to determine whether specialized prompts add value over the general one.

        **Data:** {len(df):,} evaluated translations,
        {df['dataset_slug'].nunique()} datasets,
        {df['eval_prompt_slug'].nunique()} eval prompts,
        {df['judge_model'].nunique()} judge models.

        Generated from: `{results_path}`

    """)

    report_path = os.path.join(output_dir, "eval_prompt_calibration_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(sections))

    print(f"Calibration report written to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    analyze(args.results_path, args.output_dir)


if __name__ == "__main__":
    main()

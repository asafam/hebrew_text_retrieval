"""Per-dataset QA score curves and summary heatmap for the ladder pipeline."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_plots(run_dir: str, progress: dict, config: dict) -> None:
    """Render all plots (called after every shard). Overwrites files in place."""
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    threshold = config.get("qa", {}).get("min_score", 3.5)
    datasets = progress.get("datasets", {})
    for slug, entry in datasets.items():
        if entry.get("ladder_stage_scores"):
            _plot_dataset(slug, entry, threshold, plots_dir)
    _plot_summary_heatmap(datasets, threshold, plots_dir)


def _plot_dataset(slug: str, entry: dict, threshold: float, plots_dir: str) -> None:
    stage_scores = entry.get("ladder_stage_scores", {})
    stages = sorted(stage_scores.keys(), key=int)
    if not stages:
        return

    x = np.array([stage_scores[s].get("cumulative_q_rows", 0) for s in stages], dtype=float)
    q_means = np.array([stage_scores[s].get("q_score_mean") or float("nan") for s in stages], dtype=float)
    q_stds  = np.array([stage_scores[s].get("q_score_std")  or 0.0          for s in stages], dtype=float)
    d_means = np.array([stage_scores[s].get("d_score_mean") or float("nan") for s in stages], dtype=float)
    d_stds  = np.array([stage_scores[s].get("d_score_std")  or 0.0          for s in stages], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, q_means, color="steelblue",  marker="o", label="Query score")
    ax.fill_between(x, q_means - q_stds, q_means + q_stds, alpha=0.15, color="steelblue")
    ax.plot(x, d_means, color="darkorange", marker="s", label="Document score")
    ax.fill_between(x, d_means - d_stds, d_means + d_stds, alpha=0.15, color="darkorange")
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")

    if entry.get("ladder_stopped") and len(x) > 0:
        ax.axvline(x[-1], color="dimgray", linestyle=":", linewidth=1.2, label="Stopped here")

    ax.set_xlim(left=0)
    ax.set_ylim(1, 5.3)
    ax.set_xlabel("Cumulative rows translated")
    ax.set_ylabel("QA Score (1–5)")
    ax.set_title(f"{slug} — QA scores by shard")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{slug}.png"), dpi=120)
    plt.close(fig)


def _plot_summary_heatmap(datasets: dict, threshold: float, plots_dir: str) -> None:
    slugs = [s for s, e in datasets.items() if e.get("ladder_stage_scores")]
    if not slugs:
        return
    max_stages = max(len(datasets[s].get("ladder_stage_scores", {})) for s in slugs)
    if max_stages == 0:
        return

    data = np.full((len(slugs), max_stages), np.nan)
    for i, slug in enumerate(slugs):
        for s_str, sd in datasets[slug].get("ladder_stage_scores", {}).items():
            s = int(s_str)
            if s < max_stages:
                qm = sd.get("q_score_mean")
                dm = sd.get("d_score_mean")
                if qm is not None and dm is not None:
                    data[i, s] = (qm + dm) / 2

    fig, ax = plt.subplots(figsize=(max(8, max_stages * 1.8), max(4, len(slugs) * 0.75)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")

    for i in range(len(slugs)):
        for j in range(max_stages):
            val = data[i, j]
            if not np.isnan(val):
                txt = f"{val:.2f}"
                color = "white" if val < 3.0 else "black"
            else:
                txt = "—"
                color = "gray"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(max_stages))
    ax.set_xticklabels([f"shard {j}" for j in range(max_stages)], fontsize=9)
    ax.set_yticks(range(len(slugs)))
    ax.set_yticklabels(slugs, fontsize=8)
    ax.set_title("Ladder QA — avg(query, document) score per shard")
    plt.colorbar(im, ax=ax, label="Score 1–5")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "summary.png"), dpi=120)
    plt.close(fig)

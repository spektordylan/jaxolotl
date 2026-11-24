"""Generate publication-quality figures for Uncertainty-Aware SemLTL noise evaluation.

Produces:
1. Figure 1: True task success/acceptance (SR/AV) vs. proposition observation noise.
2. Figure 2: True task violations vs. proposition observation noise.
3. Figure 3: Per-formula robustness heatmap across noise levels.
4. Figure 4: Normalized performance degradation relative to p=0 baseline.
5. Figure 5: Perceived vs. true automaton state disagreement rate vs. noise.

All figures are saved in both PNG (300 DPI) and PDF formats.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
)

FORMULA_HEATMAP_LABELS = {
    0: r"$\varphi_0$ Sequential Avoidance",
    1: r"$\varphi_1$ Sequential Avoidance II",
    2: r"$\varphi_2$ Branching Reachability",
    3: r"$\varphi_3$ Nested Until",
    4: r"$\varphi_4$ Global Safety",
    5: r"$\varphi_5$ Safety + Epsilon",
    6: r"$\varphi_6$ Conditional Branching",
    7: r"$\varphi_7$ Deep Sequence",
    8: r"$\varphi_8$ Safety + Sequence",
    9: r"$\varphi_9$ Multiple Avoidance",
}

# Semantic categorization and display labels for the 10 finite-LTL evaluation formulas
FORMULA_METADATA = {
    0: {
        "name": r"$\varphi_0$",
        "label": r"$\varphi_0$  Sequential Avoidance",
        "type": "Seq-Avoid",
        "short": "F(a & (!b U c)) & F d",
    },
    1: {
        "name": r"$\varphi_1$",
        "label": r"$\varphi_1$  Sequential Avoidance II",
        "type": "Seq-Avoid",
        "short": "Fd & (!f U (d & Fb))",
    },
    2: {
        "name": r"$\varphi_2$",
        "label": r"$\varphi_2$  Branching Reachability",
        "type": "Branch-Reach",
        "short": "F((a|c|j)&Fb) & F(c&Fd) & Fk",
    },
    3: {
        "name": r"$\varphi_3$",
        "label": r"$\varphi_3$  Nested Until",
        "type": "Deep-Until",
        "short": "!a U (b & (!c U (d & (!e U f))))",
    },
    4: {
        "name": r"$\varphi_4$",
        "label": r"$\varphi_4$  Global Safety",
        "type": "Global-Safety",
        "short": "G!(k|l) & F(a & F(e & F(i & Fd)))",
    },
    5: {
        "name": r"$\varphi_5$",
        "label": r"$\varphi_5$  Safety + Epsilon",
        "type": "Safety+Eps",
        "short": "G!(a|b|c|d) & G(h=>!fUg) & Fg & F(h&Fi)",
    },
    6: {
        "name": r"$\varphi_6$",
        "label": r"$\varphi_6$  Conditional Branching",
        "type": "Cond-Branch",
        "short": "F((a|l) & (a=>(!b U (c&Fd))) & (l=>Fg))",
    },
    7: {
        "name": r"$\varphi_7$",
        "label": r"$\varphi_7$  Deep Sequence",
        "type": "Deep-Sequence",
        "short": "F(a & F(b & ... & Fh)) [8 steps]",
    },
    8: {
        "name": r"$\varphi_8$",
        "label": r"$\varphi_8$  Safety + Sequence",
        "type": "Safety+Seq",
        "short": "G!l & F(i & F(e & ... & Fc)) [6 steps]",
    },
    9: {
        "name": r"$\varphi_9$",
        "label": r"$\varphi_9$  Multiple Avoidance",
        "type": "Multi-Avoid",
        "short": "!(b|d|f|h|j) U (l & F((a|k) & (a=>Fc)))",
    },
}


def plot_success_vs_noise(summary_df: pd.DataFrame, output_dir: Path):
    """Figure 1: Overall True Task Success/Acceptance Rate (SR/AV) vs. Noise."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    task_set_df = summary_df[
        (summary_df["scope"] == "task_set") & (summary_df["metric"] == "return")
    ].sort_values("noise_prob")

    x = task_set_df["noise_prob"].values
    y = task_set_df["mean"].values
    y_lower = task_set_df["ci_lower"].values
    y_upper = task_set_df["ci_upper"].values

    # Primary baseline: Noisy SemLTL (Frozen Policy)
    color = "#1f77b4"
    ax.plot(
        x,
        y,
        marker="o",
        color=color,
        label="SemLTL Baseline (Frozen Policy)",
        zorder=4,
    )
    ax.fill_between(
        x,
        y_lower,
        y_upper,
        color=color,
        alpha=0.22,
        label="95% Confidence Interval",
        zorder=3,
    )

    # Reference point at zero noise
    ax.axhline(
        y[0],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"Zero-Noise Baseline ({y[0]:.3f})",
    )

    ax.set_xlabel("Proposition Observation Noise Probability ($p$)")
    ax.set_ylabel("True Task Success / Acceptance Rate (SR/AV)")
    ax.set_title("Task Success under Categorical Proposition Noise")
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_dir / "success_vs_noise.png", dpi=300)
    fig.savefig(output_dir / "success_vs_noise.pdf")
    plt.close(fig)


def plot_violations_vs_noise(summary_df: pd.DataFrame, output_dir: Path):
    """Figure 2: Overall True Violation Rate vs. Noise."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    viol_df = summary_df[
        (summary_df["scope"] == "task_set") & (summary_df["metric"] == "violations")
    ].sort_values("noise_prob")

    x = viol_df["noise_prob"].values
    y = viol_df["mean"].values
    y_lower = viol_df["ci_lower"].values
    y_upper = viol_df["ci_upper"].values

    color = "#d62728"
    ax.plot(
        x,
        y,
        marker="s",
        color=color,
        label="SemLTL Baseline Violations",
        zorder=4,
    )
    ax.fill_between(
        x,
        y_lower,
        y_upper,
        color=color,
        alpha=0.22,
        label="95% Confidence Interval",
        zorder=3,
    )

    ax.axhline(
        y[0],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"Zero-Noise Violation Rate ({y[0]:.3f})",
    )

    ax.set_xlabel("Proposition Observation Noise Probability ($p$)")
    ax.set_ylabel("True Task Violation Rate (Sink Entry)")
    ax.set_title("Safety Violations under Categorical Proposition Noise")
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(-0.02, max(0.65, np.max(y_upper) + 0.05))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_dir / "violations_vs_noise.png", dpi=300)
    fig.savefig(output_dir / "violations_vs_noise.pdf")
    plt.close(fig)


def plot_per_formula_heatmap(summary_df: pd.DataFrame, output_dir: Path):
    """Figure 3: Per-Formula Robustness Heatmap across noise levels."""
    formula_df = summary_df[
        (summary_df["scope"] == "formula") & (summary_df["metric"] == "return")
    ].copy()

    # Pivot table: rows = formula_idx, columns = noise_prob, values = mean SR/AV
    pivot_df = formula_df.pivot(
        index="formula_idx", columns="noise_prob", values="mean"
    )
    pivot_df = pivot_df.sort_index()

    row_labels = [FORMULA_HEATMAP_LABELS[idx] for idx in pivot_df.index]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "True Task Success / Acceptance Rate"},
        yticklabels=row_labels,
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
    )

    ax.set_title("SemLTL Robustness by Temporal Specification")
    ax.set_xlabel("Proposition Observation Noise Probability ($p$)")
    ax.set_ylabel("LTL Specification")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(output_dir / "per_formula_heatmap.png", dpi=300)
    fig.savefig(output_dir / "per_formula_heatmap.pdf")
    plt.close(fig)


def plot_normalized_degradation(summary_df: pd.DataFrame, output_dir: Path):
    """Figure 4: Relative Performance Degradation SR(p) / SR(0)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    formula_df = summary_df[
        (summary_df["scope"] == "formula") & (summary_df["metric"] == "return")
    ].copy()

    # Normalize by p=0 per formula
    p0_returns = formula_df[formula_df["noise_prob"] == 0.0].set_index("formula_idx")[
        "mean"
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for f_idx in range(10):
        f_data = formula_df[formula_df["formula_idx"] == f_idx].sort_values(
            "noise_prob"
        )
        base_val = p0_returns.get(f_idx, 1.0)
        norm_vals = f_data["mean"].values / max(base_val, 1e-6)
        label = FORMULA_METADATA[f_idx]["label"]
        ax.plot(
            f_data["noise_prob"].values,
            norm_vals,
            marker=".",
            alpha=0.75,
            linewidth=1.4,
            color=colors[f_idx],
            label=label,
        )

    # Also plot overall mean normalized
    task_df = summary_df[
        (summary_df["scope"] == "task_set") & (summary_df["metric"] == "return")
    ].sort_values("noise_prob")
    overall_norm = task_df["mean"].values / task_df["mean"].values[0]
    ax.plot(
        task_df["noise_prob"].values,
        overall_norm,
        marker="o",
        color="black",
        linewidth=2.8,
        label="Overall Mean (Normalized)",
        zorder=5,
    )

    ax.set_xlabel("Proposition Observation Noise Probability ($p$)")
    ax.set_ylabel("Normalized Success Rate ($SR(p) / SR(0)$)")
    ax.set_title("Relative Performance Degradation across Specifications")
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "normalized_degradation.png", dpi=300)
    fig.savefig(output_dir / "normalized_degradation.pdf")
    plt.close(fig)


def plot_disagreement_vs_noise(summary_df: pd.DataFrame, output_dir: Path):
    """Figure 5: Perceived-True Automaton State Disagreement vs. Observation Noise.

    Note: This measures episode-level disagreement D_episode = 1[exists t: q_t^true != q_t^perc],
    representing the fraction of episodes where the perceived automaton state diverged from
    the ground-truth automaton state at any timestep, distinct from average per-step disagreement.
    """
    disag_df = summary_df[
        (summary_df["scope"] == "task_set") & (summary_df["metric"] == "disagreement")
    ].sort_values("noise_prob")

    if disag_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = disag_df["noise_prob"].values
    y = disag_df["mean"].values
    y_lower = disag_df["ci_lower"].values
    y_upper = disag_df["ci_upper"].values

    color = "#9467bd"
    ax.plot(
        x,
        y,
        marker="^",
        color=color,
        label=r"Episode Disagreement ($q_{\mathrm{true}} \neq q_{\mathrm{perc}}$ at any step)",
        zorder=4,
    )
    ax.fill_between(
        x,
        y_lower,
        y_upper,
        color=color,
        alpha=0.22,
        label="95% Confidence Interval",
        zorder=3,
    )

    ax.set_xlabel("Proposition Observation Noise Probability ($p$)")
    ax.set_ylabel("Episode State Disagreement Rate")
    ax.set_title("Perceived–True Automaton State Disagreement")
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_dir / "disagreement_vs_noise.png", dpi=300)
    fig.savefig(output_dir / "disagreement_vs_noise.pdf")
    plt.close(fig)


def generate_all_figures(
    summary_csv_path: Path | str,
    _raw_csv_path: Path | str | None = None,
    output_dir: Path | str = "experiments/belief_semltl/figures",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv_path)

    plot_success_vs_noise(summary_df, output_dir)
    plot_violations_vs_noise(summary_df, output_dir)
    plot_per_formula_heatmap(summary_df, output_dir)
    plot_normalized_degradation(summary_df, output_dir)
    plot_disagreement_vs_noise(summary_df, output_dir)


if __name__ == "__main__":
    import sys

    summary_csv = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "experiments/belief_semltl/results/summary.csv"
    )
    raw_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "experiments/belief_semltl/results/noise_sweep.csv"
    )
    generate_all_figures(summary_csv, raw_csv)

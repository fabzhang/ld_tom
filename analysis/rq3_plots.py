"""
analysis/rq3_plots.py

Visualization for RQ3 results.

Figures:
  1. Challenge calibration curves (one line per agent)
  2. Implied tau bar chart with CI
  3. Strategy fingerprint PCA scatter
  4. Non-CH fraction bar chart + top state examples
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


AGENT_COLORS = {
    "random":    "#9e9e9e",
    "heuristic": "#2196F3",
    "bayesian":  "#4CAF50",
    "ppo":       "#FF5722",
    "he2016":    "#9C27B0",
    "tomnet":    "#FF9800",
}

AGENT_ORDER = ["random", "heuristic", "bayesian", "ppo", "he2016"]

LEVEL_LABELS = {0: "Level-0 (uniform)", 1: "Level-1 (heuristic)", 2: "Level-2 (bayesian)"}


def _agent_color(name):
    return AGENT_COLORS.get(name, "#333333")


# ---------------------------------------------------------------------------
# Figure 1: Challenge calibration curves
# ---------------------------------------------------------------------------

def plot_challenge_calibration(calibration_results, output_path,
                                show_level_refs=True, split_by_position=False):
    """
    Plot challenge rate vs P(bid valid) for each agent.
    
    Args:
        calibration_results: dict from e3a_calibration.json
        show_level_refs: draw vertical line at p=0.25 (Level-1 threshold)
        split_by_position: if True, plot separate lines for player_0 and player_1
    """
    agents_in_order = [a for a in AGENT_ORDER if a in calibration_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    for agent_name in agents_in_order:
        cal = calibration_results[agent_name]
        x = np.array(cal["bin_centers"])
        y = np.array([v if v is not None else np.nan for v in cal["challenge_rates"]])
        counts = np.array(cal["bin_counts"])

        # Only plot bins with enough data
        mask = counts >= 10
        color = _agent_color(agent_name)

        ax.plot(x[mask], y[mask], "o-", color=color, label=agent_name.capitalize(),
                linewidth=2, markersize=4)

        if split_by_position and "by_position" in cal:
            pos_data = cal["by_position"]
            xp = np.array(pos_data.get("p0_bin_centers", x[:10]))
            yp0 = np.array([v if v is not None else np.nan
                            for v in pos_data["p0_challenge_rates"]])
            yp1 = np.array([v if v is not None else np.nan
                            for v in pos_data["p1_challenge_rates"]])
            ax.plot(xp, yp0, "--", color=color, alpha=0.5, linewidth=1.5,
                    label=f"{agent_name} (P0)")
            ax.plot(xp, yp1, ":", color=color, alpha=0.5, linewidth=1.5,
                    label=f"{agent_name} (P1)")

    if show_level_refs:
        ax.axvline(0.25, color="#2196F3", linestyle="--", alpha=0.4, label="L1 threshold (0.25)")
        ax.axvline(0.30, color="#4CAF50", linestyle=":", alpha=0.4, label="L2 threshold (0.30)")

    ax.set_xlabel("P(bid valid | uniform opponent dice)", fontsize=12)
    ax.set_ylabel("Challenge rate", fontsize=12)
    ax.set_title("Challenge Calibration by Agent", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_challenge_calibration_with_positional(calibration_results, output_path):
    """Plot calibration split by player position — reveals PPO positional bias."""
    agents_in_order = [a for a in AGENT_ORDER if a in calibration_results]
    n_agents = len(agents_in_order)
    fig, axes = plt.subplots(1, n_agents, figsize=(4 * n_agents, 4), sharey=True)
    if n_agents == 1:
        axes = [axes]

    for ax, agent_name in zip(axes, agents_in_order):
        cal = calibration_results[agent_name]
        color = _agent_color(agent_name)

        x = np.array(cal["bin_centers"])
        y = np.array([v if v is not None else np.nan for v in cal["challenge_rates"]])
        counts = np.array(cal["bin_counts"])
        mask = counts >= 5

        ax.plot(x[mask], y[mask], "k-", label="Overall", linewidth=2)

        if "by_position" in cal:
            pos = cal["by_position"]
            xp = np.array(pos["p0_bin_centers"])
            yp0 = np.array([v if v is not None else np.nan for v in pos["p0_challenge_rates"]])
            yp1 = np.array([v if v is not None else np.nan for v in pos["p1_challenge_rates"]])
            ax.plot(xp, yp0, color="#E91E63", linestyle="--", label="Player 0", linewidth=1.5)
            ax.plot(xp, yp1, color="#03A9F4", linestyle=":", label="Player 1", linewidth=1.5)

        ax.axvline(0.25, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(agent_name.capitalize(), color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("P(bid valid)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Challenge rate")
    fig.suptitle("Challenge Calibration by Player Position", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Tau bar chart
# ---------------------------------------------------------------------------

def plot_tau_comparison(tau_results, output_path):
    """Horizontal bar chart of implied tau per agent with bootstrap CI."""
    agents_in_order = [a for a in AGENT_ORDER if a in tau_results]

    fig, ax = plt.subplots(figsize=(7, 4))

    taus = [tau_results[a]["tau_bootstrap_mean"] for a in agents_in_order]
    lo = [tau_results[a]["tau_ci_lower"] for a in agents_in_order]
    hi = [tau_results[a]["tau_ci_upper"] for a in agents_in_order]
    colors = [_agent_color(a) for a in agents_in_order]

    yerr_lo = [t - l for t, l in zip(taus, lo)]
    yerr_hi = [h - t for t, h in zip(taus, hi)]

    x = np.arange(len(agents_in_order))
    bars = ax.bar(x, taus, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.errorbar(x, taus, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="black", capsize=5, linewidth=2)

    # Human range band (Camerer 2004: tau ~ 1-2)
    ax.axhspan(1.0, 2.0, alpha=0.10, color="green", label="Human range (Camerer 2004)")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(2.0, color="green", linestyle="--", alpha=0.4, linewidth=1)

    for bar, tau_val in zip(bars, taus):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{tau_val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents_in_order], fontsize=11)
    ax.set_ylabel("Implied τ (Poisson CH model)", fontsize=12)
    ax.set_title("Implied Level-k Sophistication per Agent", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(taus) * 1.3 + 0.5)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: KL table heatmap
# ---------------------------------------------------------------------------

def plot_kl_heatmap(kl_table, output_path):
    """Heatmap of mean KL divergence: agent (rows) vs Level-k (cols)."""
    import matplotlib.colors as mcolors

    agents_in_order = [a for a in AGENT_ORDER if a in kl_table]
    K_vals = sorted({k for v in kl_table.values() for k in v.keys()})
    matrix = np.array([[kl_table[a].get(k, np.nan) for k in K_vals]
                        for a in agents_in_order])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean KL (nats)")

    ax.set_xticks(range(len(K_vals)))
    ax.set_xticklabels([f"Level-{k}" for k in K_vals], fontsize=11)
    ax.set_yticks(range(len(agents_in_order)))
    ax.set_yticklabels([a.capitalize() for a in agents_in_order], fontsize=11)
    ax.set_title("KL Divergence: Agent vs CH Level-k", fontsize=12)

    for i in range(len(agents_in_order)):
        for j in range(len(K_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=text_color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: PCA scatter
# ---------------------------------------------------------------------------

def plot_fingerprint_pca(fingerprint_path, tau_results, output_path):
    """2D PCA scatter of agent strategy fingerprints."""
    with open(fingerprint_path) as f:
        data = json.load(f)

    pca_data = data["pca"]
    agent_names = pca_data["agent_names"]
    coords = np.array(pca_data["pca_coords"])
    evr = pca_data["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, name in enumerate(agent_names):
        color = _agent_color(name)
        x, y = coords[i]
        tau = tau_results.get(name, {}).get("tau_bootstrap_mean", None)
        tau_str = f"\nτ={tau:.1f}" if tau is not None else ""
        ax.scatter(x, y, s=200, color=color, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(f"{name.capitalize()}{tau_str}", (x, y),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color, fontweight="bold")

    # Draw Level-k reference labels in legend
    patches = [mpatches.Patch(color=_agent_color(a), label=a.capitalize())
               for a in AGENT_ORDER if a in agent_names]
    ax.legend(handles=patches, fontsize=9, loc="best")

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Strategy Fingerprint PCA", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 5: Non-CH fraction
# ---------------------------------------------------------------------------

def plot_non_ch_fraction(non_ch_results, output_path):
    """Bar chart of fraction of non-CH decisions per agent."""
    agents_in_order = [a for a in AGENT_ORDER if a in non_ch_results]
    fracs = [non_ch_results[a]["non_ch_fraction"] for a in agents_in_order]
    colors = [_agent_color(a) for a in agents_in_order]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(agents_in_order))
    bars = ax.bar(x, fracs, color=colors, alpha=0.85, edgecolor="white")

    for bar, f in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents_in_order], fontsize=11)
    ax.set_ylabel("Fraction of non-CH decisions\n(min KL to any level > 0.5 nats)", fontsize=10)
    ax.set_title("Non-CH Behavior Prevalence per Agent", fontsize=12)
    ax.set_ylim(0, max(fracs) * 1.25 + 0.02)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    def load(fname):
        with open(os.path.join(results_dir, fname)) as f:
            return json.load(f)

    cal = load("e3a_calibration.json")
    tau = load("e3b_tau_fitting.json")
    kl = load("e3b_kl_table.json")
    nc = load("e3d_non_ch_states.json")
    fp_path = os.path.join(results_dir, "e3c_fingerprints.json")

    plot_challenge_calibration(cal, os.path.join(plots_dir, "fig_e3a_calibration.pdf"))
    plot_challenge_calibration_with_positional(
        cal, os.path.join(plots_dir, "fig_e3a_calibration_by_position.pdf"))
    plot_tau_comparison(tau, os.path.join(plots_dir, "fig_e3b_tau.pdf"))
    plot_kl_heatmap(kl, os.path.join(plots_dir, "fig_e3b_kl_heatmap.pdf"))
    if os.path.exists(fp_path):
        plot_fingerprint_pca(fp_path, tau, os.path.join(plots_dir, "fig_e3c_pca.pdf"))
    plot_non_ch_fraction(nc, os.path.join(plots_dir, "fig_e3d_non_ch.pdf"))

    print(f"\nAll RQ3 plots saved to {plots_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/rq3/")
    parser.add_argument("--plots_dir", default="results/rq3/plots/")
    args = parser.parse_args()
    generate_all_plots(args.results_dir, args.plots_dir)

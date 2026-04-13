"""
rq1_plots.py — Visualization for RQ1 experiments.

Generates:
  - Win-rate heatmap (6×6)
  - Elo ranking bar chart
  - Directed dominance graph
  - Population sweep line plot
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


AGENT_ORDER = ["random", "heuristic", "bayesian", "ppo", "he2016", "tomnet"]
AGENT_LABELS = {
    "random": "Random",
    "heuristic": "Heuristic",
    "bayesian": "Bayesian",
    "ppo": "PPO",
    "he2016": "He2016\n(DRON-MoE)",
    "tomnet": "ToMnet",
}


def _ordered_names(names):
    """Sort agent names by canonical ToM level order."""
    ordered = [n for n in AGENT_ORDER if n in names]
    ordered += [n for n in names if n not in AGENT_ORDER]
    return ordered


def plot_win_rate_heatmap(matrix: np.ndarray, agent_names: list,
                          save_path: Optional[str] = None,
                          title: str = "Win Rate Matrix (row agent vs column agent)"):
    """
    Plot win-rate heatmap. matrix[i,j] = win rate of agent i vs agent j.
    """
    names = _ordered_names(agent_names)
    idx = [agent_names.index(n) for n in names]
    m = matrix.values[np.ix_(idx, idx)]
    labels = [AGENT_LABELS.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(
            m,
            annot=True,
            fmt=".0%",
            cmap="RdYlGn",
            vmin=0, vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Win rate"},
            mask=np.eye(len(names), dtype=bool),
        )
        # Grey out diagonal
        for i in range(len(names)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="lightgrey", lw=0))
    else:
        im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Win rate")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    ax.text(j, i, f"{m[i,j]:.0%}", ha="center", va="center",
                            fontsize=9, color="black")

    ax.set_title(title)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved heatmap to {save_path}")
    return fig


def plot_elo_ranking(elo_dict: dict, save_path: Optional[str] = None,
                     title: str = "Elo Ratings"):
    """Horizontal bar chart of Elo ratings sorted by score."""
    names = _ordered_names(list(elo_dict.keys()))
    scores = [elo_dict[n] for n in names]
    labels = [AGENT_LABELS.get(n, n) for n in names]

    # Sort descending
    sorted_pairs = sorted(zip(scores, labels, names), reverse=True)
    scores_s, labels_s, _ = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores_s)))
    bars = ax.barh(range(len(scores_s)), scores_s, color=colors)
    ax.set_yticks(range(len(scores_s)))
    ax.set_yticklabels(labels_s)
    ax.set_xlabel("Elo Rating")
    ax.set_title(title)

    for i, (bar, score) in enumerate(zip(bars, scores_s)):
        ax.text(score + 5, i, f"{score:.0f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved Elo chart to {save_path}")
    return fig


def plot_dominance_graph(win_rate_matrix: np.ndarray, agent_names: list,
                          threshold: float = 0.55,
                          save_path: Optional[str] = None,
                          title: str = "Dominance Graph (edge: A→B if A wins >55%)"):
    """Directed graph: A→B if win_rate_matrix[A,B] > threshold."""
    if not HAS_NETWORKX:
        print("[plot] networkx not available, skipping dominance graph")
        return None

    names = _ordered_names(agent_names)
    idx_map = {n: agent_names.index(n) for n in names}
    labels = {n: AGENT_LABELS.get(n, n) for n in names}

    G = nx.DiGraph()
    G.add_nodes_from(names)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            wr = win_rate_matrix.values[idx_map[a], idx_map[b]]
            if wr > threshold:
                G.add_edge(a, b, weight=wr)

    # Detect cycles
    cycles = list(nx.simple_cycles(G))
    cycle_edges = set()
    for cycle in cycles:
        for k in range(len(cycle)):
            cycle_edges.add((cycle[k], cycle[(k+1) % len(cycle)]))

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    edge_colors = ["red" if (u, v) in cycle_edges else "steelblue"
                   for u, v in G.edges()]
    edge_widths = [2.5 if (u, v) in cycle_edges else 1.5
                   for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="white",
                           edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=20, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.0%}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    if cycle_edges:
        red_patch = mpatches.Patch(color="red", label="Cyclic dominance edge")
        ax.legend(handles=[red_patch], loc="upper left")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved dominance graph to {save_path}")
    return fig


def plot_population_sweep(sweep_results: list, focal_agent_name: str,
                           save_path: Optional[str] = None):
    """
    Line plot: focal agent win rate vs fraction of each opponent type.

    sweep_results: output of run_population_sweep() — list of
      {"mix": {name: frac}, "focal_win_rate": float}
    """
    if not sweep_results:
        return None

    opp_names = list(sweep_results[0]["mix"].keys())
    fig, axes = plt.subplots(1, len(opp_names), figsize=(4 * len(opp_names), 4),
                              sharey=True)
    if len(opp_names) == 1:
        axes = [axes]

    focal_label = AGENT_LABELS.get(focal_agent_name, focal_agent_name)

    for ax, opp_name in zip(axes, opp_names):
        fracs = [r["mix"][opp_name] for r in sweep_results]
        win_rates = [r["focal_win_rate"] for r in sweep_results]
        # Sort by fraction
        sorted_pairs = sorted(zip(fracs, win_rates))
        fracs_s, wr_s = zip(*sorted_pairs)

        ax.plot(fracs_s, wr_s, "o-", markersize=6)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Fraction of {AGENT_LABELS.get(opp_name, opp_name)}")
        ax.set_ylabel("Win rate" if ax == axes[0] else "")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(f"{focal_label} vs {AGENT_LABELS.get(opp_name, opp_name)}")

    plt.suptitle(f"E1b: {focal_label} win rate vs population composition")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved population sweep to {save_path}")
    return fig


def generate_all_rq1_plots(results_dir: str = "results/rq1",
                            plots_dir: str = "results/rq1/plots"):
    """Load all RQ1 result files and generate all figures."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.elo import compute_win_rate_matrix, compute_elo

    os.makedirs(plots_dir, exist_ok=True)

    # Load E1a results
    e1a_path = os.path.join(results_dir, "e1a_results.json")
    if not os.path.exists(e1a_path):
        print(f"E1a results not found at {e1a_path}")
        return

    with open(e1a_path) as f:
        raw = json.load(f)
    results = {tuple(k.split("|")): tuple(v) for k, v in raw.items()}

    # Infer agent names
    all_names = list({n for pair in results for n in pair})
    agent_names = _ordered_names(all_names)

    matrix = compute_win_rate_matrix(results, agent_names)
    elo = compute_elo(results)

    plot_win_rate_heatmap(matrix, agent_names,
                          save_path=os.path.join(plots_dir, "e1a_heatmap.pdf"))
    plot_elo_ranking(elo, save_path=os.path.join(plots_dir, "e1a_elo.pdf"))
    plot_dominance_graph(matrix, agent_names, threshold=0.55,
                         save_path=os.path.join(plots_dir, "e1a_dominance_graph.pdf"))

    # Load and plot E1b sweeps
    import glob
    for sweep_file in glob.glob(os.path.join(results_dir, "e1b_*_sweep.json")):
        with open(sweep_file) as f:
            data = json.load(f)
        focal = data["focal"]
        plot_population_sweep(
            data["results"], focal_agent_name=focal,
            save_path=os.path.join(plots_dir, f"e1b_{focal}_sweep.pdf"),
        )

    print(f"\nAll RQ1 plots saved to {plots_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/rq1")
    parser.add_argument("--plots_dir", default="results/rq1/plots")
    args = parser.parse_args()
    generate_all_rq1_plots(args.results_dir, args.plots_dir)

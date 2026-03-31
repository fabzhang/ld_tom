"""
analysis/elo.py — Elo ratings and win-rate matrix computation for RQ1 tournament.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_win_rate_matrix(
    results: Dict[Tuple[str, str], Tuple[int, int]],
    agent_names: list[str],
) -> pd.DataFrame:
    """
    Build a win-rate matrix from pairwise results.

    Args:
        results: {(A, B): (A_wins, B_wins)} — games played between every pair.
        agent_names: Ordered list of agent names (defines row/col order).

    Returns:
        DataFrame where matrix[i][j] = win rate of agent_i against agent_j.
        Diagonal is NaN (no self-play).
    """
    n = len(agent_names)
    idx = {name: i for i, name in enumerate(agent_names)}
    matrix = np.full((n, n), np.nan)

    for (a, b), (a_wins, b_wins) in results.items():
        total = a_wins + b_wins
        if total == 0:
            continue
        i, j = idx[a], idx[b]
        matrix[i][j] = a_wins / total
        matrix[j][i] = b_wins / total

    return pd.DataFrame(matrix, index=agent_names, columns=agent_names)


def compute_elo(
    results: Dict[Tuple[str, str], Tuple[int, int]],
    K: float = 32.0,
    initial: float = 1000.0,
    n_iterations: int = 1000,
) -> Dict[str, float]:
    """
    Compute Elo ratings via iterative updates over all pairwise results.

    Each (A_wins, B_wins) pair is expanded into individual game outcomes,
    then processed in shuffled order.

    Args:
        results: {(A, B): (A_wins, B_wins)}.
        K: Elo K-factor (learning rate per game).
        initial: Starting Elo for all agents.
        n_iterations: Number of full passes over all games.

    Returns:
        {agent_name: elo_rating}
    """
    # Collect all agents
    agents: set[str] = set()
    for a, b in results:
        agents.add(a)
        agents.add(b)
    elo = {a: initial for a in agents}

    # Expand results into individual game outcomes
    games: list[Tuple[str, str, float]] = []  # (winner, loser, score_for_winner)
    for (a, b), (a_wins, b_wins) in results.items():
        for _ in range(a_wins):
            games.append((a, b, 1.0))
        for _ in range(b_wins):
            games.append((b, a, 1.0))

    rng = np.random.default_rng(42)
    for _ in range(n_iterations):
        order = rng.permutation(len(games))
        for idx in order:
            winner, loser, _ = games[idx]
            e_w = elo[winner]
            e_l = elo[loser]
            expected_w = 1.0 / (1.0 + 10 ** ((e_l - e_w) / 400.0))
            delta = K * (1.0 - expected_w)
            elo[winner] += delta
            elo[loser] -= delta

    return elo


def compute_cyclic_dominance_index(
    win_rate_matrix: pd.DataFrame,
    threshold: float = 0.55,
) -> Tuple[float, list[Tuple[str, str, str]]]:
    """
    Compute the cyclic dominance index (CDI) over all agent triples.

    For a triple (A, B, C), dominance is transitive if:
      A > B and B > C implies A > C.
    A cycle exists when: A > B, B > C, C > A (all above threshold).

    Args:
        win_rate_matrix: DataFrame from compute_win_rate_matrix.
        threshold: Win rate threshold to declare dominance (default 0.55).

    Returns:
        (cdi, cycles) where cdi is in [0, 1] and cycles is a list of
        (A, B, C) triples where A > B > C > A forms a cycle.
    """
    agents = list(win_rate_matrix.index)
    n = len(agents)
    mat = win_rate_matrix.values

    def dominates(i: int, j: int) -> bool:
        v = mat[i, j]
        return not np.isnan(v) and v >= threshold

    n_triples = 0
    n_cycles = 0
    cycles: list[Tuple[str, str, str]] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                n_triples += 1
                # Check cycle: i > j > k > i
                if dominates(i, j) and dominates(j, k) and dominates(k, i):
                    n_cycles += 1
                    cycles.append((agents[i], agents[j], agents[k]))

    cdi = n_cycles / n_triples if n_triples > 0 else 0.0
    return cdi, cycles

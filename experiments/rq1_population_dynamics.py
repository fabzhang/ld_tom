"""
rq1_population_dynamics.py — RQ1 experiments: Is higher ToM always better,
or do cyclic dominance patterns emerge?

E1a: 6×6 round-robin tournament (1000 games/pair)
E1b: Population composition sweep (focal agent vs varying opponent mixes)
E1c: Cyclic dominance index + cycle enumeration

Usage:
    python experiments/rq1_population_dynamics.py --experiment e1a
    python experiments/rq1_population_dynamics.py --experiment e1b --focal bayesian
    python experiments/rq1_population_dynamics.py --experiment e1c
    python experiments/rq1_population_dynamics.py --experiment all
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import itertools
from typing import Optional

# Allow running from project root or experiments/ dir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from env.liars_dice_env import LiarsDiceEnv
from analysis.elo import compute_win_rate_matrix, compute_elo, compute_cyclic_dominance_index


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_env_state(env, agent: str) -> dict:
    return {
        "own_dice": env._dice[agent],
        "current_bid": env._current_bid,
        "n_opp_dice": sum(
            env._die_counts[a] for a in env.agents if a != agent
        ),
        "die_counts": dict(env._die_counts),
        "agent_name": agent,
        "opp_action_history": [e for e in env._history if e["agent"] != agent],
    }


def load_agents(agent_configs: dict) -> dict:
    """
    Load agents by name. agent_configs maps name → config dict.

    Supported types: random, heuristic, bayesian, ppo

    Example:
      {"random": {"type": "random"},
       "heuristic": {"type": "heuristic", "t_challenge": 0.25, "p_bluff": 0.15},
       "bayesian": {"type": "bayesian"},
       "ppo": {"type": "ppo", "path": "checkpoints/ppo_stage2/final_model.zip"}}
    """
    agents = {}
    for name, cfg in agent_configs.items():
        atype = cfg["type"]
        if atype == "random":
            from agents.random_agent import RandomAgent
            agents[name] = RandomAgent()
        elif atype == "heuristic":
            from agents.heuristic_agent import HeuristicAgent
            agents[name] = HeuristicAgent(
                t_challenge=cfg.get("t_challenge", 0.25),
                p_bluff=cfg.get("p_bluff", 0.15),
            )
        elif atype == "bayesian":
            from agents.bayesian_agent import BayesianAgent
            agents[name] = BayesianAgent(
                t_challenge=cfg.get("t_challenge", 0.25),
            )
        elif atype == "ppo":
            from agents.ppo_agent import PPOAgent
            agents[name] = PPOAgent(
                model_path=cfg.get("path"),
                name=name,
                deterministic=cfg.get("deterministic", False),
            )
        elif atype == "he2016":
            from agents.tom.he2016_agent import He2016Agent
            agent = He2016Agent.load(
                directory=cfg["inference_dir"],
                expert_dirs=cfg["expert_dirs"],
                deterministic=cfg.get("deterministic", False),
            )
            agent.name = name
            agents[name] = agent
        else:
            raise ValueError(f"Unknown agent type: {atype!r}")
    return agents


# ---------------------------------------------------------------------------
# Core game runner
# ---------------------------------------------------------------------------

def run_match(agent_a, agent_b, env_config: dict,
              n_games: int = 100, seed: Optional[int] = None) -> tuple[int, int]:
    """
    Run n_games between agent_a (player_0) and agent_b (player_1).
    Returns (a_wins, b_wins).
    """
    a_wins = 0
    b_wins = 0
    rng = np.random.default_rng(seed)

    agent_map = {"player_0": agent_a, "player_1": agent_b}
    # Reset per game to clear Bayesian posteriors etc.
    for _ in range(n_games):
        for ag in agent_map.values():
            if hasattr(ag, "reset"):
                ag.reset()

        env = LiarsDiceEnv(**env_config)
        game_seed = int(rng.integers(0, 2**31))
        env.reset(seed=game_seed)

        while env.agents:
            current = env.agent_selection
            if env.terminations.get(current, False) or env.truncations.get(current, False):
                env.step(None)
                continue
            obs = env.observe(current).astype(np.float32)
            mask = env.action_mask(current).astype(bool)
            env_state = _build_env_state(env, current)
            action = agent_map[current].act(obs, mask, env_state)
            env.step(action)

        # Determine winner from cumulative rewards
        r0 = env._cumulative_rewards.get("player_0", 0.0)
        r1 = env._cumulative_rewards.get("player_1", 0.0)
        if r0 > r1:
            a_wins += 1
        elif r1 > r0:
            b_wins += 1
        # Ties are rare but possible on truncation — ignore

    return a_wins, b_wins


# ---------------------------------------------------------------------------
# E1a: Round-robin tournament
# ---------------------------------------------------------------------------

def run_tournament(agents: dict, n_games_per_pair: int = 1000,
                   env_config: Optional[dict] = None,
                   results_path: Optional[str] = None) -> dict:
    """
    Run a full round-robin tournament.

    Returns results dict: {(a_name, b_name): (a_wins, b_wins)}
    """
    if env_config is None:
        env_config = {"n_players": 2, "n_dice": 5}

    names = list(agents.keys())
    results = {}

    total_pairs = len(names) * (len(names) - 1)
    done = 0

    for i, a_name in enumerate(names):
        for j, b_name in enumerate(names):
            if i == j:
                continue
            print(f"[E1a] {a_name} vs {b_name} ({n_games_per_pair} games)...",
                  end=" ", flush=True)
            a_wins, b_wins = run_match(
                agents[a_name], agents[b_name],
                env_config=env_config,
                n_games=n_games_per_pair,
                seed=i * 1000 + j,
            )
            results[(a_name, b_name)] = (a_wins, b_wins)
            done += 1
            wr = a_wins / (a_wins + b_wins) if (a_wins + b_wins) > 0 else 0.5
            print(f"{a_wins}/{a_wins+b_wins} ({wr:.1%})")

    if results_path is not None:
        # Serialize with string keys for JSON
        serializable = {f"{a}|{b}": list(v) for (a, b), v in results.items()}
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[E1a] Saved results to {results_path}")

    return results


def print_tournament_summary(results: dict, agents: dict):
    names = list(agents.keys())
    matrix = compute_win_rate_matrix(results, names)
    elo = compute_elo(results)
    cdi, cycles = compute_cyclic_dominance_index(matrix)

    print("\n=== Win Rate Matrix ===")
    header = f"{'':>12}" + "".join(f"{n:>12}" for n in names)
    print(header)
    for i, a in enumerate(names):
        row = f"{a:>12}"
        for j, b in enumerate(names):
            if i == j:
                row += f"{'—':>12}"
            else:
                row += f"{matrix.values[i,j]:>11.1%} "
        print(row)

    print("\n=== Elo Ratings ===")
    sorted_elo = sorted(elo.items(), key=lambda x: -x[1])
    for name, rating in sorted_elo:
        print(f"  {name:>12}: {rating:.1f}")

    print(f"\n=== Cyclic Dominance Index: {cdi:.3f} ===")
    if cycles:
        print(f"  Cycles detected ({len(cycles)}):")
        for c in cycles[:10]:
            print(f"    {' > '.join(c)} > {c[0]}")
    else:
        print("  No cycles detected (threshold=0.55)")

    return matrix, elo, cdi, cycles


# ---------------------------------------------------------------------------
# E1b: Population composition sweep
# ---------------------------------------------------------------------------

def run_population_sweep(
    focal_agent_name: str,
    focal_agent,
    population_agents: dict,
    n_mixes: int = 10,
    n_games_per_mix: int = 500,
    env_config: Optional[dict] = None,
    results_path: Optional[str] = None,
) -> list[dict]:
    """
    Fix the focal agent and vary the fraction of each opponent type.

    For each mix, opponent is drawn from population_agents according to
    a randomly sampled Dirichlet distribution over agent types.

    Returns list of {mix: {name: fraction}, focal_win_rate: float}
    """
    if env_config is None:
        env_config = {"n_players": 2, "n_dice": 5}

    opp_names = list(population_agents.keys())
    K = len(opp_names)
    rng = np.random.default_rng(42)

    # Sample K-dimensional Dirichlet mixes
    # Also include pure-type mixes (one opponent at 100%)
    mixes = []
    for k in range(K):
        pure = np.zeros(K)
        pure[k] = 1.0
        mixes.append(pure)
    # Add uniform mix
    mixes.append(np.ones(K) / K)
    # Add random mixes
    for _ in range(max(0, n_mixes - K - 1)):
        mixes.append(rng.dirichlet(np.ones(K)))

    sweep_results = []

    for mix_idx, mix in enumerate(mixes):
        mix_desc = {opp_names[k]: float(mix[k]) for k in range(K)}
        print(f"[E1b] Mix {mix_idx+1}/{len(mixes)}: {mix_desc}")

        focal_wins = 0
        total = 0

        for game_idx in range(n_games_per_mix):
            # Sample opponent for this game
            opp_name = rng.choice(opp_names, p=mix)
            opp_agent = population_agents[str(opp_name)]

            if hasattr(focal_agent, "reset"):
                focal_agent.reset()
            if hasattr(opp_agent, "reset"):
                opp_agent.reset()

            env = LiarsDiceEnv(**env_config)
            env.reset(seed=int(rng.integers(0, 2**31)))

            agent_map = {
                "player_0": focal_agent,
                "player_1": opp_agent,
            }

            while env.agents:
                current = env.agent_selection
                if env.terminations.get(current, False) or env.truncations.get(current, False):
                    env.step(None)
                    continue
                obs = env.observe(current).astype(np.float32)
                mask = env.action_mask(current).astype(bool)
                env_state = _build_env_state(env, current)
                action = agent_map[current].act(obs, mask, env_state)
                env.step(action)

            r0 = env._cumulative_rewards.get("player_0", 0.0)
            r1 = env._cumulative_rewards.get("player_1", 0.0)
            if r0 > r1:
                focal_wins += 1
            total += 1

        focal_wr = focal_wins / total if total > 0 else 0.5
        result = {"mix": mix_desc, "focal_win_rate": focal_wr,
                  "focal_wins": focal_wins, "total_games": total}
        sweep_results.append(result)
        print(f"         focal win rate: {focal_wr:.1%}")

    if results_path is not None:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({"focal": focal_agent_name, "results": sweep_results}, f, indent=2)
        print(f"[E1b] Saved sweep results to {results_path}")

    return sweep_results


# ---------------------------------------------------------------------------
# E1c: Cyclic dominance analysis (thin wrapper — real logic is in elo.py)
# ---------------------------------------------------------------------------

def run_cyclic_dominance_analysis(results: dict, agent_names: list,
                                   threshold: float = 0.55) -> tuple[float, list]:
    matrix = compute_win_rate_matrix(results, agent_names)
    cdi, cycles = compute_cyclic_dominance_index(matrix, threshold)
    print(f"\n=== E1c: Cyclic Dominance Analysis (threshold={threshold}) ===")
    print(f"  CDI = {cdi:.4f}")
    if cycles:
        print(f"  Cycles ({len(cycles)}):")
        for c in cycles:
            print(f"    {' > '.join(c)} > {c[0]}")
    else:
        print("  No cycles detected")
    return cdi, cycles


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

DEFAULT_AGENTS_3 = {
    "random":    {"type": "random"},
    "heuristic": {"type": "heuristic", "t_challenge": 0.25, "p_bluff": 0.15},
    "bayesian":  {"type": "bayesian", "t_challenge": 0.20},
}

DEFAULT_AGENTS_6 = {
    "random":    {"type": "random"},
    "heuristic": {"type": "heuristic", "t_challenge": 0.25, "p_bluff": 0.15},
    "bayesian":  {"type": "bayesian", "t_challenge": 0.20},
    # Add trained agents here as they become available:
    # "ppo":    {"type": "ppo", "path": "checkpoints/ppo_stage2/final_model.zip"},
    # "he2016": {"type": "he2016",
    #             "inference_dir": "checkpoints/he2016",
    #             "expert_dirs": {
    #                 "random":    "checkpoints/he2016_experts/expert_random.zip",
    #                 "heuristic": "checkpoints/he2016_experts/expert_heuristic.zip",
    #                 "bayesian":  "checkpoints/he2016_experts/expert_bayesian.zip",
    #                 "ppo":       "checkpoints/he2016_experts/expert_ppo.zip",
    #             }},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="e1a",
                        choices=["e1a", "e1b", "e1c", "all"])
    parser.add_argument("--n_games", type=int, default=1000,
                        help="Games per pair for E1a, or per mix for E1b")
    parser.add_argument("--focal", default="bayesian",
                        help="Focal agent name for E1b sweep")
    parser.add_argument("--env_n_dice", type=int, default=5)
    parser.add_argument("--results_dir", default="results/rq1")
    parser.add_argument("--ppo_path", default=None,
                        help="Path to trained PPO model zip (optional)")
    parser.add_argument("--he2016_dir", default=None,
                        help="Dir with inference_net.pt + config.json")
    parser.add_argument("--experts_dir", default="checkpoints/he2016_experts",
                        help="Dir with expert_*.zip files")
    args = parser.parse_args()

    env_config = {"n_players": 2, "n_dice": args.env_n_dice}
    os.makedirs(args.results_dir, exist_ok=True)

    agent_configs = dict(DEFAULT_AGENTS_6)
    if args.ppo_path and os.path.exists(args.ppo_path):
        agent_configs["ppo"] = {"type": "ppo", "path": args.ppo_path}

    if args.he2016_dir and os.path.exists(os.path.join(args.he2016_dir, "config.json")):
        agent_configs["he2016"] = {
            "type": "he2016",
            "inference_dir": args.he2016_dir,
            "expert_dirs": {
                "random":    os.path.join(args.experts_dir, "expert_random.zip"),
                "heuristic": os.path.join(args.experts_dir, "expert_heuristic.zip"),
                "bayesian":  os.path.join(args.experts_dir, "expert_bayesian.zip"),
                "ppo":       os.path.join(args.experts_dir, "expert_ppo.zip"),
            },
        }

    agents = load_agents(agent_configs)
    print(f"Loaded agents: {list(agents.keys())}")

    tournament_results = None

    if args.experiment in ("e1a", "all"):
        print("\n" + "="*60)
        print("E1a: Round-Robin Tournament")
        print("="*60)
        rpath = os.path.join(args.results_dir, "e1a_results.json")
        tournament_results = run_tournament(
            agents,
            n_games_per_pair=args.n_games,
            env_config=env_config,
            results_path=rpath,
        )
        print_tournament_summary(tournament_results, agents)

    if args.experiment in ("e1b", "all"):
        print("\n" + "="*60)
        print(f"E1b: Population Sweep (focal={args.focal})")
        print("="*60)
        if args.focal not in agents:
            print(f"Focal agent {args.focal!r} not in loaded agents: {list(agents.keys())}")
        else:
            pop_agents = {k: v for k, v in agents.items() if k != args.focal}
            rpath = os.path.join(args.results_dir, f"e1b_{args.focal}_sweep.json")
            run_population_sweep(
                focal_agent_name=args.focal,
                focal_agent=agents[args.focal],
                population_agents=pop_agents,
                n_games_per_mix=args.n_games,
                env_config=env_config,
                results_path=rpath,
            )

    if args.experiment in ("e1c", "all"):
        print("\n" + "="*60)
        print("E1c: Cyclic Dominance Index")
        print("="*60)
        if tournament_results is None:
            rpath = os.path.join(args.results_dir, "e1a_results.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    raw = json.load(f)
                tournament_results = {
                    tuple(k.split("|")): tuple(v) for k, v in raw.items()
                }
            else:
                print("No tournament results found; run --experiment e1a first")
                sys.exit(1)
        run_cyclic_dominance_analysis(
            tournament_results,
            list(agents.keys()),
            threshold=0.55,
        )

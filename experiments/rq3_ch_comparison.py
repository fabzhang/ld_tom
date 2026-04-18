"""
experiments/rq3_ch_comparison.py

RQ3: Do emergent strategies from self-play recapitulate human-like level-k thinking,
or do agents discover qualitatively different equilibria?

Experiments:
  E3a: Challenge calibration curve (challenge rate vs p_valid)
  E3b: Implied tau fitting (Poisson CH model)
  E3c: Strategy fingerprint + PCA
  E3d: Non-CH state detection

Usage:
  python experiments/rq3_ch_comparison.py \
      --n_games 5000 \
      --ppo_path checkpoints/ppo/stage2/final.zip \
      --he2016_dir checkpoints/he2016/ \
      --experts_dir checkpoints/experts/ \
      --output_dir results/rq3/ \
      --env_n_dice 5

All experiments run sequentially. Trajectories are collected once and reused.
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.liars_dice_env import LiarsDiceEnv
from env.bid_encoder import BidEncoder
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.bayesian_agent import BayesianAgent
from agents.ppo_agent import PPOAgent
from agents.tom.he2016_agent import He2016Agent
from analysis.cognitive_hierarchy import (
    p_bid_valid,
    fit_tau_fast,
    bootstrap_tau_ci_fast,
    compute_kl_table,
    compute_challenge_calibration,
    state_kl_divergences,
)
from analysis.strategy_fingerprint import extract_fingerprint, pca_agents, save_fingerprints


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories(agent, agent_name, n_games, env_config):
    """
    Run agent for n_games and record every decision point.

    Opponent is always a fresh RandomAgent (neutral, known Level-0).
    This ensures p_valid has a clean interpretation.

    Returns:
        list of dicts:
        {
            "own_dice": list,
            "current_bid": [q, f] or None,
            "n_opp_dice": int,
            "player_position": int,   # 0 or 1
            "action": int,
            "action_mask": list,
            "p_valid": float,
            "bid_history": list,      # list of bid events before this decision
            "game_id": int,
        }
    """
    from pettingzoo.test import api_test  # noqa: import check

    trajectories = []
    n_dice = env_config.get("n_dice", 5)
    n_players = 2

    print(f"Collecting {n_games} games for {agent_name}...")

    for game_id in range(n_games):
        if game_id % 500 == 0:
            print(f"  Game {game_id}/{n_games}")

        env = LiarsDiceEnv(n_players=n_players, n_dice=n_dice)
        env.reset()
        opponent = RandomAgent()

        # Determine which player index our agent is (alternate each game)
        focal_player_idx = game_id % 2
        agent_player = env.agents[focal_player_idx]
        opp_player = env.agents[1 - focal_player_idx]

        if hasattr(agent, "reset"):
            agent.reset()
        if hasattr(opponent, "reset"):
            opponent.reset()

        bid_history_this_game = []

        for current_agent in env.agent_iter():
            obs = env.observe(current_agent)
            action_mask = env.action_mask(current_agent)

            if env.terminations.get(current_agent, False) or env.truncations.get(current_agent, False):
                env.step(None)
                continue

            env_state = {
                "own_dice": env._dice[current_agent].copy(),
                "current_bid": env._current_bid,
                "n_opp_dice": sum(
                    env._die_counts[a] for a in env.agents if a != current_agent
                ),
                "die_counts": dict(env._die_counts),
                "agent_name": current_agent,
                "opp_action_history": [
                    e for e in env._history if e.get("agent") != current_agent
                ],
            }

            if current_agent == agent_player:
                action = agent.act(obs, action_mask, env_state=env_state)

                # Record trajectory point
                own_dice = env_state["own_dice"]
                current_bid = env_state["current_bid"]
                n_opp_dice = env_state["n_opp_dice"]

                pv = (
                    p_bid_valid(own_dice, current_bid, n_opp_dice)
                    if current_bid is not None
                    else None
                )

                traj = {
                    "own_dice": own_dice.tolist(),
                    "current_bid": list(current_bid) if current_bid is not None else None,
                    "n_opp_dice": n_opp_dice,
                    "player_position": focal_player_idx,
                    "action": int(action),
                    "action_mask": action_mask.tolist(),
                    "p_valid": float(pv) if pv is not None else None,
                    "bid_history": [
                        {k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                         for k, v in e.items()}
                        for e in bid_history_this_game
                    ],
                    "game_id": game_id,
                }
                trajectories.append(traj)

            else:
                action = opponent.act(obs, action_mask)

            # Update local bid history after the action
            bid_event = {
                "agent": current_agent,
                "type": "challenge" if action == 0 else "bid",
            }
            if action > 0 and env._current_bid is not None:
                # We'll capture the new bid after the step
                pass
            env.step(action)

            # After step, record what was done
            if action == 0:
                bid_history_this_game.append({"type": "challenge", "agent": current_agent})
            else:
                # Decode bid from action index
                bid_action = action - 1
                face = (bid_action % 6) + 1
                qty = bid_action // 6 + 1
                bid_history_this_game.append({
                    "type": "bid", "agent": current_agent,
                    "quantity": qty, "face": face,
                })

        env.close()

    print(f"  Collected {len(trajectories)} decision points from {n_games} games")
    return trajectories


def save_trajectories(trajectories, path):
    with open(path, "w") as f:
        json.dump(trajectories, f)
    print(f"Trajectories saved: {path} ({len(trajectories)} points)")


def load_trajectories(path):
    with open(path) as f:
        data = json.load(f)
    # Convert lists back to np arrays for own_dice and action_mask
    for t in data:
        t["own_dice"] = np.array(t["own_dice"])
        t["action_mask"] = np.array(t["action_mask"])
        if t.get("current_bid"):
            t["current_bid"] = tuple(t["current_bid"])
    return data


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def load_agents(ppo_path, he2016_dir, experts_dir):
    agents = {
        "random": RandomAgent(),
        "heuristic": HeuristicAgent(),
        "bayesian": BayesianAgent(),
    }

    if ppo_path and os.path.exists(ppo_path):
        ppo = PPOAgent(deterministic=False)
        ppo.load(ppo_path)
        agents["ppo"] = ppo
        print(f"PPO loaded from {ppo_path}")
    else:
        print(f"PPO path not found: {ppo_path}")

    if he2016_dir and os.path.exists(he2016_dir):
        expert_dirs = {
            "random":    os.path.join(experts_dir, "expert_random.zip"),
            "heuristic": os.path.join(experts_dir, "expert_heuristic.zip"),
            "bayesian":  os.path.join(experts_dir, "expert_bayesian.zip"),
            "ppo":       os.path.join(experts_dir, "expert_ppo.zip"),
        }
        he = He2016Agent.load(
            directory=he2016_dir,
            expert_dirs=expert_dirs,
            deterministic=False,
        )
        agents["he2016"] = he
        print(f"He2016 loaded from {he2016_dir}")
    else:
        print(f"He2016 dir not found: {he2016_dir}")

    return agents


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_rq3(args):
    os.makedirs(args.output_dir, exist_ok=True)

    env_config = {"n_dice": args.env_n_dice}

    # Load agents
    agents = load_agents(args.ppo_path, args.he2016_dir, args.experts_dir)
    print(f"Agents loaded: {list(agents.keys())}")

    # --- Collect or load trajectories ---
    trajectories_per_agent = {}
    for name, agent in agents.items():
        traj_path = os.path.join(args.output_dir, f"trajectories_{name}.json")
        if os.path.exists(traj_path) and not args.force_recollect:
            print(f"Loading cached trajectories for {name}...")
            trajectories_per_agent[name] = load_trajectories(traj_path)
        else:
            trajs = collect_trajectories(agent, name, args.n_games, env_config)
            save_trajectories(trajs, traj_path)
            trajectories_per_agent[name] = trajs

    # --- E3a: Challenge calibration ---
    print("\n=== E3a: Challenge Calibration ===")
    calibration_results = {}
    for name, trajs in trajectories_per_agent.items():
        cal = compute_challenge_calibration(trajs, n_bins=20)
        calibration_results[name] = {
            "bin_centers": cal["bin_centers"].tolist(),
            "challenge_rates": [
                float(x) if not np.isnan(x) else None
                for x in cal["challenge_rates"]
            ],
            "bin_counts": cal["bin_counts"].tolist(),
        }
        # Also compute positional breakdown
        trajs_p0 = [t for t in trajs if t.get("player_position") == 0]
        trajs_p1 = [t for t in trajs if t.get("player_position") == 1]
        cal_p0 = compute_challenge_calibration(trajs_p0, n_bins=10)
        cal_p1 = compute_challenge_calibration(trajs_p1, n_bins=10)
        calibration_results[name]["by_position"] = {
            "p0_challenge_rates": [
                float(x) if not np.isnan(x) else None
                for x in cal_p0["challenge_rates"]
            ],
            "p1_challenge_rates": [
                float(x) if not np.isnan(x) else None
                for x in cal_p1["challenge_rates"]
            ],
            "p0_bin_centers": cal_p0["bin_centers"].tolist(),
        }
        print(f"  {name}: calibration computed ({len(trajs)} points)")

    cal_path = os.path.join(args.output_dir, "e3a_calibration.json")
    with open(cal_path, "w") as f:
        json.dump(calibration_results, f, indent=2)
    print(f"E3a saved: {cal_path}")

    # --- E3b: Tau fitting ---
    print("\n=== E3b: Implied Tau Fitting ===")
    tau_results = {}
    for name, trajs in trajectories_per_agent.items():
        print(f"  Fitting tau for {name}...")
        fit = fit_tau_fast(trajs)
        mean_tau, lo, hi = bootstrap_tau_ci_fast(trajs, n_bootstrap=200)
        tau_results[name] = {
            "tau_mle": fit["tau"],
            "tau_bootstrap_mean": mean_tau,
            "tau_ci_lower": lo,
            "tau_ci_upper": hi,
            "log_likelihoods": fit["log_likelihoods"],
            "n_observations": fit["n_observations"],
        }
        print(f"    {name}: tau={fit['tau']:.2f} (bootstrap: {mean_tau:.2f} [{lo:.2f}, {hi:.2f}])")

    tau_path = os.path.join(args.output_dir, "e3b_tau_fitting.json")
    with open(tau_path, "w") as f:
        json.dump(tau_results, f, indent=2)
    print(f"E3b saved: {tau_path}")

    # --- KL table (for paper) ---
    print("\n=== KL Table (agent vs Level-k) ===")
    kl_table = compute_kl_table(trajectories_per_agent, K=3)
    for agent_name, kl_by_k in kl_table.items():
        kl_str = " | ".join(f"L{k}: {v:.3f}" for k, v in kl_by_k.items())
        print(f"  {agent_name}: {kl_str}")

    kl_path = os.path.join(args.output_dir, "e3b_kl_table.json")
    with open(kl_path, "w") as f:
        json.dump(kl_table, f, indent=2)

    # --- E3c: Strategy fingerprints ---
    print("\n=== E3c: Strategy Fingerprints ===")
    fingerprints = {}
    for name, trajs in trajectories_per_agent.items():
        fingerprints[name] = extract_fingerprint(trajs, name)
        print(f"  {name}: positional_bias={fingerprints[name]['positional_bias']:.3f}, "
              f"bluff_rate={fingerprints[name]['bluff_rate']:.3f}, "
              f"overall_cr={fingerprints[name]['overall_challenge_rate']:.3f}")

    pca_result = pca_agents(fingerprints)
    fp_path = os.path.join(args.output_dir, "e3c_fingerprints.json")
    save_fingerprints(fingerprints, pca_result, fp_path)
    print(f"E3c saved: {fp_path}")

    # --- E3d: Non-CH state detection ---
    print("\n=== E3d: Non-CH State Detection ===")
    non_ch_results = {}
    for name, trajs in trajectories_per_agent.items():
        # Compute min KL to any level for each trajectory point
        kl_per_point = []
        for k in range(3):
            kls = state_kl_divergences(trajs, k)
            kl_per_point.append(kls)

        kl_per_point = np.stack(kl_per_point, axis=1)  # (N, 3)
        min_kl = kl_per_point.min(axis=1)  # min over levels

        # Non-CH states: min_kl > 0.5 nats
        threshold = 0.5
        non_ch_mask = min_kl > threshold
        non_ch_frac = float(non_ch_mask.mean())

        # Top-10 non-CH states
        top_idx = np.argsort(min_kl)[-10:][::-1]
        top_states = []
        for idx in top_idx:
            t = trajs[idx]
            top_states.append({
                "p_valid": float(t["p_valid"]) if t.get("p_valid") is not None else None,
                "action": int(t["action"]),
                "is_challenge": bool(t["action"] == 0),
                "current_bid": t.get("current_bid"),
                "n_opp_dice": int(t["n_opp_dice"]),
                "player_position": int(t.get("player_position", -1)),
                "min_kl_to_ch": float(min_kl[idx]),
                "game_id": int(t.get("game_id", -1)),
            })

        non_ch_results[name] = {
            "non_ch_fraction": non_ch_frac,
            "mean_min_kl": float(min_kl.mean()),
            "top_non_ch_states": top_states,
        }
        print(f"  {name}: non-CH fraction={non_ch_frac:.3f}, "
              f"mean min KL={min_kl.mean():.3f}")

    nc_path = os.path.join(args.output_dir, "e3d_non_ch_states.json")
    with open(nc_path, "w") as f:
        json.dump(non_ch_results, f, indent=2)
    print(f"E3d saved: {nc_path}")

    print(f"\nAll RQ3 results saved to {args.output_dir}")
    return {
        "calibration": calibration_results,
        "tau": tau_results,
        "kl_table": kl_table,
        "fingerprints": {n: {k: v for k, v in fp.items() if k != "fingerprint"}
                         for n, fp in fingerprints.items()},
        "non_ch": non_ch_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=5000,
                        help="Games per agent for trajectory collection")
    parser.add_argument("--ppo_path", type=str, default="checkpoints/ppo/stage2/final.zip")
    parser.add_argument("--he2016_dir", type=str, default="checkpoints/he2016/")
    parser.add_argument("--experts_dir", type=str, default="checkpoints/experts/")
    parser.add_argument("--output_dir", type=str, default="results/rq3/")
    parser.add_argument("--env_n_dice", type=int, default=5)
    parser.add_argument("--force_recollect", action="store_true",
                        help="Re-collect trajectories even if cached")
    args = parser.parse_args()

    run_rq3(args)

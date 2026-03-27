"""
tournament.py — Quick sanity check: 100-game tournament between two random agents.
Run with: python tournament.py (from project root)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env.liars_dice_env import LiarsDiceEnv
from agents.random_agent import RandomAgent


def run_tournament(n_games: int = 100, n_players: int = 2, n_dice: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    env = LiarsDiceEnv(n_players=n_players, n_dice=n_dice, max_rounds=300)
    agents = {name: RandomAgent(name) for name in env.possible_agents}
    wins = {name: 0 for name in env.possible_agents}
    game_lengths = []

    for game_idx in range(n_games):
        game_seed = int(rng.integers(0, 2**31))
        env.reset(seed=game_seed)
        for ag in agents.values():
            ag.reset()

        steps = 0
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                steps += 1
                continue
            mask = env.action_mask(agent)
            action = agents[agent].act(obs, mask)
            env.step(action)
            steps += 1

        game_lengths.append(steps)
        # Award win to whoever accumulated positive reward
        for name in env.possible_agents:
            if env._cumulative_rewards.get(name, 0) > 0:
                wins[name] += 1

    print(f"\n{'='*50}")
    print(f"Tournament: {n_games} games, {n_players} players, {n_dice} dice")
    print(f"{'='*50}")
    for name, w in wins.items():
        print(f"  {name}: {w}/{n_games} wins ({100*w/n_games:.1f}%)")
    print(f"  Avg game length: {np.mean(game_lengths):.1f} steps")
    print(f"  Min/Max: {min(game_lengths)}/{max(game_lengths)} steps")
    return wins


if __name__ == "__main__":
    run_tournament(n_games=100)

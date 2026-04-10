"""
training/expert_training.py — Train one PPO expert per opponent type for He2016.

For each opponent type (random, heuristic, bayesian, ppo), trains a specialist
PPO agent that plays exclusively against that opponent type.

Usage:
    # Train all 4 experts sequentially:
    python training/expert_training.py --all --env_n_dice 5

    # Train a single expert:
    python training/expert_training.py --opponent_type bayesian --env_n_dice 5

    # Use a trained PPO as one of the opponents:
    python training/expert_training.py --all --ppo_path checkpoints/ppo_stage2/final_model.zip
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def load_opponent(opponent_type: str, ppo_path: str | None = None):
    """Load a fixed opponent agent by type name."""
    if opponent_type == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent()
    elif opponent_type == "heuristic":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(t_challenge=0.25, p_bluff=0.15)
    elif opponent_type == "bayesian":
        from agents.bayesian_agent import BayesianAgent
        return BayesianAgent(t_challenge=0.20)
    elif opponent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        if ppo_path and os.path.exists(ppo_path):
            return PPOAgent(model_path=ppo_path, deterministic=False)
        else:
            from agents.random_agent import RandomAgent
            print(f"[expert_training] PPO path not found, using random as 'ppo' opponent")
            return RandomAgent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type!r}")


def _build_env_state(env, agent: str) -> dict:
    opponent = [a for a in env.agents if a != agent]
    opp_history = [e for e in env._history if e["agent"] != agent]
    return {
        "own_dice": env._dice[agent],
        "current_bid": env._current_bid,
        "n_opp_dice": sum(env._die_counts[a] for a in env.agents if a != agent),
        "die_counts": dict(env._die_counts),
        "agent_name": agent,
        "opp_action_history": opp_history,
    }


class FixedOpponentEnv(gym.Env):
    """
    Single-agent Gym env where the opponent is always the same fixed agent.
    Focal agent is player_0.
    """
    metadata = {}

    def __init__(self, env_config: dict, opponent):
        super().__init__()
        self._env_config = env_config
        self._opponent = opponent
        self._focal = "player_0"

        from env.liars_dice_env import LiarsDiceEnv
        self._env = LiarsDiceEnv(**env_config)

        obs_size = self._env.observation_spaces[self._focal].shape[0]
        n_actions = self._env.action_spaces[self._focal].n

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

    def reset(self, seed=None, options=None):
        if hasattr(self._opponent, "reset"):
            self._opponent.reset()
        self._env.reset(seed=seed)
        obs, mask = self._advance_to_focal()
        return obs, {}

    def step(self, action):
        self._env.step(action)
        obs, mask = self._advance_to_focal()
        terminated = len(self._env.agents) == 0
        reward = self._env._cumulative_rewards.get(self._focal, 0.0)
        return obs, reward, terminated, False, {"action_mask": mask}

    def _advance_to_focal(self):
        while True:
            if not self._env.agents:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                mask = np.zeros(self.action_space.n, dtype=bool)
                mask[0] = True
                return obs, mask

            current = self._env.agent_selection
            if current == self._focal:
                return (
                    self._env.observe(current).astype(np.float32),
                    self._env.action_mask(current).astype(bool),
                )

            if (self._env.terminations.get(current, False) or
                    self._env.truncations.get(current, False)):
                self._env.step(None)
                continue

            obs = self._env.observe(current).astype(np.float32)
            mask = self._env.action_mask(current).astype(bool)
            env_state = _build_env_state(self._env, current)
            action = self._opponent.act(obs, mask, env_state)
            self._env.step(action)

    def action_masks(self) -> np.ndarray:
        if not self._env.agents or self._focal not in self._env.agents:
            m = np.zeros(self.action_space.n, dtype=bool)
            m[0] = True
            return m
        return self._env.action_mask(self._focal).astype(bool)


def train_expert(
    opponent_type: str,
    env_config: dict,
    total_timesteps: int,
    output_dir: str,
    ppo_path: str | None = None,
    device: str = "cpu",
) -> str:
    """
    Train a PPO expert specializing against one fixed opponent type.

    Returns the path to the saved model zip.
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback

    os.makedirs(output_dir, exist_ok=True)
    opponent = load_opponent(opponent_type, ppo_path=ppo_path)

    env = FixedOpponentEnv(env_config=env_config, opponent=opponent)
    masked_env = ActionMasker(env, lambda e: e.action_masks())

    model = MaskablePPO(
        policy="MlpPolicy",
        env=masked_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        device=device,
        policy_kwargs={"net_arch": [256, 256]},
    )

    print(f"[expert_training] Training expert vs {opponent_type!r} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = os.path.join(output_dir, f"expert_{opponent_type}")
    model.save(save_path)
    print(f"[expert_training] Saved to {save_path}.zip")
    return save_path + ".zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent_type", default=None,
                        choices=["random", "heuristic", "bayesian", "ppo"])
    parser.add_argument("--all", action="store_true",
                        help="Train all 4 experts sequentially")
    parser.add_argument("--env_n_dice", type=int, default=5)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--output_dir", default="checkpoints/he2016_experts")
    parser.add_argument("--ppo_path", default="checkpoints/ppo_stage2/final_model.zip")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env_config = {"n_players": 2, "n_dice": args.env_n_dice}

    if args.all:
        types = ["random", "heuristic", "bayesian", "ppo"]
    elif args.opponent_type:
        types = [args.opponent_type]
    else:
        parser.error("Specify --opponent_type or --all")

    for opp_type in types:
        train_expert(
            opponent_type=opp_type,
            env_config=env_config,
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            ppo_path=args.ppo_path,
            device=args.device,
        )
    print("All expert training complete.")

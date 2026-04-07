"""
self_play.py — Snapshot pool self-play training for the PPO agent.

Protocol:
  Stage 1: 2p/3-dice, 2M steps  (fast convergence, debug)
  Stage 2: 2p/5-dice, 3M steps  (standard config)

The opponent at each episode is drawn uniformly from the snapshot pool,
which starts with a random agent and grows as new snapshots are saved.
"""
from __future__ import annotations

import os
import random
import shutil
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from collections import deque


class SnapshotPool:
    """Maintains a pool of saved model paths for self-play opponents."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._paths: deque[str] = deque()

    def add(self, model_path: str):
        self._paths.append(model_path)
        while len(self._paths) > self.max_size:
            self._paths.popleft()

    def sample(self) -> Optional[str]:
        """Returns None (→ random agent) if pool empty, else a random path."""
        if not self._paths:
            return None
        return random.choice(list(self._paths))

    def __len__(self):
        return len(self._paths)


class SelfPlayEnv(gym.Env):
    """
    Single-agent Gym wrapper around LiarsDiceEnv for SB3 training.

    The opponent is sampled from a SnapshotPool at the start of each episode.
    The focal agent always plays as player_0; the opponent plays all other seats.
    """

    metadata = {}

    def __init__(self, env_config: dict, pool: SnapshotPool,
                 focal_agent_name: str = "player_0"):
        super().__init__()
        self._env_config = env_config
        self._pool = pool
        self._focal = focal_agent_name
        self._env = None
        self._opp_agent = None
        self._model_cache: dict[str, object] = {}  # path → loaded MaskablePPO model
        self._make_env()

        obs_size = self._env.observation_spaces[self._focal].shape[0]
        n_actions = self._env.action_spaces[self._focal].n

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

    def _make_env(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from env.liars_dice_env import LiarsDiceEnv
        self._env = LiarsDiceEnv(**self._env_config)

    def _load_opponent(self):
        path = self._pool.sample()
        if path is None:
            from agents.random_agent import RandomAgent
            self._opp_agent = RandomAgent()
        else:
            from agents.ppo_agent import PPOAgent
            from sb3_contrib import MaskablePPO
            # Cache loaded models to avoid re-reading from disk every episode
            if path not in self._model_cache:
                self._model_cache[path] = MaskablePPO.load(path)
            agent = PPOAgent(name="opp")
            agent.model = self._model_cache[path]
            self._opp_agent = agent

    def reset(self, seed=None, options=None):
        self._load_opponent()
        self._env.reset(seed=seed)
        # Advance to focal agent's first turn (stepping opponents through their turns)
        obs, mask = self._advance_to_focal()
        return obs, {}

    def step(self, action):
        # Apply focal agent's action
        self._env.step(action)
        # Advance opponents until it's focal's turn again (or game over)
        obs, mask = self._advance_to_focal()
        terminated = len(self._env.agents) == 0
        reward = self._env._cumulative_rewards.get(self._focal, 0.0)
        truncated = False
        info = {"action_mask": mask}
        return obs, reward, terminated, truncated, info

    def _advance_to_focal(self):
        """Step through opponent turns until it is the focal agent's turn."""
        while True:
            if len(self._env.agents) == 0:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                mask = np.zeros(self.action_space.n, dtype=bool)
                mask[0] = True
                return obs, mask

            current = self._env.agent_selection
            if current == self._focal:
                obs = self._env.observe(self._focal).astype(np.float32)
                mask = self._env.action_mask(self._focal).astype(bool)
                return obs, mask

            # Handle dead-step
            if (self._env.terminations.get(current, False) or
                    self._env.truncations.get(current, False)):
                self._env.step(None)
                continue

            # Opponent acts
            opp_obs_arr = self._env.observe(current).astype(np.float32)
            opp_mask = self._env.action_mask(current).astype(bool)
            env_state = self._build_env_state(current)
            action = self._opp_agent.act(opp_obs_arr, opp_mask, env_state)
            self._env.step(action)

    def _build_env_state(self, agent: str) -> dict:
        return {
            "own_dice": self._env._dice[agent],
            "current_bid": self._env._current_bid,
            "n_opp_dice": sum(
                self._env._die_counts[a]
                for a in self._env.agents
                if a != agent
            ),
            "die_counts": dict(self._env._die_counts),
            "agent_name": agent,
        }

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO to get valid action mask."""
        if len(self._env.agents) == 0 or self._focal not in self._env.agents:
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[0] = True
            return mask
        return self._env.action_mask(self._focal).astype(bool)


def train_ppo_selfplay(
    env_config: dict,
    total_timesteps: int,
    snapshot_freq: int,
    pool: SnapshotPool,
    output_dir: str,
    device: str = "auto",
    wandb_project: Optional[str] = None,
) -> "PPOAgent":
    """
    Train a PPO agent via snapshot pool self-play.

    Args:
        env_config: Passed directly to LiarsDiceEnv(**env_config).
        total_timesteps: Total env steps to train for.
        snapshot_freq: Save a snapshot to pool every this many steps.
        pool: SnapshotPool instance (seeded with random agent = None initially).
        output_dir: Directory to save snapshots and final model.
        device: "auto", "cpu", or "cuda".
        wandb_project: If set, log to W&B under this project.

    Returns:
        Trained PPOAgent.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    os.makedirs(output_dir, exist_ok=True)

    train_env = SelfPlayEnv(env_config=env_config, pool=pool)
    masked_env = ActionMasker(train_env, lambda env: env.action_masks())

    callbacks = []
    if wandb_project is not None:
        try:
            from wandb.integration.sb3 import WandbCallback
            import wandb
            wandb.init(project=wandb_project, config={
                "env_config": env_config,
                "total_timesteps": total_timesteps,
                "snapshot_freq": snapshot_freq,
            })
            callbacks.append(WandbCallback(verbose=0))
        except ImportError:
            print("wandb not available, skipping W&B logging")

    snapshot_callback = _SnapshotCallback(
        snapshot_freq=snapshot_freq,
        pool=pool,
        output_dir=output_dir,
    )
    callbacks.append(snapshot_callback)

    # Resume from latest checkpoint if one exists
    resume_path = os.path.join(output_dir, "latest_checkpoint.zip")
    steps_done = 0
    if os.path.exists(resume_path):
        print(f"[train_ppo_selfplay] Resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=masked_env, device=device)
        # Also seed the pool with all snapshots already saved
        import glob
        for snap in sorted(glob.glob(os.path.join(output_dir, "snapshot_*.zip"))):
            pool.add(snap)
            train_env._model_cache[snap] = MaskablePPO.load(snap)
        # Read steps_done from a sidecar file
        steps_file = os.path.join(output_dir, "steps_done.txt")
        if os.path.exists(steps_file):
            with open(steps_file) as f:
                steps_done = int(f.read().strip())
        print(f"[train_ppo_selfplay] Pool has {len(pool)} snapshots, resuming at step {steps_done}")
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=masked_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.05,
            verbose=1,
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
        )

    remaining_steps = max(0, total_timesteps - steps_done)
    print(f"[train_ppo_selfplay] Training for {remaining_steps} more steps "
          f"({steps_done}/{total_timesteps} done)")
    if remaining_steps > 0:
        model.learn(total_timesteps=remaining_steps, callback=callbacks, reset_num_timesteps=False)

    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"[train_ppo_selfplay] Saved final model to {final_path}")

    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(name="ppo")
    agent.model = model
    return agent


class _SnapshotCallback:
    """SB3 callback that saves model snapshots and adds them to the pool."""

    def __init__(self, snapshot_freq: int, pool: SnapshotPool, output_dir: str):
        self.snapshot_freq = snapshot_freq
        self.pool = pool
        self.output_dir = output_dir
        self._last_snapshot = 0
        # SB3 callbacks need these attributes
        self.n_calls = 0

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        """Called after each rollout collection step by SB3."""
        return True

    # SB3 BaseCallback interface methods
    def init_callback(self, model):
        self.model = model

    def on_step(self) -> bool:
        self.n_calls += 1
        timestep = self.model.num_timesteps
        if timestep - self._last_snapshot >= self.snapshot_freq:
            path = os.path.join(self.output_dir, f"snapshot_{timestep}")
            self.model.save(path)
            self.pool.add(path + ".zip")
            self._last_snapshot = timestep
            print(f"[snapshot] Saved at step {timestep}: {path}.zip (pool size={len(self.pool)})")
        return True

    def on_training_end(self):
        pass

    def on_rollout_start(self):
        pass

    def on_rollout_end(self):
        pass


# SB3 proper callback class
try:
    from stable_baselines3.common.callbacks import BaseCallback

    class SnapshotCallback(BaseCallback):
        def __init__(self, snapshot_freq: int, pool: SnapshotPool,
                     output_dir: str, verbose: int = 0):
            super().__init__(verbose)
            self.snapshot_freq = snapshot_freq
            self.pool = pool
            self.output_dir = output_dir
            self._last_snapshot = 0

        def _on_step(self) -> bool:
            t = self.num_timesteps
            if t - self._last_snapshot >= self.snapshot_freq:
                path = os.path.join(self.output_dir, f"snapshot_{t}")
                self.model.save(path)
                self.pool.add(path + ".zip")
                self._last_snapshot = t
                # Overwrite latest_checkpoint so resume picks up here
                self.model.save(os.path.join(self.output_dir, "latest_checkpoint"))
                with open(os.path.join(self.output_dir, "steps_done.txt"), "w") as f:
                    f.write(str(t))
                if self.verbose:
                    print(f"[snapshot] step={t} pool={len(self.pool)}")
            return True

    # Replace the simple callback with the proper SB3 one
    _SnapshotCallback = SnapshotCallback  # noqa: F811

except ImportError:
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="1=2p/3dice, 2=2p/5dice")
    parser.add_argument("--output_dir", default="checkpoints/ppo_stage1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb_project", default=None)
    args = parser.parse_args()

    if args.stage == 1:
        cfg = {"n_players": 2, "n_dice": 3}
        steps = 2_000_000
    else:
        cfg = {"n_players": 2, "n_dice": 5}
        steps = 3_000_000

    pool = SnapshotPool(max_size=20)
    agent = train_ppo_selfplay(
        env_config=cfg,
        total_timesteps=steps,
        snapshot_freq=100_000,
        pool=pool,
        output_dir=args.output_dir,
        device=args.device,
        wandb_project=args.wandb_project,
    )
    print("Training complete.")

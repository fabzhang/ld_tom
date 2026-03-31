"""
ppo_agent.py — MaskablePPO agent wrapper compatible with the unified act() interface.

Uses sb3_contrib.MaskablePPO with MLP policy [obs_size → 256 → 256 → n_actions].
Supports load/save and is compatible with tournament runners and self-play training.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional


class PPOAgent:
    """Wraps a trained MaskablePPO model with the unified act() interface."""

    def __init__(self, model_path: Optional[str] = None, name: str = "ppo",
                 deterministic: bool = False):
        self.name = name
        self.deterministic = deterministic
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def act(self, obs: np.ndarray, action_mask: np.ndarray,
            env_state: dict | None = None) -> int:
        if self.model is None:
            # Fallback to random if not loaded
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))
        obs_2d = obs[np.newaxis, :]  # (1, obs_size)
        action, _ = self.model.predict(
            obs_2d,
            action_masks=action_mask[np.newaxis, :],
            deterministic=self.deterministic,
        )
        return int(action[0])

    def reset(self):
        pass

    def load(self, path: str):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(path)

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)


def make_ppo_agent(obs_size: int, n_actions: int, lr: float = 3e-4,
                   batch_size: int = 2048, n_epochs: int = 10,
                   ent_coef: float = 0.05, gamma: float = 0.99,
                   device: str = "auto") -> "PPOAgent":
    """
    Create a fresh (untrained) PPOAgent with a MaskablePPO model.
    Used by training/self_play.py to initialize training.
    """
    from sb3_contrib import MaskablePPO
    from gymnasium import spaces

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
    act_space = spaces.Discrete(n_actions)

    # Dummy env needed for SB3 init — we use a thin wrapper
    dummy_env = _DummyEnvWrapper(obs_space, act_space)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=dummy_env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0,
        device=device,
        policy_kwargs={"net_arch": [256, 256]},
    )

    agent = PPOAgent(name="ppo")
    agent.model = model
    return agent


class _DummyEnvWrapper:
    """Minimal env object with observation_space and action_space for SB3 init."""
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space
        self.num_envs = 1
        self.metadata = {}
        self.render_mode = None
        self.spec = None
        self.np_random = np.random.default_rng()

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}

    def render(self):
        pass

    def close(self):
        pass

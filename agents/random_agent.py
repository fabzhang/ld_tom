"""
random_agent.py — Level-0 baseline: uniform random over legal actions.
"""
from __future__ import annotations
import numpy as np


class RandomAgent:
    """Samples uniformly from the legal action mask."""

    def __init__(self, name: str = "random"):
        self.name = name

    def act(self, obs: np.ndarray, action_mask: np.ndarray,
            env_state: dict | None = None) -> int:
        legal = np.where(action_mask)[0]
        return int(np.random.choice(legal))

    def reset(self):
        pass

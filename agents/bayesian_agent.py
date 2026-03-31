"""
bayesian_agent.py — Level-3 agent maintaining a Bayesian posterior over opponent dice.

For a 2-player game with N_opp opponent dice, the state space is 6^N_opp which is
tractable up to N_opp=5 (7,776 states).  For larger N_opp, falls back to a
particle-filter approximation with n_particles samples.
"""
from __future__ import annotations
import numpy as np
from itertools import product
from scipy.stats import binom


_MAX_EXACT = 5   # Use exact inference up to this many opponent dice


class BayesianAgent:
    def __init__(
        self,
        name: str = "bayesian",
        t_challenge: float = 0.3,
        n_particles: int = 2000,
    ):
        self.name = name
        self.t_challenge = t_challenge
        self.n_particles = n_particles
        self._posterior: np.ndarray | None = None
        self._particles: np.ndarray | None = None
        self._n_opp_dice: int = 0

    def reset(self):
        self._posterior = None
        self._particles = None
        self._n_opp_dice = 0

    def _init_posterior(self, n_opp_dice: int):
        self._n_opp_dice = n_opp_dice
        if n_opp_dice <= _MAX_EXACT:
            configs = list(product(range(1, 7), repeat=n_opp_dice))
            self._configs = np.array(configs, dtype=np.int8)
            self._posterior = np.ones(len(configs), dtype=np.float64)
            self._posterior /= self._posterior.sum()
        else:
            self._particles = np.random.randint(1, 7, size=(self.n_particles, n_opp_dice))

    def update(self, bid: tuple[int, int]):
        """Update posterior after observing opponent bid (q, f)."""
        if self._posterior is None and self._particles is None:
            return
        q, f = bid
        if self._posterior is not None:
            counts = np.sum(self._configs == f, axis=1)
            likelihood = (counts >= max(0, q - 2)).astype(np.float64) * 0.7 + 0.3
            self._posterior *= likelihood
            s = self._posterior.sum()
            if s > 0:
                self._posterior /= s
        else:
            counts = np.sum(self._particles == f, axis=1)
            weights = (counts >= max(0, q - 2)).astype(np.float64) * 0.7 + 0.3
            weights /= weights.sum()
            idxs = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
            self._particles = self._particles[idxs]
            jitter_mask = np.random.random(self.n_particles) < 0.05
            self._particles[jitter_mask] = np.random.randint(
                1, 7, size=(jitter_mask.sum(), self._n_opp_dice)
            )

    def p_bid_true(self, q: int, f: int, own_count: int) -> float:
        """P(total count of face f across all dice >= q)."""
        if self._posterior is not None:
            opp_counts = np.sum(self._configs == f, axis=1)
            total_counts = opp_counts + own_count
            return float(np.sum(self._posterior * (total_counts >= q)))
        elif self._particles is not None:
            opp_counts = np.sum(self._particles == f, axis=1)
            return float(np.mean(opp_counts + own_count >= q))
        else:
            n_opp = self._n_opp_dice or 5
            opp_needed = max(0, q - own_count)
            return 1.0 - binom.cdf(opp_needed - 1, n_opp, 1.0 / 6)

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        env_state: dict | None = None,
    ) -> int:
        """
        Args:
            obs: Full observation vector (not used directly).
            action_mask: Boolean mask of legal actions.
            env_state: Optional dict with keys:
                own_dice (np.ndarray), current_bid (tuple|None), n_opp_dice (int).
        """
        if env_state is None or env_state.get("current_bid") is None:
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))

        own_dice = env_state.get("own_dice")
        current_bid = env_state["current_bid"]
        n_opp_dice = env_state.get("n_opp_dice", 5)

        if own_dice is None:
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))

        if self._posterior is None and self._particles is None:
            self._init_posterior(n_opp_dice)

        # Update posterior with the observed bid
        self.update(current_bid)

        q, f = current_bid
        own_count = int(np.sum(own_dice == f))
        p_true = self.p_bid_true(q, f, own_count)

        if p_true < self.t_challenge and action_mask[0]:
            return 0  # Challenge

        # Pick the first (lowest) legal bid
        for idx, valid in enumerate(action_mask):
            if valid and idx != 0:
                return idx

        return 0  # Forced challenge

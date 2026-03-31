"""
heuristic_agent.py — Level-1 agent using binomial probability reasoning.

Strategy:
  - Compute P(true_count_of_face >= q) using binomial distribution
    over opponent dice (own dice are known).
  - Challenge if P < T_challenge.
  - Otherwise bid the smallest valid raise, with p_bluff chance of
    making the highest valid bid instead.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binom


class HeuristicAgent:
    def __init__(
        self,
        name: str = "heuristic",
        t_challenge: float = 0.25,
        p_bluff: float = 0.15,
    ):
        self.name = name
        self.t_challenge = t_challenge
        self.p_bluff = p_bluff

    def reset(self):
        pass

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
            # No bid yet or no info — pick random legal action
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))

        own_dice = env_state.get("own_dice")
        current_bid = env_state["current_bid"]
        n_opp_dice = env_state.get("n_opp_dice", 5)

        if own_dice is None:
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))

        q, f = current_bid
        own_count = int(np.sum(own_dice == f))
        opp_needed = max(0, q - own_count)

        # P(total count >= q) = P(Binomial(n_opp, 1/6) >= opp_needed)
        p_true = 1.0 - binom.cdf(opp_needed - 1, n_opp_dice, 1.0 / 6)

        if p_true < self.t_challenge and action_mask[0]:
            return 0  # Challenge

        # Collect all valid bid actions (non-challenge)
        legal_bids = [idx for idx, valid in enumerate(action_mask)
                      if valid and idx != 0]
        if not legal_bids:
            return 0  # Must challenge

        # Bluff: pick the highest valid bid
        if np.random.random() < self.p_bluff:
            return legal_bids[-1]

        # Default: smallest valid raise
        return legal_bids[0]

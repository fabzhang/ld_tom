"""
heuristic_agent.py — Level-1 agent using binomial probability reasoning.

Strategy:
  - Compute P(true_count_of_face >= q) using binomial distribution
    over opponent dice (own dice are known).
  - Challenge if P < T_challenge.
  - Otherwise bid the (q, f) that maximizes expected utility,
    with probability p_bluff of making an unsupported bid.
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
        own_dice: np.ndarray | None = None,
        current_bid: tuple[int, int] | None = None,
        n_opp_dice: int = 5,
    ) -> int:
        """
        Args:
            obs: Full observation (not used directly; convenience for policy API).
            action_mask: Boolean mask of legal actions.
            own_dice: Own dice values (1–6), shape (n_own_dice,).
            current_bid: (quantity, face) or None.
            n_opp_dice: Total opponent dice remaining.
        """
        if own_dice is None or current_bid is None:
            # Fallback: random legal action
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))

        q, f = current_bid
        own_count = int(np.sum(own_dice == f))
        opp_needed = max(0, q - own_count)

        # P(total count >= q) = P(Binomial(n_opp, 1/6) >= opp_needed)
        p_true = 1.0 - binom.cdf(opp_needed - 1, n_opp_dice, 1.0 / 6)

        if p_true < self.t_challenge and action_mask[0]:
            return 0  # Challenge

        # Find a valid bid to make
        legal_bids = [(idx, act) for idx, act in enumerate(action_mask) if act and idx != 0]
        if not legal_bids:
            return 0  # Must challenge

        # Occasionally bluff (pick a bid above what own dice support)
        if np.random.random() < self.p_bluff:
            idx, _ = legal_bids[-1]  # highest valid bid
            return idx

        # Bid the smallest valid bid above current that own dice can support
        for idx, _ in legal_bids:
            return idx  # first valid bid (lowest)

        return 0

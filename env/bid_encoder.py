"""
bid_encoder.py — Bid ↔ action index mapping for Liar's Dice.

Action space layout:
  0              → Challenge ("Liar!")
  1 … max_dice*6 → Bids, ordered (q=1,f=1), (q=1,f=2), …, (q=1,f=6), (q=2,f=1), …
  max_dice*6 + 1 → Calza (exact-count claim), only present if calza_enabled=True

A bid (q, f) maps to index: (q-1)*6 + (f-1) + 1
"""

from __future__ import annotations
import numpy as np


CHALLENGE = 0
N_FACES = 6


class BidEncoder:
    """Pre-computed lookup tables for bid ↔ action index conversion."""

    def __init__(self, max_dice: int, calza_enabled: bool = False):
        """
        Args:
            max_dice: Maximum total dice that can ever be in play
                      (= n_players * n_dice at game start).
            calza_enabled: Whether the Calza action is included.
        """
        self.max_dice = max_dice
        self.calza_enabled = calza_enabled
        self.n_bid_actions = max_dice * N_FACES  # indices 1 … max_dice*6
        self.calza_action = self.n_bid_actions + 1 if calza_enabled else None
        self.n_actions = self.n_bid_actions + 1 + (1 if calza_enabled else 0)

        # Forward: (quantity, face) → action index
        self._bid_to_idx: dict[tuple[int, int], int] = {}
        # Reverse: action index → (quantity, face)
        self._idx_to_bid: dict[int, tuple[int, int]] = {}

        for q in range(1, max_dice + 1):
            for f in range(1, N_FACES + 1):
                idx = (q - 1) * N_FACES + (f - 1) + 1
                self._bid_to_idx[(q, f)] = idx
                self._idx_to_bid[idx] = (q, f)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def bid_to_action(self, quantity: int, face: int) -> int:
        return self._bid_to_idx[(quantity, face)]

    def action_to_bid(self, action: int) -> tuple[int, int]:
        """Returns (quantity, face). Raises ValueError for non-bid actions."""
        if action == CHALLENGE:
            raise ValueError("Action 0 is Challenge, not a bid.")
        if self.calza_enabled and action == self.calza_action:
            raise ValueError(f"Action {action} is Calza, not a bid.")
        if action not in self._idx_to_bid:
            raise ValueError(f"Unknown action index: {action}")
        return self._idx_to_bid[action]

    def is_bid_action(self, action: int) -> bool:
        return action in self._idx_to_bid

    # ------------------------------------------------------------------
    # Validity checking
    # ------------------------------------------------------------------

    def is_valid_action(
        self,
        action: int,
        current_bid: tuple[int, int] | None,
        total_dice: int,
    ) -> bool:
        """True if `action` is legal given the current game state.

        Args:
            action: Proposed action index.
            current_bid: (quantity, face) of the last bid, or None for round start.
            total_dice: Total dice still in play (across all players).
        """
        if action == CHALLENGE:
            # Can only challenge if there is an active bid.
            return current_bid is not None

        if self.calza_enabled and action == self.calza_action:
            # Calza only valid when there is an active bid.
            return current_bid is not None

        if action not in self._idx_to_bid:
            return False

        q, f = self._idx_to_bid[action]
        if q > total_dice:
            return False
        if current_bid is None:
            return True

        cq, cf = current_bid
        # New bid must be strictly higher: higher quantity, or same quantity + higher face.
        return (q > cq) or (q == cq and f > cf)

    def get_action_mask(
        self,
        current_bid: tuple[int, int] | None,
        total_dice: int,
    ) -> np.ndarray:
        """Boolean array of length n_actions; True = legal action."""
        mask = np.zeros(self.n_actions, dtype=bool)
        for idx in range(self.n_actions):
            mask[idx] = self.is_valid_action(idx, current_bid, total_dice)
        return mask

    # ------------------------------------------------------------------
    # Observation encoding helpers
    # ------------------------------------------------------------------

    def encode_bid(self, bid: tuple[int, int] | None) -> np.ndarray:
        """Encode a bid as a 7-dim vector: [q_norm, f_one_hot(6)].
        All zeros + a leading -1 if bid is None (no current bid).
        Returns shape (8,): [has_bid, q_norm, f1, f2, f3, f4, f5, f6]
        """
        vec = np.zeros(8, dtype=np.float32)
        if bid is not None:
            q, f = bid
            vec[0] = 1.0                         # has_bid flag
            vec[1] = q / self.max_dice           # normalized quantity
            vec[2 + (f - 1)] = 1.0              # face one-hot
        return vec

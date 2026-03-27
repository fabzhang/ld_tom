"""
liars_dice_env.py — Liar's Dice as a PettingZoo AEC environment.

Observation per agent (flat float32 array, named segments):
  [own_dice_onehot | current_bid | per_player_die_counts | bid_history | round_features]

  own_dice_onehot  : (n_dice * 6,)   — each die as face one-hot
  current_bid      : (8,)            — [has_bid, q_norm, f1..f6]
  per_player_die_counts : (n_players,) — normalized die counts per player
  bid_history      : (history_len * bid_entry_size,)
                     bid_entry: [player_id_onehot(n_players), q_norm, f1..f6, is_challenge]
                     = n_players + 1 + 6 + 1 = n_players + 8
  round_features   : (2,)            — [round_norm, total_dice_norm]

Actions:
  0        → Challenge
  1..B     → Bids (see BidEncoder)
  B+1      → Calza (if enabled)

Rewards: sparse terminal only. +1 for the last surviving player, -1 for all others.
         All other steps get reward 0.
"""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from env.bid_encoder import BidEncoder, CHALLENGE


class LiarsDiceEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "liars_dice_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        n_players: int = 2,
        n_dice: int = 5,
        history_len: int = 20,
        calza_enabled: bool = False,
        max_rounds: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert 2 <= n_players <= 6, "n_players must be 2–6"
        assert 1 <= n_dice <= 10, "n_dice must be 1–10"

        self.n_players = n_players
        self.n_dice = n_dice
        self.history_len = history_len
        self.calza_enabled = calza_enabled
        self.max_rounds = max_rounds
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(n_players)]
        self._agent_to_idx = {a: i for i, a in enumerate(self.possible_agents)}

        self._max_total_dice = n_players * n_dice
        self.encoder = BidEncoder(
            max_dice=self._max_total_dice, calza_enabled=calza_enabled
        )

        # Compute observation size
        self._own_dice_size = n_dice * 6
        self._bid_size = 8                        # [has_bid, q_norm, f1..f6]
        self._die_counts_size = n_players
        self._bid_entry_size = n_players + 8      # [player_onehot | q_norm | f1..f6 | is_challenge]
        self._history_size = history_len * self._bid_entry_size
        self._round_features_size = 2
        self._obs_size = (
            self._own_dice_size
            + self._bid_size
            + self._die_counts_size
            + self._history_size
            + self._round_features_size
        )

        obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        act_space = spaces.Discrete(self.encoder.n_actions)

        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

        # Game state (initialised in reset)
        self._dice: dict[str, np.ndarray] = {}
        self._die_counts: dict[str, int] = {}
        self._current_bid: Optional[tuple[int, int]] = None
        self._current_bidder: Optional[str] = None
        self._history: list[dict] = []
        self._round: int = 0
        self._total_rounds: int = 0
        self._round_starter: str = ""
        self._pending_winner: Optional[str] = None  # winner to start next round after dead-steps

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = list(self.possible_agents)
        self._die_counts = {a: self.n_dice for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._roll_dice()
        self._current_bid = None
        self._current_bidder = None
        self._history = []
        self._round = 0
        self._total_rounds = 0
        self._round_starter = self.agents[0]
        self._pending_winner = None

        # Indexed by possible_agents so post-game access works after agents list empties
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str) -> np.ndarray:
        return self._build_obs(agent)

    def step(self, action: Optional[int]):
        agent = self.agent_selection

        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            # Dead agents must receive None per AEC spec; be lenient if caller forgets
            self._was_dead_step(None)
            return

        # Validate action
        total_dice = sum(self._die_counts[a] for a in self.agents
                         if not self.terminations.get(a, False))
        mask = self.encoder.get_action_mask(self._current_bid, total_dice)
        if action is None or not mask[action]:
            action = CHALLENGE

        # Zero out rewards each step
        self.rewards = {a: 0.0 for a in self.agents}
        info_extra: dict = {}

        is_challenge_or_calza = False
        if action == CHALLENGE:
            self._handle_challenge(agent, info_extra)
            is_challenge_or_calza = True
        elif self.calza_enabled and action == self.encoder.calza_action:
            self._handle_calza(agent, info_extra)
            is_challenge_or_calza = True
        else:
            self._handle_bid(agent, action, info_extra)

        self.infos[agent] = info_extra

        # Accumulate rewards (must happen before any agent removal)
        for a in list(self.agents):
            self._cumulative_rewards[a] += self.rewards.get(a, 0.0)

        # After a challenge/calza that caused eliminations, point agent_selection at the
        # first terminated agent so the external loop can dead-step it.  The new round
        # (stored in _pending_winner) will be started inside _was_dead_step once all
        # terminated agents have been processed.
        if is_challenge_or_calza:
            terminated_now = [a for a in self.agents
                              if self.terminations.get(a, False) or self.truncations.get(a, False)]
            if terminated_now:
                self.agent_selection = terminated_now[0]
            # else: calza that didn't cause elimination — fall through to normal advance
            else:
                if self.agents:
                    self.agent_selection = self._agent_selector.next()
        else:
            # Bid: advance to next player
            if self.agents:
                self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def _was_dead_step(self, action: Optional[int]) -> None:
        """Remove a terminated/truncated agent and start the next round if needed."""
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        agent = self.agent_selection
        assert (
            self.terminations.get(agent, False) or self.truncations.get(agent, False)
        ), f"_was_dead_step called on non-dead agent {agent}"

        # Remove agent from all tracking dicts.
        # _cumulative_rewards is intentionally kept so tests can read it via possible_agents.
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        if not self.agents:
            # All agents gone — game over.
            self._clear_rewards()
            return

        # Check for more terminated/truncated agents to process
        more_dead = [a for a in self.agents
                     if self.terminations.get(a, False) or self.truncations.get(a, False)]
        if more_dead:
            self.agent_selection = more_dead[0]
        else:
            # All dead agents processed.  Start the next round if the game continues.
            if self._pending_winner is not None:
                winner = self._pending_winner
                self._pending_winner = None
                self._start_new_round(starter=winner)
            # else: truncation or unexpected state; agent_selection stays wherever it is

        self._clear_rewards()

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n--- Round {self._round} | Bid: {self._current_bid} "
              f"| Dice: {self._die_counts} ---")
        print(f"  Current agent: {self.agent_selection}")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_bid(self, agent: str, action: int, info: dict):
        q, f = self.encoder.action_to_bid(action)

        own_count = int(np.sum(self._dice[agent] == f))
        is_bluff = own_count < q
        info["bid"] = (q, f)
        info["is_bluff"] = is_bluff
        info["event"] = "bid"

        self._current_bid = (q, f)
        self._current_bidder = agent
        self._history.append({
            "agent": agent,
            "type": "bid",
            "quantity": q,
            "face": f,
        })

    def _handle_challenge(self, agent: str, info: dict):
        """Challenger: `agent`.  Target: `self._current_bidder`."""
        if self._current_bid is None or self._current_bidder is None:
            return

        q, f = self._current_bid
        bidder = self._current_bidder

        total_count = sum(
            int(np.sum(self._dice[a] == f))
            for a in self.agents
            if not self.terminations.get(a, False)
        )
        bid_was_valid = total_count >= q
        info["event"] = "challenge"
        info["challenger"] = agent
        info["bidder"] = bidder
        info["bid"] = (q, f)
        info["true_count"] = total_count
        info["bid_was_valid"] = bid_was_valid
        info["challenge_correct"] = not bid_was_valid

        self._history.append({
            "agent": agent,
            "type": "challenge",
            "bid": (q, f),
            "true_count": total_count,
            "bid_was_valid": bid_was_valid,
        })

        loser = agent if bid_was_valid else bidder
        winner = bidder if loser == agent else agent
        info["loser"] = loser
        info["winner"] = winner

        self._lose_die(loser)

        # Check how many active (non-terminated) agents remain
        active = [a for a in self.agents if not self.terminations.get(a, False)]
        if len(active) >= 2:
            # Game continues — record pending winner; round will start after dead-steps
            self._pending_winner = winner
        else:
            self._pending_winner = None

    def _handle_calza(self, agent: str, info: dict):
        """Calza: agent claims the bid is exactly correct."""
        if self._current_bid is None:
            return

        q, f = self._current_bid
        total_count = sum(
            int(np.sum(self._dice[a] == f))
            for a in self.agents
            if not self.terminations.get(a, False)
        )
        exact = (total_count == q)
        info["event"] = "calza"
        info["agent"] = agent
        info["bid"] = (q, f)
        info["true_count"] = total_count
        info["calza_correct"] = exact

        self._history.append({
            "agent": agent,
            "type": "calza",
            "bid": (q, f),
            "true_count": total_count,
            "exact": exact,
        })

        if exact:
            self._die_counts[agent] = min(self._die_counts[agent] + 1, self.n_dice)
            self._dice[agent] = np.random.randint(1, 7, size=self._die_counts[agent])
            info["result"] = "gained_die"
        else:
            self._lose_die(agent)
            info["result"] = "lost_die"

        active = [a for a in self.agents if not self.terminations.get(a, False)]
        if len(active) >= 2:
            self._pending_winner = agent
        else:
            self._pending_winner = None

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def _lose_die(self, agent: str):
        """Remove one die from agent; eliminate if they reach 0."""
        self._die_counts[agent] -= 1
        if self._die_counts[agent] <= 0:
            self._eliminate(agent)

    def _eliminate(self, agent: str):
        """Mark agent as terminated. Removal from self.agents happens in _was_dead_step."""
        self.terminations[agent] = True
        self.rewards[agent] = -1.0

        # If only one non-terminated agent remains, they win
        active = [a for a in self.agents if not self.terminations.get(a, False)]
        if len(active) == 1:
            winner = active[0]
            self.terminations[winner] = True
            self.rewards[winner] = 1.0

        # Truncation guard
        self._total_rounds += 1
        if self._total_rounds >= self.max_rounds:
            for a in self.agents:
                if not self.terminations.get(a, False) and not self.truncations.get(a, False):
                    self.truncations[a] = True

    def _start_new_round(self, starter: str):
        """Reset bid, re-roll active dice, rebuild agent selector starting from `starter`."""
        self._current_bid = None
        self._current_bidder = None
        self._round += 1
        self._roll_dice()
        self._round_starter = starter

        # At this point all terminated agents have been removed from self.agents by _was_dead_step
        if self.agents:
            start_idx = self.agents.index(starter) if starter in self.agents else 0
            ordered = self.agents[start_idx:] + self.agents[:start_idx]
            self._agent_selector = agent_selector(ordered)
            self.agent_selection = self._agent_selector.reset()

    def _roll_dice(self):
        for a in self.agents:
            if not self.terminations.get(a, False):
                n = self._die_counts.get(a, self.n_dice)
                self._dice[a] = np.random.randint(1, 7, size=n)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, agent: str) -> np.ndarray:
        parts: list[np.ndarray] = []

        # 1. Own dice (one-hot per die, shape: n_dice * 6)
        dice = self._dice.get(agent, np.array([], dtype=np.int64))
        own_onehot = np.zeros(self.n_dice * 6, dtype=np.float32)
        for i, d in enumerate(dice):
            if i < self.n_dice:
                own_onehot[i * 6 + (d - 1)] = 1.0
        parts.append(own_onehot)

        # 2. Current bid (8-dim)
        parts.append(self.encoder.encode_bid(self._current_bid))

        # 3. Per-player die counts (normalized)
        counts = np.array(
            [self._die_counts.get(a, 0) / self.n_dice for a in self.possible_agents],
            dtype=np.float32,
        )
        parts.append(counts)

        # 4. Bid history (history_len most recent events)
        # bid_entry layout: [player_onehot(n_players) | q_norm | f1..f6 | is_challenge]
        #   indices:          0..n_players-1            | n_p    | n_p+1..n_p+6 | n_p+7
        history_vec = np.zeros(
            self.history_len * self._bid_entry_size, dtype=np.float32
        )
        recent = self._history[-self.history_len:]
        for i, event in enumerate(reversed(recent)):
            offset = i * self._bid_entry_size
            aidx = self._agent_to_idx.get(event["agent"], 0)
            history_vec[offset + aidx] = 1.0
            if event["type"] == "bid":
                q, f = event["quantity"], event["face"]
                history_vec[offset + self.n_players] = q / self._max_total_dice          # q_norm
                history_vec[offset + self.n_players + 1 + (f - 1)] = 1.0                 # face one-hot
            elif event["type"] in ("challenge", "calza"):
                history_vec[offset + self.n_players + 7] = 1.0                           # is_challenge
        parts.append(history_vec)

        # 5. Round features
        total_dice = sum(self._die_counts.get(a, 0) for a in self.possible_agents)
        parts.append(np.array([
            self._round / self.max_rounds,
            total_dice / self._max_total_dice,
        ], dtype=np.float32))

        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def action_mask(self, agent: Optional[str] = None) -> np.ndarray:
        """Return action mask for the given (or current) agent."""
        if agent is None:
            agent = self.agent_selection
        total_dice = sum(
            self._die_counts.get(a, 0)
            for a in self.possible_agents
            if not self.terminations.get(a, False)
        )
        return self.encoder.get_action_mask(self._current_bid, total_dice)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self.action_spaces[agent]

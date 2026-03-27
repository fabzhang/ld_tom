"""
tests/test_env.py — Basic correctness tests for LiarsDiceEnv.
Run with: pytest liars_dice_tom/env/tests/test_env.py -v
"""
from __future__ import annotations
import numpy as np
import pytest
from pettingzoo.test import api_test

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from env.liars_dice_env import LiarsDiceEnv
from env.bid_encoder import BidEncoder, CHALLENGE


# ------------------------------------------------------------------ helpers

def make_env(**kwargs):
    return LiarsDiceEnv(**kwargs)


def run_random_game(env: LiarsDiceEnv, seed: int = 0):
    """Play one game with random agents; return game log."""
    env.reset(seed=seed)
    log = []
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        mask = env.action_mask(agent)
        legal = np.where(mask)[0]
        action = int(np.random.choice(legal))
        env.step(action)
        log.append({"agent": agent, "action": action, "reward": reward})
    return log


# ------------------------------------------------------------------ Test 1: action validity

class TestBidEncoder:
    def test_challenge_only_after_bid(self):
        enc = BidEncoder(max_dice=10)
        # No current bid → challenge invalid
        assert not enc.is_valid_action(CHALLENGE, current_bid=None, total_dice=10)
        # After a bid → challenge valid
        assert enc.is_valid_action(CHALLENGE, current_bid=(2, 3), total_dice=10)

    def test_bid_must_be_higher(self):
        enc = BidEncoder(max_dice=10)
        current = (3, 4)
        # Same quantity, lower face → invalid
        assert not enc.is_valid_action(enc.bid_to_action(3, 3), current, 10)
        # Same quantity, same face → invalid
        assert not enc.is_valid_action(enc.bid_to_action(3, 4), current, 10)
        # Same quantity, higher face → valid
        assert enc.is_valid_action(enc.bid_to_action(3, 5), current, 10)
        # Higher quantity, any face → valid
        assert enc.is_valid_action(enc.bid_to_action(4, 1), current, 10)

    def test_bid_cannot_exceed_total_dice(self):
        enc = BidEncoder(max_dice=10)
        assert not enc.is_valid_action(enc.bid_to_action(5, 1), None, total_dice=4)
        assert enc.is_valid_action(enc.bid_to_action(4, 1), None, total_dice=4)

    def test_roundtrip(self):
        enc = BidEncoder(max_dice=10)
        for q in range(1, 11):
            for f in range(1, 7):
                idx = enc.bid_to_action(q, f)
                assert enc.action_to_bid(idx) == (q, f)

    def test_mask_shape(self):
        enc = BidEncoder(max_dice=10)
        mask = enc.get_action_mask(None, total_dice=10)
        assert mask.shape == (enc.n_actions,)
        # At round start: challenge invalid, all bids up to total_dice valid
        assert not mask[CHALLENGE]
        assert mask[enc.bid_to_action(1, 1)]

    def test_encode_bid_shape(self):
        enc = BidEncoder(max_dice=10)
        vec = enc.encode_bid(None)
        assert vec.shape == (8,)
        assert vec[0] == 0.0    # no bid
        vec2 = enc.encode_bid((3, 4))
        assert vec2[0] == 1.0   # has bid
        assert vec2[2 + 3] == 1.0  # face 4 → index 5


# ------------------------------------------------------------------ Test 2: terminal conditions

class TestTerminalConditions:
    def test_game_terminates(self):
        """A randomly-played game must eventually end."""
        env = make_env(n_players=2, n_dice=2, max_rounds=200)
        log = run_random_game(env, seed=42)
        assert len(env.agents) == 0  # all agents terminated
        # Exactly one agent should have reward +1
        rewards = {a: env._cumulative_rewards[a] for a in env.possible_agents}
        winners = [a for a, r in rewards.items() if r > 0]
        assert len(winners) == 1

    def test_loser_gets_negative_reward(self):
        env = make_env(n_players=2, n_dice=2, max_rounds=200)
        run_random_game(env, seed=7)
        rewards = {a: env._cumulative_rewards[a] for a in env.possible_agents}
        losers = [a for a, r in rewards.items() if r < 0]
        assert len(losers) >= 1


# ------------------------------------------------------------------ Test 3: observation determinism

class TestObservation:
    def test_obs_deterministic(self):
        env = make_env(n_players=2, n_dice=3)
        env.reset(seed=0)
        agent = env.agent_selection
        obs1 = env.observe(agent)
        obs2 = env.observe(agent)
        np.testing.assert_array_equal(obs1, obs2)

    def test_obs_shape(self):
        for n_players in (2, 4):
            for n_dice in (3, 5):
                env = make_env(n_players=n_players, n_dice=n_dice)
                env.reset(seed=0)
                agent = env.agent_selection
                obs = env.observe(agent)
                assert obs.shape == (env._obs_size,), (
                    f"Expected {env._obs_size}, got {obs.shape} "
                    f"for {n_players}p/{n_dice}d"
                )

    def test_obs_in_bounds(self):
        env = make_env(n_players=2, n_dice=5)
        env.reset(seed=1)
        for _ in range(50):
            agent = env.agent_selection
            obs = env.observe(agent)
            assert obs.min() >= -1.0 and obs.max() <= 1.0
            mask = env.action_mask(agent)
            legal = np.where(mask)[0]
            env.step(int(np.random.choice(legal)))
            if not env.agents:
                break


# ------------------------------------------------------------------ Test 4: PettingZoo API compliance

class TestPettingZooAPI:
    def test_api_2player(self):
        env = make_env(n_players=2, n_dice=3, max_rounds=100)
        api_test(env, num_cycles=50, verbose_progress=False)

    def test_api_4player(self):
        env = make_env(n_players=4, n_dice=3, max_rounds=100)
        api_test(env, num_cycles=50, verbose_progress=False)


# ------------------------------------------------------------------ Test 5: dice distribution sanity

class TestDiceDistribution:
    def test_uniform_faces(self):
        """Over many games, face distribution should be approximately uniform."""
        env = make_env(n_players=2, n_dice=5)
        face_counts = np.zeros(6, dtype=int)
        for seed in range(200):
            env.reset(seed=seed)
            for agent, dice in env._dice.items():
                for d in dice:
                    face_counts[d - 1] += 1

        total = face_counts.sum()
        probs = face_counts / total
        # Each face should be ~1/6; allow ±3% tolerance
        expected = 1.0 / 6
        for i, p in enumerate(probs):
            assert abs(p - expected) < 0.03, (
                f"Face {i+1} frequency {p:.3f} deviates from {expected:.3f}"
            )

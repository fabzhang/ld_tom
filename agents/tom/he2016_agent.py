"""
agents/tom/he2016_agent.py — DRON-MoE (He et al. 2016) agent for Liar's Dice.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

OPPONENT_TYPES = ["random", "heuristic", "bayesian", "ppo"]
K = len(OPPONENT_TYPES)
OPP_ACTION_DIM = 8
HISTORY_LEN = 10


def encode_opp_action(event: dict, max_total_dice: int = 10) -> np.ndarray:
    vec = np.zeros(OPP_ACTION_DIM, dtype=np.float32)
    if event["type"] == "bid":
        q, f = event["quantity"], event["face"]
        vec[0] = q / max(max_total_dice, 1)
        vec[1 + (f - 1)] = 1.0
    elif event["type"] in ("challenge", "calza"):
        vec[7] = 1.0
    return vec


class InferenceNet(nn.Module):
    def __init__(self, history_len: int = HISTORY_LEN,
                 action_dim: int = OPP_ACTION_DIM, k: int = K, hidden: int = 128):
        super().__init__()
        input_dim = history_len * action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, k),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(history), dim=-1)


class He2016Agent:
    def __init__(self, expert_models: list, inference_net: InferenceNet,
                 opponent_types: list = OPPONENT_TYPES,
                 max_total_dice: int = 10, history_len: int = HISTORY_LEN,
                 deterministic: bool = False, name: str = "he2016"):
        self.expert_models = expert_models
        self.inference_net = inference_net
        self.opponent_types = opponent_types
        self.max_total_dice = max_total_dice
        self.history_len = history_len
        self.deterministic = deterministic
        self.name = name
        self.inference_net.eval()
        self._opp_history_buf = []
        self._seen_event_count = 0

    def reset(self):
        self._opp_history_buf = []
        self._seen_event_count = 0

    def _update_history(self, opp_action_history: list):
        new_events = opp_action_history[self._seen_event_count:]
        self._seen_event_count = len(opp_action_history)
        for event in new_events:
            enc = encode_opp_action(event, self.max_total_dice)
            self._opp_history_buf.append(enc)
        if len(self._opp_history_buf) > self.history_len:
            self._opp_history_buf = self._opp_history_buf[-self.history_len:]

    def _get_history_tensor(self) -> torch.Tensor:
        buf = list(self._opp_history_buf)
        while len(buf) < self.history_len:
            buf.insert(0, np.zeros(OPP_ACTION_DIM, dtype=np.float32))
        arr = np.concatenate(buf, axis=0)
        return torch.from_numpy(arr).unsqueeze(0)

    def _get_gate_weights(self) -> np.ndarray:
        h = self._get_history_tensor()
        with torch.no_grad():
            weights = self.inference_net(h).squeeze(0).numpy()
        return weights

    def _get_expert_logits(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).unsqueeze(0).float()
        logits_list = []
        for model in self.expert_models:
            if model is None:
                logits_list.append(np.zeros(obs_t.shape[0], dtype=np.float32))
                continue
            with torch.no_grad():
                policy = model.policy
                features = policy.extract_features(obs_t, policy.pi_features_extractor)
                latent_pi, _ = policy.mlp_extractor(features)
                logits = policy.action_net(latent_pi)
                logits_list.append(logits.squeeze(0).numpy())
        return np.stack(logits_list, axis=0)

    def act(self, obs: np.ndarray, action_mask: np.ndarray,
            env_state: dict = None) -> int:
        if not self.expert_models or env_state is None:
            legal = np.where(action_mask)[0]
            return int(np.random.choice(legal))
        opp_history = env_state.get("opp_action_history", [])
        self._update_history(opp_history)
        weights = self._get_gate_weights()
        expert_logits = self._get_expert_logits(obs)
        mixed_logits = (weights[:, np.newaxis] * expert_logits).sum(axis=0)
        mixed_logits = np.where(action_mask, mixed_logits, -1e8)
        if self.deterministic:
            return int(np.argmax(mixed_logits))
        logits_shifted = mixed_logits - mixed_logits.max()
        probs = np.exp(logits_shifted)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def get_opponent_type_belief(self) -> dict:
        weights = self._get_gate_weights()
        return {t: float(w) for t, w in zip(self.opponent_types, weights)}

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.inference_net.state_dict(),
                   os.path.join(directory, "inference_net.pt"))
        import json
        config = {"opponent_types": self.opponent_types,
                  "max_total_dice": self.max_total_dice,
                  "history_len": self.history_len}
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, directory: str, expert_dirs: dict,
             deterministic: bool = False) -> "He2016Agent":
        import json
        from sb3_contrib import MaskablePPO
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        opponent_types = config["opponent_types"]
        history_len = config.get("history_len", HISTORY_LEN)
        inference_net = InferenceNet(history_len=history_len, k=len(opponent_types))
        state_dict = torch.load(os.path.join(directory, "inference_net.pt"),
                                map_location="cpu")
        inference_net.load_state_dict(state_dict)
        expert_models = []
        for opp_type in opponent_types:
            path = expert_dirs.get(opp_type)
            if path and os.path.exists(path):
                expert_models.append(MaskablePPO.load(path, device="cpu"))
            else:
                expert_models.append(None)
        return cls(expert_models=expert_models, inference_net=inference_net,
                   opponent_types=opponent_types,
                   max_total_dice=config.get("max_total_dice", 10),
                   history_len=history_len, deterministic=deterministic)

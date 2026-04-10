"""
training/he2016_training.py — Phase 2 of He2016 training: supervised inference net.

After expert_training.py has trained K expert policies, this script:
  1. Simulates many games against each opponent type, collecting opponent
     action histories with type labels.
  2. Trains the InferenceNet with cross-entropy loss to classify opponent type
     from action history.
  3. Saves the combined He2016Agent (inference net + expert pointers).

Usage:
    python training/he2016_training.py \
        --experts_dir checkpoints/he2016_experts \
        --ppo_path checkpoints/ppo_stage2/final_model.zip \
        --output_dir checkpoints/he2016 \
        --n_games_per_type 2000
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agents.tom.he2016_agent import (
    InferenceNet, He2016Agent, OPPONENT_TYPES, HISTORY_LEN,
    OPP_ACTION_DIM, encode_opp_action, K,
)


def _build_env_state(env, agent: str) -> dict:
    opp_history = [e for e in env._history if e["agent"] != agent]
    return {
        "own_dice": env._dice[agent],
        "current_bid": env._current_bid,
        "n_opp_dice": sum(env._die_counts[a] for a in env.agents if a != agent),
        "die_counts": dict(env._die_counts),
        "agent_name": agent,
        "opp_action_history": opp_history,
    }


def collect_histories(
    opponent_type: str,
    opponent_agent,
    n_games: int,
    env_config: dict,
    focal_agent=None,
) -> list[list[dict]]:
    """
    Play n_games with focal_agent vs opponent_agent.
    Returns list of opponent action history lists (one per game).
    focal_agent defaults to random if None.
    """
    from env.liars_dice_env import LiarsDiceEnv
    from agents.random_agent import RandomAgent

    if focal_agent is None:
        focal_agent = RandomAgent()

    histories = []
    rng = np.random.default_rng(hash(opponent_type) % (2**31))

    for _ in range(n_games):
        env = LiarsDiceEnv(**env_config)
        env.reset(seed=int(rng.integers(0, 2**31)))

        if hasattr(focal_agent, "reset"):
            focal_agent.reset()
        if hasattr(opponent_agent, "reset"):
            opponent_agent.reset()

        agent_map = {
            "player_0": focal_agent,
            "player_1": opponent_agent,
        }

        while env.agents:
            current = env.agent_selection
            if (env.terminations.get(current, False) or
                    env.truncations.get(current, False)):
                env.step(None)
                continue
            obs = env.observe(current).astype(np.float32)
            mask = env.action_mask(current).astype(bool)
            env_state = _build_env_state(env, current)
            action = agent_map[current].act(obs, mask, env_state)
            env.step(action)

        # Extract opponent (player_1) action history
        opp_history = [e for e in env._history if e["agent"] == "player_1"]
        histories.append(opp_history)

    return histories


def histories_to_tensors(
    histories: list[list[dict]],
    labels: list[int],
    max_total_dice: int = 10,
    history_len: int = HISTORY_LEN,
    action_dim: int = OPP_ACTION_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert game histories + type labels to (X, y) tensors for training.

    For each game, sample multiple windows from the history to augment data.
    X: (N, history_len * action_dim)
    y: (N,) long tensor of opponent type indices
    """
    X_list = []
    y_list = []

    for hist, label in zip(histories, labels):
        # Encode all opponent actions in this game
        encoded = [encode_opp_action(e, max_total_dice) for e in hist]
        if not encoded:
            continue

        # Sample windows from different points in the game
        # (early, middle, end, and the full last history_len)
        game_len = len(encoded)
        sample_points = set()
        # Always include the full end window
        sample_points.add(game_len)
        # Add a few intermediate points
        for frac in [0.25, 0.5, 0.75]:
            sample_points.add(max(1, int(game_len * frac)))

        for endpoint in sample_points:
            window = encoded[:endpoint]
            # Pad front with zeros
            padded = []
            while len(padded) + len(window) < history_len:
                padded.append(np.zeros(action_dim, dtype=np.float32))
            padded.extend(window[-history_len:])
            flat = np.concatenate(padded, axis=0)  # (history_len * action_dim,)
            X_list.append(flat)
            y_list.append(label)

    X = torch.from_numpy(np.stack(X_list, axis=0)).float()
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


def train_inference_net(
    X: torch.Tensor,
    y: torch.Tensor,
    k: int = K,
    history_len: int = HISTORY_LEN,
    hidden: int = 128,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> InferenceNet:
    """Train the inference network with cross-entropy loss."""
    net = InferenceNet(history_len=history_len, k=k, hidden=hidden).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X.to(device), y.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = net.net(xb)  # raw logits before softmax
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = correct / total
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"loss={total_loss/total:.4f} | acc={acc:.1%}")

    net.eval()
    return net.cpu()


def build_and_save_he2016(
    experts_dir: str,
    inference_net: InferenceNet,
    output_dir: str,
    opponent_types: list[str] = OPPONENT_TYPES,
    max_total_dice: int = 10,
):
    """Save the combined He2016 agent (inference net + config)."""
    os.makedirs(output_dir, exist_ok=True)

    # Save inference net
    torch.save(inference_net.state_dict(),
               os.path.join(output_dir, "inference_net.pt"))

    # Save config with expert paths
    expert_paths = {}
    for opp_type in opponent_types:
        candidate = os.path.join(experts_dir, f"expert_{opp_type}.zip")
        if os.path.exists(candidate):
            expert_paths[opp_type] = candidate
        else:
            print(f"[he2016_training] Warning: no expert found for {opp_type!r} at {candidate}")

    config = {
        "opponent_types": opponent_types,
        "expert_paths": expert_paths,
        "max_total_dice": max_total_dice,
        "history_len": HISTORY_LEN,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[he2016_training] He2016 agent saved to {output_dir}/")
    print(f"  Experts found: {list(expert_paths.keys())}")


def evaluate_inference_net(net: InferenceNet, X: torch.Tensor, y: torch.Tensor,
                            opponent_types: list[str] = OPPONENT_TYPES):
    """Print per-class accuracy."""
    net.eval()
    with torch.no_grad():
        probs = net(X)
        preds = probs.argmax(dim=1)
    correct_total = (preds == y).float().mean().item()
    print(f"\nInference net accuracy: {correct_total:.1%}")
    for k_idx, opp_type in enumerate(opponent_types):
        mask = (y == k_idx)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == k_idx).float().mean().item()
        print(f"  {opp_type:>12}: {acc:.1%} ({mask.sum().item()} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts_dir", default="checkpoints/he2016_experts")
    parser.add_argument("--ppo_path", default="checkpoints/ppo_stage2/final_model.zip")
    parser.add_argument("--output_dir", default="checkpoints/he2016")
    parser.add_argument("--n_games_per_type", type=int, default=2000)
    parser.add_argument("--env_n_dice", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env_config = {"n_players": 2, "n_dice": args.env_n_dice}
    max_total_dice = 2 * args.env_n_dice

    from training.expert_training import load_opponent

    # -------------------------------------------------------------------------
    # Step 1: Collect labeled opponent action histories
    # -------------------------------------------------------------------------
    print("=== Step 1: Collecting opponent histories ===")
    all_histories = []
    all_labels = []

    for k_idx, opp_type in enumerate(OPPONENT_TYPES):
        ppo_path = args.ppo_path if opp_type == "ppo" else None
        opponent = load_opponent(opp_type, ppo_path=ppo_path)
        print(f"  Collecting {args.n_games_per_type} games vs {opp_type!r}...")
        histories = collect_histories(
            opponent_type=opp_type,
            opponent_agent=opponent,
            n_games=args.n_games_per_type,
            env_config=env_config,
        )
        all_histories.extend(histories)
        all_labels.extend([k_idx] * len(histories))
        avg_len = np.mean([len(h) for h in histories])
        print(f"    Avg opponent actions per game: {avg_len:.1f}")

    # -------------------------------------------------------------------------
    # Step 2: Build tensors + train inference net
    # -------------------------------------------------------------------------
    print(f"\n=== Step 2: Training inference net ({len(all_histories)} games total) ===")
    X, y = histories_to_tensors(
        all_histories, all_labels,
        max_total_dice=max_total_dice,
        history_len=HISTORY_LEN,
    )
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]}-dim input, {K} classes")

    # Train/val split
    n = X.shape[0]
    idx = torch.randperm(n)
    split = int(0.9 * n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val, y_val = X[idx[split:]], y[idx[split:]]

    inference_net = train_inference_net(
        X_train, y_train,
        k=K,
        n_epochs=args.n_epochs,
        device=args.device,
    )

    print("\nValidation set:")
    evaluate_inference_net(inference_net, X_val, y_val)

    # -------------------------------------------------------------------------
    # Step 3: Save He2016 agent
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Saving He2016 agent ===")
    build_and_save_he2016(
        experts_dir=args.experts_dir,
        inference_net=inference_net,
        output_dir=args.output_dir,
        max_total_dice=max_total_dice,
    )
    print("Done.")

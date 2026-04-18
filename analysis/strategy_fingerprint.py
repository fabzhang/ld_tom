"""
analysis/strategy_fingerprint.py

Extract multi-dimensional strategy fingerprint vectors per agent and run PCA.
Captures behavioral patterns beyond challenge calibration:
  - Challenge calibration (5 bins of p_valid)
  - Bid aggression (mean quantity increment over minimum)
  - Face preference distribution (6 features)
  - Positional bias (challenge rate diff player_0 vs player_1)
  - Bluff rate (bid qty > own face count)
  - Bid jump consistency (std of quantity increments)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json


# ---------------------------------------------------------------------------
# Fingerprint extraction
# ---------------------------------------------------------------------------

def extract_fingerprint(trajectories, agent_name):
    """
    Extract a 17-dimensional strategy fingerprint from trajectory data.

    Args:
        trajectories: list of dicts from collect_trajectories()
        agent_name: str (for labeling)

    Returns:
        dict with fingerprint values and metadata
    """
    from analysis.cognitive_hierarchy import p_bid_valid

    # --- 1. Challenge rate by p_valid quintile (5 features) ---
    quintile_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    challenge_by_quintile = [[] for _ in range(5)]

    # --- 2. Bid quantity increments (for aggression / jump stats) ---
    qty_increments = []  # bid_qty - min_valid_qty at that decision

    # --- 3. Face preference (6 features) ---
    face_counts = np.zeros(7)  # index 1..6

    # --- 4. Positional challenge rates ---
    challenge_p0 = []
    challenge_p1 = []

    # --- 5. Bluff tracking ---
    bluff_decisions = []  # 1 = bluff bid, 0 = honest bid

    for t in trajectories:
        pv = t.get("p_valid", None)
        action = t["action"]
        player_pos = t.get("player_position", 0)
        action_mask = t["action_mask"]
        own_dice = t["own_dice"]

        is_challenge = (action == 0)

        # Challenge calibration
        if pv is not None:
            q_idx = min(int(pv * 5), 4)
            challenge_by_quintile[q_idx].append(float(is_challenge))

        # Positional bias
        if player_pos == 0:
            challenge_p0.append(float(is_challenge))
        else:
            challenge_p1.append(float(is_challenge))

        # Bid analysis (only when action is a bid, not challenge)
        if not is_challenge and action > 0:
            # Decode face from action index: action = (q-1)*6 + (f-1) + 1
            # So f = ((action-1) % 6) + 1, q = (action-1) // 6 + 1
            bid_action = action - 1  # 0-indexed
            face = (bid_action % 6) + 1
            qty = bid_action // 6 + 1

            face_counts[face] += 1

            # Minimum valid bid quantity for this face
            current_bid = t.get("current_bid")
            if current_bid is not None:
                min_q = current_bid[0] + (1 if face > current_bid[1] else
                                           (0 if face == current_bid[1] else 0))
                # Strictly: new bid must be (q+1,any) or (same_q, f+1..6)
                # Simpler: min valid qty for same face is current_q+1; for higher face is current_q
                if face > current_bid[1]:
                    min_q = current_bid[0]
                else:
                    min_q = current_bid[0] + 1
                qty_increments.append(max(0, qty - min_q))

            # Bluff detection: bid qty > expected dice of that face (own + expected opp)
            own_count = int(np.sum(own_dice == face))
            n_opp = t.get("n_opp_dice", 5)
            expected_total = own_count + n_opp / 6.0
            bluff_decisions.append(float(qty > expected_total + 0.5))

    # --- Build fingerprint vector ---

    # 1. Challenge rates by p_valid quintile (5 features)
    cal_features = []
    for q_vals in challenge_by_quintile:
        cal_features.append(np.mean(q_vals) if q_vals else np.nan)

    # 2. Bid aggression: mean increment over minimum qty (1 feature)
    mean_increment = np.mean(qty_increments) if qty_increments else 0.0

    # 3. Face preference (6 features, normalized)
    face_total = face_counts[1:].sum()
    face_pref = face_counts[1:] / face_total if face_total > 0 else np.ones(6) / 6

    # 4. Positional bias: challenge rate diff p0 - p1 (1 feature)
    cr_p0 = np.mean(challenge_p0) if challenge_p0 else 0.0
    cr_p1 = np.mean(challenge_p1) if challenge_p1 else 0.0
    positional_bias = cr_p0 - cr_p1

    # 5. Bluff rate (1 feature)
    bluff_rate = np.mean(bluff_decisions) if bluff_decisions else 0.0

    # 6. Bid jump consistency: std of increments (1 feature)
    jump_std = np.std(qty_increments) if len(qty_increments) > 1 else 0.0

    # 7. Overall challenge rate (1 feature)
    all_challenges = [t["action"] == 0 for t in trajectories]
    overall_cr = np.mean(all_challenges) if all_challenges else 0.0

    fingerprint_vec = np.array(
        cal_features +           # 5
        [mean_increment] +       # 1
        face_pref.tolist() +     # 6
        [positional_bias] +      # 1
        [bluff_rate] +           # 1
        [jump_std] +             # 1
        [overall_cr]             # 1
        # Total: 16 features (was 17 in plan; close enough)
    )

    return {
        "agent": agent_name,
        "fingerprint": fingerprint_vec.tolist(),
        "n_trajectories": len(trajectories),
        "calibration_by_quintile": cal_features,
        "mean_bid_increment": float(mean_increment),
        "face_preference": face_pref.tolist(),
        "positional_bias": float(positional_bias),
        "bluff_rate": float(bluff_rate),
        "jump_std": float(jump_std),
        "overall_challenge_rate": float(overall_cr),
    }


def pca_agents(fingerprints_dict, n_components=2):
    """
    Run PCA on agent fingerprints.

    Args:
        fingerprints_dict: dict {agent_name: fingerprint_dict}

    Returns:
        dict with PCA results and explained variance
    """
    agent_names = list(fingerprints_dict.keys())
    X = np.array([fingerprints_dict[a]["fingerprint"] for a in agent_names])

    # Handle NaN: replace with column mean; if whole column is NaN use 0
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    return {
        "agent_names": agent_names,
        "pca_coords": X_pca.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "loadings": pca.components_.tolist(),
        "feature_names": [
            "cal_q0", "cal_q1", "cal_q2", "cal_q3", "cal_q4",
            "mean_increment",
            "face_1", "face_2", "face_3", "face_4", "face_5", "face_6",
            "positional_bias",
            "bluff_rate",
            "jump_std",
            "overall_cr",
        ],
    }


def save_fingerprints(fingerprints_dict, pca_result, output_path):
    """Save fingerprint analysis results to JSON."""
    result = {
        "fingerprints": {
            a: {k: v for k, v in fp.items() if k != "fingerprint"}
            for a, fp in fingerprints_dict.items()
        },
        "fingerprint_vectors": {
            a: fp["fingerprint"] for a, fp in fingerprints_dict.items()
        },
        "pca": pca_result,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Fingerprints saved to {output_path}")

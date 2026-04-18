"""
analysis/cognitive_hierarchy.py

Cognitive Hierarchy (CH) model for Liar's Dice, following Camerer et al. (QJE 2004).
Level-k is applied at the single-decision level: each bid/challenge decision is treated
quasi-independently (memoryless Level-k, consistent with the CH framing for imperfect-info games).

Level-0: Uniform over all valid actions.
Level-1: Challenge when P(bid valid | uniform opponent dice) < t1=0.25; else minimum legal bid.
         This mirrors HeuristicAgent exactly.
Level-2: Challenge when P(bid valid | particle-filter posterior) < t2=0.30; else minimum legal bid.
         Approximated by a simplified Bayesian update using observed bid history.

Poisson CH model: P(agent plays level-k) = Poisson(k | tau).
We fit tau per agent by maximum likelihood over collected (state, action) trajectories.
"""

import numpy as np
from scipy.stats import binom, poisson


# ---------------------------------------------------------------------------
# Core probability computation (shared with HeuristicAgent)
# ---------------------------------------------------------------------------

def p_bid_valid(own_dice, current_bid, n_opp_dice):
    """
    P(current bid is valid | own dice, assuming opponent dice are i.i.d. Uniform{1..6}).

    'Valid' means the total count of face `f` across ALL dice is >= quantity `q`.

    Args:
        own_dice: np.ndarray of shape (n_dice,) with face values 1-6
        current_bid: tuple (quantity, face) — the bid to evaluate
        n_opp_dice: int — number of opponent dice remaining

    Returns:
        float in [0, 1]
    """
    if current_bid is None:
        return 1.0  # No bid yet → not relevant

    q, f = current_bid
    own_count = int(np.sum(own_dice == f))
    opp_needed = max(0, q - own_count)

    if opp_needed == 0:
        return 1.0  # Already satisfied by own dice

    if n_opp_dice == 0:
        return 0.0  # No opponent dice, can't satisfy

    # P(Binomial(n_opp_dice, 1/6) >= opp_needed) = binom.sf(opp_needed-1, n_opp_dice, 1/6)
    return float(binom.sf(opp_needed - 1, n_opp_dice, 1.0 / 6.0))


# ---------------------------------------------------------------------------
# Level-k action distributions
# ---------------------------------------------------------------------------

def _uniform_dist(action_mask):
    """Level-0: uniform over valid actions."""
    dist = np.array(action_mask, dtype=float)
    total = dist.sum()
    if total == 0:
        return dist
    return dist / total


def _level1_dist(own_dice, current_bid, n_opp_dice, action_mask,
                 t_challenge=0.25, p_bluff=0.0):
    """
    Level-1: best-respond to Level-0 opponent.
    Challenge if P(bid valid | uniform dice) < t_challenge.
    Otherwise bid: prefer minimum valid bid (weight decays with bid index).

    p_bluff: probability of making a bluff bid (high bid); set to 0 for pure Level-1.
    """
    dist = np.zeros(len(action_mask), dtype=float)

    if current_bid is None:
        # No bid yet — must bid (challenge is invalid); uniform over valid bids
        valid_bids = np.array(action_mask, dtype=float)
        valid_bids[0] = 0.0  # index 0 is challenge, invalid here
        if valid_bids.sum() == 0:
            return _uniform_dist(action_mask)
        return valid_bids / valid_bids.sum()

    pv = p_bid_valid(own_dice, current_bid, n_opp_dice)

    if pv < t_challenge and action_mask[0] == 1:
        # Challenge deterministically
        dist[0] = 1.0
        return dist

    # Bid: weight valid bids by 1/(rank+1) so minimum bid is preferred
    valid_bid_indices = [i for i in range(1, len(action_mask)) if action_mask[i] == 1]
    if not valid_bid_indices:
        # No valid bids → must challenge
        dist[0] = 1.0
        return dist

    weights = np.array([1.0 / (rank + 1) for rank, _ in enumerate(valid_bid_indices)])
    weights /= weights.sum()
    for idx, bid_i in enumerate(valid_bid_indices):
        dist[bid_i] = weights[idx]
    return dist


def _level2_dist(own_dice, current_bid, n_opp_dice, action_mask,
                 bid_history=None, t_challenge=0.30):
    """
    Level-2: best-respond to Level-1 opponent.
    Uses a soft Bayesian posterior over opponent dice (simplified update).
    Approximated by adjusting p_valid based on observed bids:
      - If opponent has been bidding aggressively (high quantities), posterior shifts upward.
      - Otherwise, same as Level-1 but with a slightly different threshold.

    For simplicity (tractable closed-form): uses the same binomial p_valid as Level-1
    but with posterior-adjusted n_opp_dice_effective and t_challenge=0.30.
    Full particle-filter posterior is too expensive to compute per trajectory point here.
    """
    # Approximate: same structure as Level-1 but with posterior-corrected n_opp_dice
    # The correction: observed aggressive bids → assume opponent has more matching dice
    n_opp_eff = n_opp_dice
    if bid_history and current_bid is not None:
        q, f = current_bid
        # Count how many bids in history referenced this face
        face_refs = sum(1 for e in bid_history
                        if e.get("type") == "bid" and e.get("face") == f)
        # Soft adjustment: each face reference adds ~0.2 expected dice
        n_opp_eff = min(n_opp_dice + 0.2 * face_refs, n_opp_dice * 1.5)

    pv = p_bid_valid(own_dice, current_bid, n_opp_eff) if current_bid is not None else 1.0

    dist = np.zeros(len(action_mask), dtype=float)

    if current_bid is None:
        valid_bids = np.array(action_mask, dtype=float)
        valid_bids[0] = 0.0
        if valid_bids.sum() == 0:
            return _uniform_dist(action_mask)
        return valid_bids / valid_bids.sum()

    if pv < t_challenge and action_mask[0] == 1:
        dist[0] = 1.0
        return dist

    valid_bid_indices = [i for i in range(1, len(action_mask)) if action_mask[i] == 1]
    if not valid_bid_indices:
        dist[0] = 1.0
        return dist

    weights = np.array([1.0 / (rank + 1) for rank, _ in enumerate(valid_bid_indices)])
    weights /= weights.sum()
    for idx, bid_i in enumerate(valid_bid_indices):
        dist[bid_i] = weights[idx]
    return dist


def level_k_action_dist(k, own_dice, current_bid, n_opp_dice, action_mask,
                        bid_history=None):
    """
    Compute the Level-k action probability vector.

    Args:
        k: int in {0, 1, 2}
        own_dice: np.ndarray shape (n_dice,)
        current_bid: tuple (q, f) or None
        n_opp_dice: int
        action_mask: np.ndarray of 0/1, shape (n_actions,)
        bid_history: list of bid event dicts (used for k=2)

    Returns:
        np.ndarray of shape (n_actions,), sums to 1
    """
    if k == 0:
        return _uniform_dist(action_mask)
    elif k == 1:
        return _level1_dist(own_dice, current_bid, n_opp_dice, action_mask)
    elif k == 2:
        return _level2_dist(own_dice, current_bid, n_opp_dice, action_mask,
                            bid_history=bid_history)
    else:
        # Level-3+: no closed-form; fall back to Level-2 with tighter threshold
        return _level2_dist(own_dice, current_bid, n_opp_dice, action_mask,
                            bid_history=bid_history, t_challenge=0.35)


# ---------------------------------------------------------------------------
# Poisson CH model and tau fitting
# ---------------------------------------------------------------------------

def ch_action_dist(tau, own_dice, current_bid, n_opp_dice, action_mask,
                   bid_history=None, K=3, eps=1e-8):
    """
    CH action distribution: mixture of Level-k distributions weighted by Poisson(tau).

    CH(a | state, tau) = sum_{k=0}^{K} Poisson(k | tau) * Level-k(a | state)

    Returns: np.ndarray of shape (n_actions,), sums to 1
    """
    weights = np.array([poisson.pmf(k, tau) for k in range(K + 1)])
    weights /= weights.sum()  # Renormalize truncated Poisson

    dist = np.zeros(len(action_mask), dtype=float)
    for k, w in enumerate(weights):
        if w < 1e-10:
            continue
        lk = level_k_action_dist(k, own_dice, current_bid, n_opp_dice,
                                  action_mask, bid_history=bid_history)
        dist += w * lk

    # Add small epsilon for numerical stability in log
    dist = dist + eps
    dist /= dist.sum()
    return dist


def fit_tau(trajectories, tau_grid=None, K=3, eps=1e-8):
    """
    Fit Poisson tau for one agent by maximum likelihood over observed trajectories.

    Args:
        trajectories: list of dicts, each with:
            {
                "own_dice": np.ndarray,
                "current_bid": tuple or None,
                "n_opp_dice": int,
                "action": int,
                "action_mask": np.ndarray,
                "bid_history": list (optional),
            }
        tau_grid: list of tau values to try (default: fine grid from 0.1 to 5)
        K: max level to include in mixture

    Returns:
        dict with keys:
            "tau": best tau (float)
            "log_likelihoods": dict {tau: avg_log_likelihood}
            "n_observations": int
    """
    if tau_grid is None:
        tau_grid = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5,
                    2.0, 2.5, 3.0, 4.0, 5.0]

    log_likelihoods = {}

    for tau in tau_grid:
        total_ll = 0.0
        n = 0
        for t in trajectories:
            own_dice = t["own_dice"]
            current_bid = t.get("current_bid")
            n_opp_dice = t["n_opp_dice"]
            action = t["action"]
            action_mask = t["action_mask"]
            bid_history = t.get("bid_history", None)

            if action_mask[action] == 0:
                continue  # Invalid action recorded — skip

            ch_dist = ch_action_dist(tau, own_dice, current_bid, n_opp_dice,
                                     action_mask, bid_history=bid_history, K=K, eps=eps)
            total_ll += np.log(ch_dist[action] + eps)
            n += 1

        log_likelihoods[tau] = total_ll / n if n > 0 else float("-inf")

    best_tau = max(log_likelihoods, key=log_likelihoods.get)
    return {
        "tau": best_tau,
        "log_likelihoods": log_likelihoods,
        "n_observations": len(trajectories),
    }


def bootstrap_tau_ci(trajectories, tau_grid=None, n_bootstrap=200, ci=0.95):
    """
    Bootstrap 95% CI for tau. Resamples trajectories with replacement.

    Returns: (tau_mean, tau_lower, tau_upper)
    """
    bootstrap_taus = []
    n = len(trajectories)
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [trajectories[i] for i in idx]
        result = fit_tau(sample, tau_grid=tau_grid)
        bootstrap_taus.append(result["tau"])

    alpha = 1 - ci
    lower = np.percentile(bootstrap_taus, 100 * alpha / 2)
    upper = np.percentile(bootstrap_taus, 100 * (1 - alpha / 2))
    mean_tau = np.mean(bootstrap_taus)
    return float(mean_tau), float(lower), float(upper)


# ---------------------------------------------------------------------------
# KL divergence computation
# ---------------------------------------------------------------------------

def state_kl_divergences(trajectories, k, eps=1e-8):
    """
    For each trajectory point, compute KL(empirical_agent || Level-k).

    Since we only have one action per state (not a full distribution), we approximate
    the agent's distribution as a point mass on the observed action and compute:
        KL = -log Level-k(observed_action | state)
    (This is the per-sample log-loss against Level-k.)

    Returns: np.ndarray of shape (n_trajectories,) — per-point log-loss
    """
    log_losses = []
    for t in trajectories:
        action = t["action"]
        action_mask = t["action_mask"]
        if action_mask[action] == 0:
            continue

        lk = level_k_action_dist(
            k,
            t["own_dice"],
            t.get("current_bid"),
            t["n_opp_dice"],
            action_mask,
            bid_history=t.get("bid_history"),
        )
        log_losses.append(-np.log(lk[action] + eps))

    return np.array(log_losses)


def compute_kl_table(trajectories_per_agent, K=3):
    """
    Compute mean KL (log-loss) for each agent × level-k combination.

    Args:
        trajectories_per_agent: dict {agent_name: list of trajectory dicts}
        K: max level

    Returns:
        dict {agent_name: {k: mean_log_loss}}
    """
    result = {}
    for agent_name, trajs in trajectories_per_agent.items():
        result[agent_name] = {}
        for k in range(K + 1):
            kls = state_kl_divergences(trajs, k)
            result[agent_name][k] = float(np.mean(kls)) if len(kls) > 0 else float("nan")
    return result


# ---------------------------------------------------------------------------
# Challenge calibration binning
# ---------------------------------------------------------------------------

def compute_challenge_calibration(trajectories, n_bins=20):
    """
    Bin trajectories by p_valid and compute challenge rate per bin.

    Args:
        trajectories: list of dicts with "p_valid" (float) and "action" (int, 0=challenge)

    Returns:
        dict with:
            "bin_centers": np.ndarray
            "challenge_rates": np.ndarray
            "bin_counts": np.ndarray
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    challenge_counts = np.zeros(n_bins)
    total_counts = np.zeros(n_bins)

    for t in trajectories:
        pv = t.get("p_valid")
        if pv is None:
            continue
        action = t["action"]
        bin_idx = min(int(pv * n_bins), n_bins - 1)
        total_counts[bin_idx] += 1
        if action == 0:  # 0 = challenge
            challenge_counts[bin_idx] += 1

    challenge_rates = np.where(total_counts > 0, challenge_counts / total_counts, np.nan)
    return {
        "bin_centers": bin_centers,
        "challenge_rates": challenge_rates,
        "bin_counts": total_counts,
    }


# ---------------------------------------------------------------------------
# Vectorized tau fitting (fast — precomputes level-k probs once)
# ---------------------------------------------------------------------------

def _precompute_levelk_probs(trajectories, K=3, eps=1e-8):
    """
    For each trajectory point, precompute Level-k(observed_action | state) for k=0..K.

    Returns:
        np.ndarray of shape (N, K+1) — log probability of observed action under each level
    """
    N = len(trajectories)
    log_probs = np.full((N, K + 1), np.nan)

    for i, t in enumerate(trajectories):
        action = t["action"]
        action_mask = t["action_mask"]
        if action_mask[action] == 0:
            continue
        for k in range(K + 1):
            lk = level_k_action_dist(
                k,
                t["own_dice"],
                t.get("current_bid"),
                t["n_opp_dice"],
                action_mask,
                bid_history=t.get("bid_history"),
            )
            log_probs[i, k] = np.log(lk[action] + eps)

    return log_probs


def fit_tau_fast(trajectories, tau_grid=None, K=3, eps=1e-8):
    """
    Fast vectorized tau fitting.  Precomputes level-k action probs once,
    then sweeps over tau_grid with pure numpy (no Python loops per point).

    Returns same format as fit_tau().
    """
    if tau_grid is None:
        tau_grid = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5,
                    2.0, 2.5, 3.0, 4.0, 5.0]

    # Precompute: lk_probs[i, k] = Level-k P(action_i | state_i)
    lk_log_probs = _precompute_levelk_probs(trajectories, K=K, eps=eps)
    valid = ~np.isnan(lk_log_probs[:, 0])
    lk_log_probs = lk_log_probs[valid]  # shape (N_valid, K+1)
    # Convert to probability domain for mixture computation
    lk_probs = np.exp(lk_log_probs)     # shape (N_valid, K+1)

    n_valid = lk_probs.shape[0]
    log_likelihoods = {}

    for tau in tau_grid:
        # Poisson weights
        weights = np.array([poisson.pmf(k, max(tau, 1e-9)) for k in range(K + 1)])
        weights /= weights.sum()  # renormalize truncated Poisson

        # Mixture probability for each point: dot product of weights and level-k probs
        mixture = lk_probs @ weights  # shape (N_valid,)
        ll = np.sum(np.log(mixture + eps))
        log_likelihoods[tau] = float(ll / n_valid) if n_valid > 0 else float("-inf")

    best_tau = max(log_likelihoods, key=log_likelihoods.get)
    return {
        "tau": best_tau,
        "log_likelihoods": log_likelihoods,
        "n_observations": n_valid,
    }


def bootstrap_tau_ci_fast(trajectories, tau_grid=None, n_bootstrap=200, ci=0.95, K=3, eps=1e-8):
    """
    Fast bootstrap CI for tau.  Precomputes level-k probs once; resampling is instant.
    """
    if tau_grid is None:
        tau_grid = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5,
                    2.0, 2.5, 3.0, 4.0, 5.0]

    # Precompute once
    lk_log_probs = _precompute_levelk_probs(trajectories, K=K, eps=eps)
    valid = ~np.isnan(lk_log_probs[:, 0])
    lk_probs = np.exp(lk_log_probs[valid])  # (N_valid, K+1)
    n_valid = lk_probs.shape[0]

    # Precompute weight vectors for all tau values
    tau_weights = {}
    for tau in tau_grid:
        w = np.array([poisson.pmf(k, max(tau, 1e-9)) for k in range(K + 1)])
        tau_weights[tau] = w / w.sum()

    rng = np.random.default_rng(42)
    bootstrap_taus = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_valid, size=n_valid)
        sample_probs = lk_probs[idx]  # (N_valid, K+1)

        best_ll = -np.inf
        best_tau = tau_grid[0]
        for tau in tau_grid:
            mixture = sample_probs @ tau_weights[tau]
            ll = float(np.sum(np.log(mixture + eps)))
            if ll > best_ll:
                best_ll = ll
                best_tau = tau
        bootstrap_taus.append(best_tau)

    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_taus, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_taus, 100 * (1 - alpha / 2)))
    mean_tau = float(np.mean(bootstrap_taus))
    return mean_tau, lower, upper


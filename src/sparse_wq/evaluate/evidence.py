"""Statistical support for Skill and Prediction at small sample sizes (Section 2.5, Table 4).

Assessment is conditional on the selected fitted models and scored laboratory
observations. Sample counts depend on target and coverage. Intervals and tests do
not propagate training or selection uncertainty; the sign test assumes independent
non-tied pairs and the alignment test assumes exchangeable observations.

Skill (model against the lowest-RMSE reference):
    skill           1 - RMSE_model / RMSE_reference
    interval        moving-block bootstrap over measurements, 5th-95th percentile
    test            exact two-sided sign test on the per-measurement loss differences
Prediction (model against the evaluation-period mean):
    R²              with a bootstrap interval over measurements
    test            one-sided permutation test of the pairing of predictions with measurements
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

NOT_SUPPORTED = "not_supported"
UNDERPOWERED = "underpowered"   # "Untestable" in Table 4
DIRECTIONAL = "directional"
SUPPORTED = "supported"
VERDICT_ORDER = (NOT_SUPPORTED, UNDERPOWERED, DIRECTIONAL, SUPPORTED)


def min_attainable_two_sided_p(n: int) -> float:
    """Smallest two-sided exact binomial p-value with `n` pairs: 2 * 0.5**n (NaN for n < 1)."""
    n = int(n)
    return float("nan") if n < 1 else min(1.0, 2.0 * (0.5 ** n))


def power_is_attainable(n: int, alpha: float = 0.05) -> bool:
    """Whether `n` pairs could reach `alpha` at all (n >= 6 for alpha = 0.05)."""
    p_min = min_attainable_two_sided_p(n)
    return bool(np.isfinite(p_min) and p_min < float(alpha))


def exact_sign_test(diff: np.ndarray) -> dict:
    """Exact two-sided sign test on model-minus-reference losses (negative = model wins); ties dropped."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    wins, losses = int(np.sum(d < 0)), int(np.sum(d > 0))
    n = wins + losses
    out = {"wins": wins, "losses": losses, "n_pairs": n,
           "win_rate": float(wins / n) if n else float("nan"),
           "p_value": float("nan"),
           "min_attainable_p": min_attainable_two_sided_p(n) if n else float("nan"),
           "direction": ("favours_model" if wins > losses else "favours_reference" if losses > wins else "tied")}
    if n >= 1:
        out["p_value"] = float(stats.binomtest(wins, n=n, p=0.5, alternative="two-sided").pvalue)
    return out


def dm_test(diff: np.ndarray, max_lag: int = 1, min_n: int = 30) -> dict:
    """Diebold-Mariano with the Harvey-Leybourne-Newbold correction; diagnostic only (not in any verdict)."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    out = {"stat": float("nan"), "p_value": float("nan"), "n": n, "corrected": False}
    if n < max(5, int(min_n)):
        return out
    centered = d - float(np.mean(d))
    var = float(np.dot(centered, centered) / n)
    h = int(max(0, min(int(max_lag), n - 1)))
    for lag in range(1, h + 1):
        var += 2.0 * (1.0 - lag / (h + 1.0)) * float(np.dot(centered[lag:], centered[:-lag]) / n)
    if not np.isfinite(var) or var <= 0:
        return out
    stat = float(np.mean(d)) / math.sqrt(var / n)
    steps = h + 1
    adj = n + 1.0 - 2.0 * steps + (steps * (steps - 1.0)) / n
    if adj <= 0:
        return out
    stat *= math.sqrt(adj / n)
    out.update(stat=stat, p_value=float(2.0 * stats.t.sf(abs(stat), df=n - 1)), corrected=True)
    return out


def grouped_skill_bootstrap(losses_model, losses_ref, group_ids, n_boot: int = 2000, seed: int = 42,
                            block_len: int = 3, moving_block: bool = True) -> dict:
    """Skill (1 - RMSE ratio) with a percentile bootstrap that resamples whole groups.

    The block length is limited to a quarter of the number of groups (and below the group
    count), so small samples are resampled independently rather than as near-rotations.
    A zero-width interval is reported as no interval (`degenerate`).
    """
    lm = np.asarray(losses_model, dtype=float).reshape(-1)
    lr = np.asarray(losses_ref, dtype=float).reshape(-1)
    gids = list(group_ids)
    n = min(lm.size, lr.size, len(gids))
    out = {"skill": float("nan"), "ci05": float("nan"), "ci95": float("nan"), "prob_skill_gt0": float("nan"),
           "n_groups": 0, "n_boot_ok": 0, "block_len_used": int(max(1, block_len)),
           "block_len_clamped": False, "degenerate": False}
    if n < 1:
        return out
    lm, lr, gids = lm[:n], lr[:n], gids[:n]
    finite = np.isfinite(lm) & np.isfinite(lr)
    if not np.any(finite):
        return out

    order: list = []
    rows_of: dict = {}
    for i in range(n):
        if finite[i]:
            if gids[i] not in rows_of:
                rows_of[gids[i]] = []
                order.append(gids[i])
            rows_of[gids[i]].append(i)
    n_groups = len(order)
    out["n_groups"] = n_groups
    if n_groups < 1:
        return out

    def skill(idx) -> float:
        rmse_m, rmse_r = math.sqrt(float(np.mean(lm[idx]))), math.sqrt(float(np.mean(lr[idx])))
        if not np.isfinite(rmse_m) or not np.isfinite(rmse_r) or rmse_r <= 0:
            return float("nan")
        return 1.0 - rmse_m / rmse_r

    out["skill"] = skill([i for g in order for i in rows_of[g]])
    if n_groups < 3:
        return out

    block = int(max(1, min(int(block_len), n_groups // 4, n_groups - 1)))
    out["block_len_used"] = block
    out["block_len_clamped"] = block != int(max(1, block_len))

    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(max(1, n_boot))):
        if moving_block and block > 1:
            chosen = []
            while len(chosen) < n_groups:
                start = int(rng.integers(0, n_groups))
                for j in range(block):
                    chosen.append(order[(start + j) % n_groups])
                    if len(chosen) >= n_groups:
                        break
        else:
            chosen = [order[k] for k in rng.integers(0, n_groups, size=n_groups)]
        idx = [i for g in chosen for i in rows_of[g]]
        if idx:
            s = skill(idx)
            if np.isfinite(s):
                values.append(s)

    arr = np.asarray(values, dtype=float)
    out["n_boot_ok"] = int(arr.size)
    if arr.size:
        lo, hi = float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))
        if not np.isfinite(hi - lo) or (hi - lo) <= 1e-9 * max(abs(lo), abs(hi), 1e-12):
            out["degenerate"] = True
        else:
            out.update(ci05=lo, ci95=hi, prob_skill_gt0=float(np.mean(arr > 0.0)))
    return out


def _by_group(y, preds, group_ids):
    y = np.asarray(y, dtype=float).reshape(-1)
    preds = np.asarray(preds, dtype=float).reshape(-1)
    gids = list(group_ids)
    groups: dict = {}
    for i in range(min(y.size, preds.size, len(gids))):
        if np.isfinite(y[i]) and np.isfinite(preds[i]):
            groups.setdefault(gids[i], []).append(i)
    return y, preds, groups


def r2_bootstrap(y, preds, group_ids, *, n_boot: int = 2000, seed: int = 42) -> dict:
    """R² with a percentile bootstrap over whole groups (the denominator is recomputed each time)."""
    y, preds, groups = _by_group(y, preds, group_ids)
    keys = list(groups)
    if len(keys) < 3:
        return {"r2": float("nan"), "ci05": float("nan"), "ci95": float("nan"), "n_groups": len(keys)}

    def r2(idx) -> float:
        yy, pp = y[idx], preds[idx]
        denom = float(np.sum((yy - yy.mean()) ** 2))
        return float("nan") if denom <= 0 else float(1.0 - np.sum((pp - yy) ** 2) / denom)

    observed = r2(np.asarray([i for k in keys for i in groups[k]], dtype=int))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        v = r2(np.asarray([i for d in drawn for i in groups[keys[d]]], dtype=int))
        if np.isfinite(v):
            values.append(v)
    if len(values) < int(n_boot) // 4:
        return {"r2": observed, "ci05": float("nan"), "ci95": float("nan"), "n_groups": len(keys)}
    return {"r2": observed, "ci05": float(np.percentile(values, 5)), "ci95": float(np.percentile(values, 95)),
            "n_groups": len(keys)}


def association_permutation_p(y, preds, group_ids, *, n_perm: int = 5000, seed: int = 42) -> float:
    """One-sided permutation p-value that the predictions carry no information about the measurements."""
    y, preds, groups = _by_group(y, preds, group_ids)
    keys = list(groups)
    if len(keys) < 4:
        return float("nan")
    yg = np.asarray([float(np.mean(y[groups[k]])) for k in keys])
    pg = np.asarray([float(np.mean(preds[groups[k]])) for k in keys])
    denom = float(np.sum((yg - yg.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    observed = float(1.0 - np.sum((pg - yg) ** 2) / denom)
    rng = np.random.default_rng(seed)
    hits = sum(1 for _ in range(int(n_perm))
               if float(1.0 - np.sum((rng.permutation(pg) - yg) ** 2) / denom) >= observed)
    return float((hits + 1) / (int(n_perm) + 1))  # add-one: p = 0 is not attainable


def classify_prediction(r2: float, ci05: float, perm_p: float, n_groups: int, alpha: float = 0.05) -> str:
    """Prediction verdict (Table 4, right column)."""
    if not np.isfinite(r2) or r2 <= 0:
        return NOT_SUPPORTED
    if not np.isfinite(n_groups) or int(n_groups) < 5:
        return UNDERPOWERED
    significant = np.isfinite(perm_p) and perm_p < float(alpha) and np.isfinite(ci05) and ci05 > 0.0
    return SUPPORTED if significant else DIRECTIONAL


def assess_prediction(y, preds, group_ids, *, alpha: float = 0.05, n_boot: int = 2000, n_perm: int = 5000,
                      seed: int = 42) -> dict:
    boot = r2_bootstrap(y, preds, group_ids, n_boot=n_boot, seed=seed)
    perm_p = association_permutation_p(y, preds, group_ids, n_perm=n_perm, seed=seed)
    return {"r2": boot["r2"], "r2_ci05": boot["ci05"], "r2_ci95": boot["ci95"], "perm_p": perm_p,
            "n_groups": boot["n_groups"],
            "verdict": classify_prediction(boot["r2"], boot["ci05"], perm_p, boot["n_groups"], alpha=alpha)}


def classify(skill: float, ci05: float, win_rate: float, sign_p: float, n_pairs: int, alpha: float = 0.05) -> str:
    """Skill verdict (Table 4, left column)."""
    if not np.isfinite(skill) or skill <= 0:
        return NOT_SUPPORTED
    if not np.isfinite(win_rate) or win_rate <= 0.5:
        return NOT_SUPPORTED
    if not power_is_attainable(n_pairs, alpha):
        return UNDERPOWERED
    significant = np.isfinite(sign_p) and sign_p < float(alpha) and np.isfinite(ci05) and ci05 > 0.0
    return SUPPORTED if significant else DIRECTIONAL


def assess(losses_model, losses_ref, group_ids, *, alpha: float = 0.05, n_boot: int = 2000, seed: int = 42,
           block_len: int = 3, moving_block: bool = True, dm_max_lag: int = 1) -> dict:
    """Skill of a model against one reference on a common, aligned evaluation set.

    `losses_*` are per-observation squared errors; `group_ids` identifies the independent unit.
    """
    boot = grouped_skill_bootstrap(losses_model, losses_ref, group_ids, n_boot=n_boot, seed=seed,
                                   block_len=block_len, moving_block=moving_block)
    lm = np.asarray(losses_model, dtype=float).reshape(-1)
    lr = np.asarray(losses_ref, dtype=float).reshape(-1)
    gids = list(group_ids)
    per_group: dict = {}
    for i in range(min(lm.size, lr.size, len(gids))):
        if np.isfinite(lm[i]) and np.isfinite(lr[i]):
            per_group.setdefault(gids[i], []).append(lm[i] - lr[i])
    diff = np.asarray([float(np.mean(v)) for v in per_group.values()], dtype=float)
    sign = exact_sign_test(diff)
    dm = dm_test(diff, max_lag=dm_max_lag)
    return {"skill": boot["skill"], "skill_ci05": boot["ci05"], "skill_ci95": boot["ci95"],
            "prob_skill_gt0": boot["prob_skill_gt0"], "n_groups": boot["n_groups"], "n_boot_ok": boot["n_boot_ok"],
            "block_len_used": boot["block_len_used"], "block_len_clamped": boot["block_len_clamped"],
            "bootstrap_degenerate": boot["degenerate"], "sign_direction": sign["direction"],
            "sign_wins": sign["wins"], "sign_losses": sign["losses"], "sign_n_pairs": sign["n_pairs"],
            "sign_win_rate": sign["win_rate"], "sign_p": sign["p_value"],
            "min_attainable_p": sign["min_attainable_p"], "power_attainable": power_is_attainable(sign["n_pairs"], alpha),
            "dm_stat": dm["stat"], "dm_p": dm["p_value"], "dm_applicable": dm["corrected"],
            "verdict": classify(boot["skill"], boot["ci05"], sign["win_rate"], sign["p_value"], sign["n_pairs"],
                                alpha=alpha),
            "alpha": float(alpha)}

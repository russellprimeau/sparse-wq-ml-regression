"""Multiple linear regression with training-only predictor screening.

Each measured window is encoded as latest inputs, preceding 12-hour means or
full-window means. Spearman screening is followed by constant removal, mutual
information screening, duplicate removal and variance-inflation screening.
Six screening seeds choose the most frequent set, with size/lexical tie breaks.
The fitting workflow handles missing rows and intercept-only predictions.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from sparse_wq.models.samples import Window

AGGREGATION = {"mlr": "last", "mlr_avg12": "avg12", "mlr_avgall": "avgall"}
SELECTION_SEEDS = 6
MAX_VIF_ROUNDS = 500


def predictor_row(block: np.ndarray, mode: str, columns: list[str]) -> np.ndarray:
    """One window as one row: last hour, 12-hour mean or window mean (prior value from the first row)."""
    arr = np.asarray(block, dtype=float)
    if mode == "last":
        return arr[-1, :].astype(float, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "avg12":
            return np.nanmean(arr[-12:, :], axis=0).astype(float, copy=False)
        row = np.nanmean(arr, axis=0).astype(float, copy=True)
    held = np.array([str(c).endswith("_state") for c in columns], dtype=bool)
    if held.any():
        row[held] = arr[0, held]
    return row


def spearman_screen(windows: list[Window], columns: list[str], mode: str,
                    p_max: float = 0.05, rho_min: float = 0.20) -> list[int]:
    """Columns whose Spearman correlation with the target has p < p_max and |rho| > rho_min;
    every column when fewer than five windows have a target, or when none passes."""
    rows, targets = [], []
    for w in windows:
        y = np.asarray(w.target, dtype=float).flatten()[0]
        if np.isfinite(y):
            rows.append(predictor_row(w.inputs, mode, columns))
            targets.append(y)
    if len(targets) < 5:
        return list(range(len(columns)))
    rows, targets = np.array(rows), np.array(targets)
    keep = []
    for c in range(len(columns)):
        ok = np.isfinite(rows[:, c]) & np.isfinite(targets)
        if ok.sum() < 5:
            continue
        rho, p = spearmanr(rows[ok, c], targets[ok])
        if float(p) < p_max and abs(float(rho)) > rho_min:
            keep.append(c)
    return keep or list(range(len(columns)))


def _complete(x, y):
    return np.all(np.isfinite(x), axis=1) & np.isfinite(y)


def _drop_constant(x, idx):
    return [c for i, c in enumerate(idx)
            if (col := x[:, i][np.isfinite(x[:, i])]).size >= 2 and np.std(col, ddof=1) > 0]


def _mutual_information(x, y, idx, quantile, seed):
    if len(idx) <= 1:
        return idx
    ok = _complete(x, y)
    if ok.sum() < 3:
        return idx
    mi = mutual_info_regression(x[ok], y[ok], random_state=seed)
    cut = np.quantile(mi, quantile)
    return [c for c, s in zip(idx, mi) if s >= cut] or idx




def _drop_duplicates(x, y, idx, threshold):
    keep = [True] * len(idx)
    has_y = np.isfinite(y)
    for i in range(len(idx)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(idx)):
            if not keep[j]:
                continue
            ok = np.isfinite(x[:, i]) & np.isfinite(x[:, j]) & has_y
            if np.count_nonzero(ok) < 3:
                continue
            xi, xj = x[ok, i], x[ok, j]
            if np.std(xi, ddof=1) <= 0 or np.std(xj, ddof=1) <= 0:
                keep[j] = False
                continue
            r = np.corrcoef(xi, xj)[0, 1]
            if not np.isfinite(r) or abs(r) < threshold:
                continue
            ri, rj = np.corrcoef(xi, y[ok])[0, 1], np.corrcoef(xj, y[ok])[0, 1]
            if (abs(ri) if np.isfinite(ri) else -np.inf) >= (abs(rj) if np.isfinite(rj) else -np.inf):
                keep[j] = False
            else:
                keep[i] = False
                break
    return [c for c, k in zip(idx, keep) if k]


def _vif(x):
    n, p = x.shape
    if p <= 1 or n < 3:
        return np.ones(p)
    out = np.full(p, np.inf)
    for j in range(p):
        others = np.column_stack([np.ones(n), np.delete(x, j, axis=1)])
        coef = np.linalg.lstsq(others, x[:, j], rcond=None)[0]
        ss_res = np.sum((x[:, j] - others @ coef) ** 2)
        ss_tot = np.sum((x[:, j] - np.mean(x[:, j])) ** 2)
        out[j] = np.inf if ss_tot == 0 else 1.0 / max(1.0 - (1.0 - ss_res / ss_tot), 1e-10)
    return out


def _variance_inflation(x, y, idx, threshold):
    if len(idx) <= 1:
        return idx
    ok = _complete(x, y)
    if ok.sum() < 3:
        return idx
    xv = x[ok].copy()
    local = list(range(len(idx)))
    for _ in range(MAX_VIF_ROUNDS):
        if len(local) <= 1:
            break
        v = _vif(xv[:, local])
        worst = int(np.argmax(v))
        if v[worst] <= threshold:
            break
        local.pop(worst)
    return [idx[i] for i in local]


def _select_once(x, y, seed, s: dict) -> list[int]:
    idx = _drop_constant(x, list(range(x.shape[1])))
    if idx and s["use_mutual_info"]:
        idx = _mutual_information(x[:, idx], y, idx, s["mi_quantile"], seed)
    if idx and s["deduplicate_threshold"] is not None and len(idx) > 1:
        idx = _drop_duplicates(x[:, idx], y, idx, float(s["deduplicate_threshold"]))
    if idx:
        idx = _variance_inflation(x[:, idx], y, idx, s["vif_threshold"])
    return idx


def selection_settings(recorded: dict) -> dict:
    """The screening settings a stored configuration was fitted with."""
    s = {"mi_quantile": 0.10, "vif_threshold": 10.0, "random_state": 0, "use_mutual_info": True,
         "deduplicate_threshold": 0.9999, "selection_seeds": 1, "use_spearman": True}
    method = str(recorded.get("method", ""))
    if method.startswith("Spearman"):        # full predictor list: library settings, six seeds
        s["selection_seeds"] = int(recorded["selection_seeds"])
    else:                                    # candidate subset or preselected set: as recorded
        s.update({k: v for k, v in recorded.items() if k not in ("method", "use_spearman_prefilter")})
        s["use_spearman"] = bool(recorded.get("use_spearman_prefilter", True))
    return s


def select(x: np.ndarray, y: np.ndarray, s: dict) -> list[int]:
    """The predictor set chosen most often over the selection seeds (column indices, ascending)."""
    first, n_seeds = int(s["random_state"]), int(s["selection_seeds"])
    if n_seeds <= 1:
        return _select_once(x, y, first, s)
    counts: dict[tuple, int] = {}
    for seed in range(first, first + n_seeds):
        key = tuple(sorted(_select_once(x, y, seed, s)))
        counts[key] = counts.get(key, 0) + 1
    return list(sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0])

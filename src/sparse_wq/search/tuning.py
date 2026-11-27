"""Search-space helpers for the chronological Optuna tuner in analysis.py."""
import numpy as np


def constrain_space(space: dict, n_eff: int) -> dict:
    """The search space narrowed to the number of training windows (see above)."""
    space ={k: dict(v) for k, v in space.items()}
    if "max_depth" in space:
        cap = int(np.floor(np.log2(max(1, n_eff)))) + 1
        if space["max_depth"].get("high", 0) > cap:
            space["max_depth"]["high"] = max(space["max_depth"].get("low", 2), cap)
    if "n_estimators" in space:
        cap = 10 * n_eff
        if space["n_estimators"].get("high", 0) > cap:
            space["n_estimators"]["high"] = max(space["n_estimators"].get("low", 50), cap)
    if "min_child_weight" in space:
        floor = int(np.ceil(n_eff / 50))
        if space["min_child_weight"].get("low", 999) < floor:
            space["min_child_weight"]["low"] = min(floor, space["min_child_weight"].get("high", floor))
    return space



def _suggest(trial, name: str, spec: dict):
    kind = str(spec.get("type", "float")).lower()
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "log":
        low = max(float(spec["low"]), 1e-12)
        return trial.suggest_float(name, low, max(float(spec["high"]), low * 1.000001), log=True)
    return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))



def _sanitise(p: dict) -> dict:
    p = dict(p)
    p["n_estimators"] = int(max(1, int(p["n_estimators"])))
    for k in ("subsample", "colsample_bytree"):
        p[k] = float(min(1.0, max(0.05, float(p[k]))))
    for k in ("reg_lambda", "reg_alpha", "gamma"):
        p[k] = float(max(0.0, float(p[k])))
    p["max_depth"] = int(max(1, int(p["max_depth"])))
    p["min_child_weight"] = float(max(0.0, float(p["min_child_weight"])))
    return p

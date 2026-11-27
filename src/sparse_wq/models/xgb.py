"""XGBoost fitting with chronologically tuned, fixed tree counts."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from sparse_wq.models.samples import Window

CONSTRAINT_FOLDS = 3


def runtime(hyper: dict, device: str) -> dict:
    """Hyperparameters with the device resolved: CUDA where requested and available."""
    out = dict(hyper)
    if str(device).lower().startswith("cuda") and cuda_available():
        out.setdefault("device", "cuda")
        out.setdefault("tree_method", "hist")
        out.pop("predictor", None)
    else:
        out.setdefault("tree_method", "hist")
        out.setdefault("n_jobs", -1)
        out.pop("device", None)
        if str(out.get("predictor", "")).strip().lower() == "gpu_predictor":
            out.pop("predictor", None)
    return out


def cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def estimator_arguments(hyper: dict) -> dict:
    args = {k: hyper[k] for k in ("tree_method", "objective", "n_estimators", "max_depth", "subsample",
                                  "colsample_bytree", "min_child_weight", "gamma", "reg_lambda",
                                  "reg_alpha", "learning_rate")}
    args["n_jobs"] = int(hyper.get("n_jobs", -1))
    if hyper.get("random_state") is not None:
        args["random_state"] = int(hyper["random_state"])
    return args


def constrain_to_sample_size(args: dict, n_train: int, n_folds: int = CONSTRAINT_FOLDS) -> dict:
    """Bound subsample, min_child_weight and max_depth by the smallest CV training fold."""
    args = dict(args)
    n_fold_train = int(n_train * (n_folds - 1) / n_folds)
    if n_fold_train < 1:
        return args
    subsample = float(args.get("subsample", 1.0))
    min_child_weight = float(args.get("min_child_weight", 1))
    min_subsample = min(1.0, max(20, 4 * min_child_weight) / max(1, n_fold_train))
    if subsample < min_subsample:
        subsample = round(min_subsample, 3)
        args["subsample"] = subsample
    per_tree = subsample * n_fold_train
    if per_tree <= 2 * min_child_weight:
        args["min_child_weight"] = max(1, int(per_tree / 4))
    depth_cap = int(np.floor(np.log2(max(1, n_fold_train)))) + 1
    if int(args.get("max_depth", 6)) > depth_cap:
        args["max_depth"] = max(1, depth_cap)
    return args


def arrays(windows: list[Window]) -> tuple[np.ndarray, np.ndarray]:
    """(inputs, target) as XGBoost receives them: float32, one scalar target per window."""
    if any(np.asarray(w.target).size != 1 for w in windows):
        raise ValueError("XGBoost is fitted to one target value per window")
    x = np.ascontiguousarray(np.array([w.inputs.flatten() for w in windows], dtype=np.float32))
    y = np.ascontiguousarray(np.array([w.target.flatten()[0] for w in windows], dtype=np.float32))
    return x, y






def fit(hyper: dict, train: list[Window], device: str, budget_splits=None) -> tuple[xgb.XGBRegressor, dict]:
    """Fit one configuration; returns the model and a record of how its size was set."""
    hyper = runtime(hyper, device)
    x, y = arrays(train)
    args = constrain_to_sample_size(estimator_arguments(hyper), n_train=len(train))
    record = {"n_estimators_tuned": int(args["n_estimators"]), "rounds_from": "configuration"}

    if hyper.get("early_stopping_rounds") is not None:
        raise ValueError("Tree counts must be selected by chronological tuning")
    record["n_estimators"] = int(args["n_estimators"])

    if "device" in hyper:
        args["device"] = hyper["device"]
    model = xgb.XGBRegressor(**args)
    model.fit(x, y, verbose=False)
    return model, record


def predict(model: xgb.XGBRegressor, windows: list[Window]) -> np.ndarray:
    """Predictions for windows, from the model as saved and reloaded for evaluation."""
    evaluator = xgb.XGBRegressor(tree_method="hist", **({"device": "cuda"} if cuda_available() else {}))
    evaluator.load_model(bytearray(model.get_booster().save_raw("json")))
    x = np.array([w.inputs.flatten() for w in windows])
    return evaluator.predict(x).reshape(-1, 1)

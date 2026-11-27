"""Ordinary exact Gaussian process with Matérn-5/2 and optional linear covariance.

Inputs and target are standardized on the training observations. The chronological
workflow selects the Adam step budget on independently prepared inner folds, then
fits all eligible training observations for that fixed budget. Prediction rebuilds
the posterior from the fitted parameters and training observations.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
from linear_operator import to_dense, to_linear_operator

from sparse_wq.models.samples import Window



# --------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------


class DenseSum(gpytorch.kernels.AdditiveKernel):
    """Sum of kernels formed as dense matrices.

    The default sum of a Matern and a linear kernel goes through a low-rank update that
    takes an SVD, which fails to converge on near-degenerate inputs; the dense sum is the
    same matrix.
    """

    def forward(self, x1, x2, diag=False, **params):
        out = None
        for kernel in self.kernels:
            term = kernel(x1, x2, diag=diag, **params)
            if not diag:
                term = to_dense(to_linear_operator(term))
            out = term if out is None else out + term
        return out


def make_kernel(name: str, noise_deltas, ard_dims):
    parts = [p.strip() for p in str(name or "matern52").lower().split("+") if p.strip()]
    if [p for p in parts if p != "linear"] != ["matern52"]:
        raise ValueError(f"unsupported kernel {name!r}")
    if noise_deltas is not None:raise ValueError('Only ordinary GP covariance is supported')
    base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
    if "linear" in parts:
        return DenseSum(base, gpytorch.kernels.LinearKernel(ard_num_dims=ard_dims))
    return base


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, kernel):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _interval(lo, hi):
    return None if lo is None and hi is None else gpytorch.constraints.Interval(float(lo), float(hi))


def build_model(x: torch.Tensor, y: torch.Tensor, hyper: dict, noise_deltas, ard_dims, device):
    """Exact GP and Gaussian likelihood with the configuration's kernel, bounds and priors."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGP(x, y, likelihood, make_kernel(hyper.get("kernel", "matern52"), noise_deltas, ard_dims)).to(device)
    lengthscale = _interval(hyper.get("lengthscale_min"), hyper.get("lengthscale_max"))
    if lengthscale is not None:
        seen = set()
        for k in model.covar_module.base_kernel.modules():
            if isinstance(k, gpytorch.kernels.Kernel) and id(k) not in seen:
                seen.add(id(k))
                if "raw_lengthscale" in k._parameters:
                    k.register_constraint("raw_lengthscale", lengthscale)
    noise = _interval(hyper.get("noise_min"), hyper.get("noise_max"))
    if noise is not None:
        likelihood.noise_covar.register_constraint("raw_noise", noise)
    for module, prior_key, name, param in ((model.covar_module, "outputscale_prior", "outputscale_prior", "outputscale"),
                                           (likelihood.noise_covar, "noise_prior", "noise_prior", "noise")):
        spec = hyper.get(prior_key)
        if spec:
            module.register_prior(name, gpytorch.priors.GammaPrior(float(spec["shape"]), float(spec["rate"])), param)
    return model, likelihood


# --------------------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------------------









# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------
@dataclass
class FittedGP:
    """A trained GP: its parameters and training standardisation."""

    hyper: dict
    model_state: dict
    likelihood_state: dict
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: float
    target_std: float
    ard_dims: int | None
    steps: int
    device_name: str


def resolve_device(requested: str) -> torch.device:
    return torch.device("cuda" if str(requested).lower().startswith("cuda") and torch.cuda.is_available() else "cpu")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _standardise(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std






def fit(hyper: dict, train: list[Window], input_columns: list[str], window_rows: int, aggregation: str,
        scaling: dict, offsets_csv, device_name: str, split_seed: int = 42, budget_splits=None,
        input_noise_deltas=None) -> FittedGP:
    """Fit for the step budget selected by chronological cross-validation."""
    for key, expected in (("early_stop_source", "cv"), ("early_stop_cv_folds", 3), ("cv_epoch_rule", "argmin"),
                          ("cv_epoch_smooth", 1), ("early_stop_cv_repeats", 1), ("early_stop_cv_scheme", "random")):
        if hyper.get(key, expected) != expected:
            raise ValueError(f"unsupported GP setting {key}={hyper[key]!r}")
    if not (hyper.get("input_standardize", True) and hyper.get("target_standardize", True)):
        raise ValueError("the reported GPs standardise inputs and target")
    _seed(int(hyper.get("random_state") or 0))
    device = resolve_device(device_name)
    x = np.array([w.inputs.flatten() for w in train], dtype=np.float32)
    y = np.array([w.target.flatten() for w in train], dtype=np.float32).reshape(len(train), -1)
    if y.shape[1] != 1:
        raise ValueError("one target value per window")
    max_train = hyper.get("max_train_size")
    if max_train is not None and len(x) > int(max_train):
        raise ValueError("training set larger than max_train_size; subsampling is not supported")
    x_mean, x_std = _standardise(x)
    x_used = (x - x_mean) / x_std

    uncertain = bool(hyper.get("use_uncertain_input_kernel", False))
    if uncertain or input_noise_deltas is not None:raise ValueError("Only ordinary GP covariance is supported")
    n_rows, n_dims = x.shape
    ard_budget = max(1, int(n_rows / max(float(hyper.get("ard_min_samples_per_dim", 1.0)), 1e-9)))
    use_ard = bool(hyper.get("ard", True)) and not n_dims > ard_budget
    ard_dims = n_dims if use_ard else None

    deltas = None
    if hyper.get('revision_fixed_steps') is None:
        raise ValueError('Supply the GP step budget selected on independently prepared chronological folds')
    steps = int(hyper['revision_fixed_steps'])
    if not 1 <= steps <= int(hyper['num_epochs']):
        raise ValueError('Fixed training steps outside configured budget')

    y_mean, y_std = float(y[:, 0].mean()), float(y[:, 0].std())
    y_std = 1.0 if y_std < 1e-8 else y_std
    x_t = torch.tensor(x_used, dtype=torch.float32, device=device)
    y_t = torch.tensor((y[:, 0] - y_mean) / y_std, dtype=torch.float32, device=device)
    model, likelihood = build_model(x_t, y_t, hyper, deltas, ard_dims, device)
    model.train()
    likelihood.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper["learning_rate"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    snapshot = lambda: ({k: v.detach().cpu() for k, v in model.state_dict().items()},
                        {k: v.detach().cpu() for k, v in likelihood.state_dict().items()})
    for _ in range(steps):
        optimiser.zero_grad()
        loss = -mll(model(x_t), y_t)
        loss.backward()
        optimiser.step()
    model_state, likelihood_state = snapshot()
    return FittedGP(hyper=dict(hyper), model_state=model_state, likelihood_state=likelihood_state,
                    input_mean=x_mean, input_std=x_std, target_mean=y_mean, target_std=y_std,
                    ard_dims=ard_dims, steps=steps, device_name=str(device))


def predict(fitted: FittedGP, train: list[Window], windows: list[Window]) -> tuple[np.ndarray, np.ndarray]:
    """(mean, variance) of the predictive distribution, in the target's scaled units."""
    device = resolve_device(fitted.device_name)
    x_mean = np.array(fitted.input_mean, dtype=np.float32)
    x_std = np.array(fitted.input_std, dtype=np.float32)
    x_std[x_std < 1e-8] = 1.0
    x_train = np.array([w.inputs.flatten() for w in train], dtype=np.float32)
    y_train = np.array([w.target.flatten() for w in train], dtype=np.float32).reshape(len(train), -1)[:, 0]
    x_t = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32, device=device)
    y_t = torch.tensor((y_train - fitted.target_mean) / max(fitted.target_std, 1e-8), dtype=torch.float32, device=device)
    model, likelihood = build_model(x_t, y_t, fitted.hyper, None,
                                    fitted.ard_dims, device)
    model.load_state_dict(fitted.model_state)
    likelihood.load_state_dict(fitted.likelihood_state)
    model.eval()
    likelihood.eval()
    x = np.array([w.inputs.flatten() for w in windows])
    x_eval = torch.tensor((x - fitted.input_mean) / fitted.input_std, dtype=torch.float32, device=device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = likelihood(model(x_eval))
        mean = dist.mean.detach().cpu().numpy()
        var = dist.variance.detach().cpu().numpy()
    return (mean * fitted.target_std + fitted.target_mean).reshape(-1, 1), (var * fitted.target_std ** 2).reshape(-1, 1)

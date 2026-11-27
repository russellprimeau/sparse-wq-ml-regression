"""GP epoch selection from independently prepared chronological folds."""
from __future__ import annotations
import math
import numpy as np
import torch
import gpytorch

def choose_steps(folds, hyper, device_name, seed=0):
    """Fold dictionaries contain training-only transforms and physical y spans."""
    from sparse_wq.models import gp
    gp._seed(seed)
    device = gp.resolve_device(device_name)
    runs = []
    for fold in folds:
        x = np.asarray(fold['x_train'], dtype=np.float32)
        y = np.asarray(fold['y_train'], dtype=np.float32)
        mean, std = gp._standardise(x)
        ymean = float(y.mean())
        ystd = float(y.std())
        if ystd < 1e-08:
            ystd = 1.0
        xt = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
        yt = torch.tensor((y - ymean) / ystd, dtype=torch.float32, device=device)
        xv = torch.tensor((np.asarray(fold['x_validation'], dtype=np.float32) - mean) / std, dtype=torch.float32, device=device)
        draws = fold.get('deltas')
        if draws is not None: raise ValueError('Only measured inputs are supported')
        deltas = None
        dims = x.shape[1]
        budget = max(1, int(len(x) / max(float(hyper.get('ard_min_samples_per_dim', 1.0)), 1e-09)))
        ard = dims if (hyper.get('ard', True) or draws is not None) and dims <= budget else None
        model, likelihood = gp.build_model(xt, yt, hyper, deltas, ard, device)
        runs.append(dict(model=model, likelihood=likelihood, x=xt, y=yt, xv=xv, yv=np.asarray(fold['y_validation']), ymean=ymean, ystd=ystd, span=fold['target_span'], optimizer=torch.optim.Adam(model.parameters(), lr=float(hyper['learning_rate'])), mll=gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)))
    if not runs:
        raise ValueError('No prepared chronological GP budget folds')
    epochs = int(hyper['num_epochs'])
    patience = int(hyper['patience'])
    minimum = max(1, min(int(hyper.get('early_stop_min_epochs', 50)), epochs))
    history = []
    best = float('inf')
    waited = 0
    for epoch in range(epochs):
        squared = []
        for row in runs:
            model, likelihood = (row['model'], row['likelihood'])
            model.train()
            likelihood.train()
            row['optimizer'].zero_grad()
            loss = -row['mll'](model(row['x']), row['y'])
            loss.backward()
            row['optimizer'].step()
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                prediction = likelihood(model(row['xv'])).mean.detach().cpu().numpy()
            error = (prediction * row['ystd'] + row['ymean'] - row['yv']) * row['span']
            if not np.isfinite(error).all():
                raise RuntimeError('Nonfinite chronological GP validation prediction')
            squared.extend(np.square(error).tolist())
        rmse = math.sqrt(math.fsum(squared) / len(squared))
        history.append(rmse)
        if rmse < best:
            best = rmse
            waited = 0
        else:
            waited += 1
        if waited >= patience and epoch + 1 >= minimum:
            break
    return (int(np.argmin(history)) + 1, history)

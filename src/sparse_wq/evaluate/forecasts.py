"""The six agreed physical-unit references, with explicit information cutoffs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calendar_columns(stamps):
    phase = 2 * np.pi * (pd.DatetimeIndex(stamps).dayofyear.to_numpy() - 1) / 365.2425
    return np.column_stack((np.sin(phase), np.cos(phase)))

def prior_values(lab, stamps, rows, horizon=0):
    """Latest laboratory value at/before the horizon-specific window start."""
    lab = lab.sort_index()
    if horizon < 0:
        raise ValueError('Horizon cannot be negative')
    indices = lab.index.searchsorted(pd.DatetimeIndex(stamps) - pd.Timedelta(hours=int(rows) + int(horizon)), side='right') - 1
    result = np.full(len(indices), np.nan)
    valid = indices >= 0
    result[valid] = lab.to_numpy(dtype=float)[indices[valid]]
    return result

def reference_forecasts(lab, train, stamps, rows, *, target, prior=None, horizon=0):
    from sparse_wq.evaluate.references import seasonal_window_mean
    lab = lab.sort_index()
    if not isinstance(lab.index, pd.DatetimeIndex) or not lab.index.is_unique:
        raise ValueError('References need unique laboratory timestamps')
    if not np.isfinite(lab.to_numpy(dtype=float)).all():
        raise ValueError('References require verified finite laboratory values')
    train, times = (pd.DatetimeIndex(train), pd.DatetimeIndex(stamps))
    if not len(train) or not train.is_unique or (not train.is_monotonic_increasing):
        raise ValueError('Supply chronological unique training observations')
    if len(times) and train.max() >= times.min():
        raise ValueError('Reference evaluation must follow its training cutoff')
    history = lab.loc[train].to_numpy(dtype=float)
    expected = prior_values(lab, times, rows, horizon)
    if prior is not None and (not np.allclose(expected, np.asarray(prior, dtype=float), equal_nan=True)):
        raise ValueError('Persistence and model priors differ')
    persistence = expected
    calendar = LinearRegression().fit(calendar_columns(train), history).predict(calendar_columns(times))
    train_prior = prior_values(lab, train, rows, horizon)
    good = np.isfinite(train_prior)
    calendar_prior = np.full(len(times), np.nan)
    eligible = np.isfinite(persistence)
    if good.sum() >= 3 and eligible.any():
        fit = LinearRegression().fit(np.column_stack((calendar_columns(train)[good], train_prior[good])), history[good])
        calendar_prior[eligible] = fit.predict(np.column_stack((calendar_columns(times)[eligible], persistence[eligible])))
    linear, seasonal = ([], [])
    for stamp in times:
        available = lab.loc[:stamp - pd.Timedelta(hours=int(horizon) + 1)]
        recent = available.loc[available.index > stamp - pd.Timedelta(hours=3.1 * (int(rows) + 1))]
        extrapolated = float(available.iloc[-1]) if len(available) else np.nan
        if len(recent) >= 2:
            x = ((recent.index - stamp).total_seconds() / 3600).to_numpy().reshape(-1, 1)
            extrapolated = float(LinearRegression().fit(x, recent.to_numpy(dtype=float)).predict([[0]])[0])
        linear.append(extrapolated)
        earlier_years = available.loc[available.index.year < stamp.year]
        seasonal.append(seasonal_window_mean(earlier_years.index.dayofyear.to_numpy(), earlier_years.index.hour.to_numpy() + earlier_years.index.minute.to_numpy() / 60, earlier_years.to_numpy(dtype=float), stamp.dayofyear, stamp.hour + stamp.minute / 60))
    from sparse_wq.output_constraints import constrain_predictions
    values = {'Training mean': np.repeat(history.mean(), len(times)), 'Calendar': calendar, 'Calendar + prior': calendar_prior, 'Naive': persistence, 'Linear': np.asarray(linear), 'Seasonal': np.asarray(seasonal)}
    return {name: constrain_predictions(prediction, target) for name, prediction in values.items()}

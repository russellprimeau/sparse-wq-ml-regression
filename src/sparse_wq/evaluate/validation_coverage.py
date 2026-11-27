"""Pooled chronological validation with explicit unavailable-fold accounting."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sparse_wq.data.qualified import evaluation_metrics
from sparse_wq.fitting import InsufficientCoverage

def score_available_folds(folds, predict, *, seeds, coverage_penalty, minimum_observations, prediction_column):
    """Empty folds count as lost coverage; unexpected fit/code errors propagate.

    predict(training, validation, seed) returns physical predictions and fit
    information. All seeds must predict the same timestamps in a usable fold.
    """
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Supply unique seeds')
    if not math.isfinite(coverage_penalty) or coverage_penalty < 0 or minimum_observations < 2:
        raise ValueError('Supply a nonnegative penalty and minimum support of at least two')
    if prediction_column not in ('prediction', 'prediction_unclipped'):
        raise ValueError('Choose the prediction-clipping policy explicitly')
    rows, details, coverage, expected_ids = ([], [], [], [])
    for fold, (training, validation) in enumerate(folds, 1):
        training, validation = (pd.DatetimeIndex(training), pd.DatetimeIndex(validation))
        if not len(training) or not len(validation) or training.max() >= validation.min():
            raise ValueError('Chronological validation must strictly follow training')
        expected_ids.extend(validation)
        replicas, identities, failure = ([], [], None)
        for seed in seeds:
            try:
                table, info = predict(training, validation, seed)
            except InsufficientCoverage as error:
                failure = str(error)
                break
            table = table.copy()
            if table.empty:
                failure = 'No finite validation predictions'
                break
            table['timestamp'] = pd.to_datetime(table.timestamp)
            if table.timestamp.duplicated().any() or not set(table.timestamp) <= set(validation):
                raise ValueError('Predictions must identify unique observations in this validation fold')
            if not np.isfinite(table[['observed', prediction_column]].to_numpy(dtype=float)).all():
                raise ValueError('Validation predictions and observations must be finite')
            table = table.sort_values('timestamp')
            identities.append(tuple(table.timestamp))
            table['fold'], table['seed'] = (fold, seed)
            replicas.append(table)
            details.append({**info, 'fold': fold, 'seed': seed})
        if failure is not None:
            if replicas:
                raise ValueError('Coverage changed between seeds; do not discard successful replicas silently')
            coverage.append(dict(fold=fold, n_expected=len(validation), n_scored=0, n_lost=len(validation), status='unavailable', reason=failure))
            continue
        if len(set(identities)) != 1:
            raise ValueError('Seed prediction sets differ; do not average unequal seed memberships')
        combined = pd.concat(replicas, ignore_index=True)
        if combined.groupby('timestamp')['observed'].nunique().gt(1).any():
            raise ValueError('Seeds disagree about observed targets')
        pooled = combined.groupby(['timestamp', 'fold'], as_index=False).agg(observed=('observed', 'first'), prediction=(prediction_column, 'mean'))
        coverage.append(dict(fold=fold, n_expected=len(validation), n_scored=len(pooled), n_lost=len(validation) - len(pooled), status='scored'))
        rows.append(pooled)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Validation observations must not recur across folds')
    n_expected = len(expected_ids)
    table = pd.concat(rows, ignore_index=True).sort_values('timestamp') if rows else pd.DataFrame(columns=['timestamp', 'fold', 'observed', 'prediction'])
    if len(table) < minimum_observations:
        raise InsufficientCoverage(f'Only {len(table)} of {n_expected} validation observations; minimum {minimum_observations}')
    metrics = evaluation_metrics(table.observed.to_numpy(), table.prediction.to_numpy())
    lost_fraction = 1 - len(table) / n_expected
    objective = 1 - metrics['r2'] + coverage_penalty * lost_fraction if math.isfinite(metrics['r2']) else float('inf')
    return (table, dict(metrics=metrics, objective=objective, details=details, fold_coverage=coverage, expected_observations=n_expected, scored_observations=len(table), lost_observations=n_expected - len(table), lost_fraction=lost_fraction, coverage_penalty=coverage_penalty, coverage_penalty_contribution=coverage_penalty * lost_fraction, minimum_observations=minimum_observations, prediction_column=prediction_column))

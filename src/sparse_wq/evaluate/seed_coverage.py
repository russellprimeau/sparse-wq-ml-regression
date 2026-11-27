"""Average seed predictions only where every requested seed predicts.

The trial adapter uses this policy without changing the frozen fitting kernels.
The original strict fold scorer still supplies metrics and coverage penalties.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sparse_wq.fitting import InsufficientCoverage
from sparse_wq.evaluate.validation_coverage import score_available_folds

def collect_seed_predictions(predict, *, seeds, expected, prediction_columns):
    """Return common rows, every raw seed row, fit details and coverage counts.

    An unavailable seed makes the intersection empty. Unexpected fitting errors
    propagate. Keep all raw predictions for auditing, even those not averaged.
    """
    seeds = tuple(seeds)
    expected = pd.DatetimeIndex(expected)
    if not seeds or len(set(seeds)) != len(seeds) or expected.has_duplicates:
        raise ValueError('Supply unique seeds and expected timestamps')
    replicas, details, counts, memberships = ([], [], [], [])
    for seed in seeds:
        reason = None
        try:
            table, info = predict(seed)
        except InsufficientCoverage as error:
            table, info, reason = (pd.DataFrame(), {}, str(error))
        table = table.copy()
        if table.empty:
            table = pd.DataFrame(columns=['timestamp', 'observed', *prediction_columns])
            reason = reason or 'No finite predictions'
        table['timestamp'] = pd.to_datetime(table.timestamp)
        if table.timestamp.duplicated().any() or not set(table.timestamp) <= set(expected):
            raise ValueError('Predictions must identify unique expected observations')
        if not np.isfinite(table[['observed', *prediction_columns]].to_numpy(dtype=float)).all():
            raise ValueError('Seed predictions and observations must be finite')
        table['seed'] = seed
        replicas.append(table)
        details.append({**info, 'seed': seed, 'n_available_predictions': len(table), **({'unavailable_reason': reason} if reason else {})})
        counts.append(dict(seed=seed, n_available=len(table), **{'reason': reason} if reason else {}))
        memberships.append(set(table.timestamp))
    common = set.intersection(*memberships)
    union = set.union(*memberships)
    raw = pd.concat(replicas, ignore_index=True)
    for column in ['observed', 'prior']:
        if column in raw and raw.groupby('timestamp')[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f'Seeds disagree about {column}')
    aligned = raw.loc[raw.timestamp.isin(common)].copy()
    coverage = dict(expected_observations=len(expected), scored_observations=len(common), lost_observations=len(expected) - len(common), lost_fraction=1 - len(common) / len(expected) if len(expected) else 0.0, available_in_any_seed=len(union), excluded_due_to_seed_disagreement=len(union - common), seed_coverage=counts, seed_coverage_policy='intersection: require a prediction from every requested seed')
    return (aligned, raw, details, coverage)

def score_common_seed_folds(folds, predict, *, seeds, coverage_penalty, minimum_observations, prediction_column):
    """Apply the intersection per chronological fold, then penalize all losses."""
    seeds = tuple(seeds)
    folds = list(folds)
    cached, audits = ({}, [])

    def aligned_predict(training, validation, seed):
        key = (tuple(training), tuple(validation))
        if key not in cached:
            aligned, _, details, coverage = collect_seed_predictions(lambda number: predict(training, validation, number), seeds=seeds, expected=validation, prediction_columns=[prediction_column])
            cached[key] = (aligned, {detail['seed']: detail for detail in details})
            audits.append(coverage)
        aligned, details = cached[key]
        if aligned.empty:
            raise InsufficientCoverage('No validation observations predicted by every requested seed')
        return (aligned.loc[aligned.seed.eq(seed)].copy(), details[seed])
    table, info = score_available_folds(folds, aligned_predict, seeds=seeds, coverage_penalty=coverage_penalty, minimum_observations=minimum_observations, prediction_column=prediction_column)
    for fold, audit in zip(info['fold_coverage'], audits):
        fold.update(audit)
    info['seed_coverage_policy'] = 'intersection: require a prediction from every requested seed'
    return (table, info)

"""Numerical fitting shared by chronological selection and fixed-configuration refits."""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from sparse_wq.data.qualified import TARGETS, qualified_records, historical_reconciliation, substitute, chronological_split, expanding_folds, evaluation_metrics

class InsufficientCoverage(ValueError):
    """Unusable training/validation observations, distinct from a fitting error."""

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')

def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def grouped_expanding(windows):
    """Indices for chronological inner validation by independent observation."""
    from sparse_wq.models.samples import base_name
    ids = list(dict.fromkeys((base_name(w.name) for w in windows)))
    if len(ids) < 6:
        raise ValueError('Fewer than six training observations for inner validation')
    start = len(ids) // 2
    parts = []
    for val in np.array_split(np.arange(start, len(ids)), 3):
        if not len(val):
            raise ValueError('Empty chronological inner validation fold')
        tr_ids = set(ids[:int(val[0])])
        va_ids = {ids[i] for i in val}
        parts.append((np.array([i for i, w in enumerate(windows) if base_name(w.name) in tr_ids]), np.array([i for i, w in enumerate(windows) if base_name(w.name) in va_ids])))
    return parts

class Analysis:

    def variant(self, name):
        if name.startswith('xgb_'):
            return ('xgb', self.settings['models']['xgb']['variants'][name])
        if name.startswith('gp_'):
            kernel, aggregation = self.settings['models']['gp']['variants'][name]
            return ('gp', aggregation)
        return ('mlr', 'none')

    def predict(self, train_stamps, eval_stamps, features, variant, seed=0, horizon=0, uncertainty=False, hyper_override=None):
        from sparse_wq.models import gp, xgb, mlr, samples
        from sklearn.linear_model import LinearRegression
        if uncertainty:raise ValueError('This study uses original measured inputs')
        context = self.context(train_stamps, features, variant, horizon, with_uncertainty=uncertainty)
        raw_train, _ = self.raw_windows(train_stamps, features, horizon, through=max(train_stamps), variant=variant)
        raw_eval, meta = self.raw_windows(eval_stamps, features, horizon, through=max(train_stamps), variant=variant)
        train, clips = self.transformed(raw_train, features, context, variant, True, seed, uncertainty)
        validation, _ = self.transformed(raw_eval, features, context, variant, False, seed, uncertainty)
        family, aggregation = self.variant(variant)
        if len({samples.base_name(w.name) for w in train}) < 6 or not validation:
            raise InsufficientCoverage('Insufficient eligible windows for fit/evaluation')
        info = dict(family=family, variant=variant, input_features=list(features), seed=seed, n_train_observations=len({samples.base_name(w.name) for w in train}), n_train_rows=len(train), perturbation_clips=clips, bounds=context['bounds'], calibration=str(context['offsets']), horizon=horizon, uncertainty=uncertainty)
        info['training_observations'] = sorted({samples.base_name(w.name) for w in train})
        if 'feature_metadata' in context:
            info['feature_metadata'] = context['feature_metadata']
        if 'perturbation_diagnostics' in context:
            info['perturbation_diagnostics'] = context['perturbation_diagnostics']
        if family == 'xgb':
            hyper = copy.deepcopy(hyper_override or self.tune(train_stamps, features, variant, horizon))
            hyper['random_state'] = seed
            model, detail = xgb.fit(hyper, train, self.device, budget_splits=grouped_expanding(train))
            pred = xgb.predict(model, validation)[:, 0]
            info.update(detail)
            info['hyperparameters'] = hyper
        elif family == 'gp':
            hyper = copy.deepcopy(hyper_override or self.settings['models']['gp']['hyperparameters'])
            hyper.update(kernel=self.settings['models']['gp']['variants'][variant][0], num_epochs=self.epochs, use_uncertain_input_kernel=bool(context['errors']) and uncertainty)
            model = gp.fit(hyper, train, features, self.rows, aggregation, context['bounds'], None, self.device, budget_splits=grouped_expanding(train))
            pred = gp.predict(model, train, validation)[0][:, 0]
            info['steps'] = model.steps
            info['hyperparameters'] = hyper
        else:
            mode = mlr.AGGREGATION[variant]
            fixed = (hyper_override or {}).get('revision_selected_features')
            screening = self.mlr_screening_windows(train) if hasattr(self, 'mlr_screening_windows') else train
            kept = [features.index(c) for c in fixed] if fixed is not None else mlr.spearman_screen(screening, features, mode)
            x = np.array([mlr.predictor_row(w.inputs, mode, features)[kept] for w in train])
            y = np.array([w.target[0, 0] for w in train])
            settings = mlr.selection_settings({'method': 'Spearman', 'selection_seeds': 6})
            selection_x = np.array([mlr.predictor_row(w.inputs, mode, features)[kept] for w in screening])
            selection_y = np.array([w.target[0, 0] for w in screening])
            chosen = list(range(len(kept))) if fixed is not None else mlr.select(selection_x, selection_y, settings)
            info['n_screening_observations'] = len(screening)
            eval_x = np.array([mlr.predictor_row(w.inputs, mode, features)[kept] for w in validation])
            pred = np.full(len(validation), np.nan)
            if not chosen:
                pred[:] = float(np.mean(y))
            else:
                good = np.isfinite(x[:, chosen]).all(axis=1) & np.isfinite(y)
                eligible = np.isfinite(eval_x[:, chosen]).all(axis=1)
                if good.sum() < 3:
                    raise InsufficientCoverage('Insufficient training rows after MLR screening')
                if not eligible.any():
                    raise InsufficientCoverage('No complete validation rows after MLR screening')
                model = LinearRegression().fit(x[good][:, chosen], y[good])
                pred[eligible] = model.predict(eval_x[eligible][:, chosen])
                info['n_train_observations'] = len({samples.base_name(w.name) for w, k in zip(train, good) if k})
                info['training_observations'] = sorted({samples.base_name(w.name) for w, k in zip(train, good) if k})
            info['selected_features'] = [features[kept[j]] for j in chosen]
        output = context['output']
        bound = context['bounds'][output]
        span = bound['max'] - bound['min'] or 1.0
        from .output_constraints import constrain_predictions, POLICY
        physical_unclipped = pred * span + bound['min']
        rows = []
        for window, raw in zip(validation, physical_unclipped):
            if not np.isfinite(raw):
                continue
            meta_row = meta[samples.base_name(window.name)]
            level = float(raw + meta_row['prior']) if self.difference else float(raw)
            physical = float(constrain_predictions(level, self.target))
            rows.append(dict(**meta_row, prediction=physical, prediction_unclipped=level,
                             level_clipped=physical != level,
                             sample_file=samples.base_name(window.name)))
        info['output_constraint'] = POLICY
        info['negative_prediction_floors'] = sum(row['level_clipped'] for row in rows)
        info['prediction_clips'] = info['negative_prediction_floors']
        self.fit_count += 1
        return (pd.DataFrame(rows), info)

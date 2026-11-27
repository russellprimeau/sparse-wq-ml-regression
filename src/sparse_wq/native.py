"""Native-time inputs and training-partition transforms for the study."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import yaml
from sparse_wq.data.qualified import chronological_split, substitute
from sparse_wq.fitting import Analysis, write_json, finite_json, InsufficientCoverage
from sparse_wq.data.native import NativeInputs, window_bounds, encode_window, digest, DEEP_SUFFIX

class NativeAnalysis(Analysis):

    def __init__(self, prepared, calibration, target, run, *, treatment='half', trials=1, epochs=5, device='cpu', output=None, inputs=None):
        from sparse_wq import design
        self.prepared = Path(prepared)
        self.inputs = inputs or NativeInputs(self.prepared, weather_cache=self.prepared.parent / 'weather_reference_cache')
        self.provenance = self.inputs.manifest
        self.study = self.provenance['study']
        self.settings = yaml.safe_load((self.prepared / 'search.yml').read_text(encoding='utf-8'))
        self.target, self.run, self.treatment = (target, run, treatment)
        self.root = Path(output or self.prepared.parent / 'integration_fits')
        self.calibration, self.device, self.trials, self.epochs = (None, device, trials, epochs)
        self.slug = self.study['targets'][target]
        self.out = self.root / treatment / run / self.slug
        records = pd.read_csv(self.prepared / 'qualified_observations.csv', parse_dates=['TIMESTAMP'], float_precision='round_trip')
        if records.is_left_censored.dtype != bool:
            raise ValueError('Laboratory censoring flags must be explicit booleans')
        records = substitute(records, treatment)
        self.lab = records.loc[records.target.eq(target)].set_index('TIMESTAMP').sort_index()
        self.train, self.test = chronological_split(self.lab.index)
        self.rows = max(1, int(round(float(np.median((self.train[1:] - self.train[:-1]).total_seconds() / 3600)))) - 1)
        self.features = list(self.study['predictors']['always'])
        if design.offers_profiler(run):
            self.features += list(self.study['predictors']['profiler'])
        if design.RUNS[run][2] == 'offered':
            self.features.append(target + '_state')
        self.difference = not design.is_absolute(run)
        self._contexts, self._tuned, self.trace, self.fit_count = ({}, {}, [], 0)
        from sparse_wq.models import gp, xgb, mlr, samples
        from sparse_wq import output_constraints
        from sparse_wq.data import native as rerun_inputs
        from sparse_wq.data import summaries as native_summary
        from sparse_wq import fitting as review_pipeline
        from sparse_wq.evaluate import validation_coverage as validation_coverage
        from sparse_wq.evaluate import forecasts as rerun_references
        kernels = {module.__name__: digest(module.__file__) for module in [gp, xgb, mlr, samples, rerun_inputs, native_summary, review_pipeline, validation_coverage, rerun_references, output_constraints]}
        versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'torch', 'gpytorch', 'xgboost', 'scikit-learn', 'optuna']}
        self.identity = hashlib.sha256(json.dumps(dict(preparation=digest(self.prepared / 'preparation.json'), target=target, run=run, treatment=treatment, rows=self.rows, trials=trials, epochs=epochs, device=device, settings=self.settings, implementation=digest(__file__), kernels=kernels, versions=versions), sort_keys=True).encode()).hexdigest()
        existing = self.out / 'identity.json'
        if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))['identity'] != self.identity:
            raise ValueError('Use a new output root after changes to inputs, code or fitting settings')
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(existing, dict(identity=self.identity, python=platform.python_version(), device=device, kernels=kernels, versions=versions, rows=self.rows, train=list(map(str, self.train)), test=list(map(str, self.test)), stage='study_analysis'))

    def representation(self, variant):
        if variant is None:
            return 'mean'
        family, representation = self.variant(variant)
        if family == 'mlr':
            return {'mlr': 'latest', 'mlr_avg12': 'last12mean', 'mlr_avgall': 'mean'}[variant]
        return representation

    def raw_windows(self, stamps, features, horizon=0, through=None, variant=None):
        from sparse_wq.models.samples import Window
        if through is None:
            raise ValueError('Native weather preparation requires an explicit training cutoff')
        windows, metadata = ([], {})
        environmental = [name for name in features if not name.endswith('_state')]
        representation = self.representation(variant)
        for stamp in pd.DatetimeIndex(stamps):
            start, cutoff, reference = window_bounds(stamp, self.rows, horizon)
            available = self.lab.loc[:reference, 'value']
            prior_stamp = available.index[-1] if len(available) else pd.NaT
            prior = float(available.iloc[-1]) if len(available) else np.nan
            if pd.notna(prior_stamp) and prior_stamp > start:
                raise AssertionError('Laboratory prior falls inside the predictor window')
            feature_start = start
            if representation == 'latest':
                feature_start = cutoff
            elif representation == 'last12mean':
                feature_start = max(start, cutoff - pd.Timedelta(hours=11))
            observation_start = feature_start - pd.Timedelta(hours=1) + pd.Timedelta(nanoseconds=1) if representation in ('none', 'latest', 'last12mean') else feature_start
            observations = self.inputs.observations(observation_start, cutoff, through, environmental)
            observations.attrs.update(prior=prior, start=feature_start, cutoff=cutoff, observation_start=observation_start, window_start=start)
            level = float(self.lab.at[stamp, 'value'])
            name = 'segment_' + stamp.strftime('%Y%m%d%H') + '.csv'
            windows.append(Window(observations, np.array([[level - prior if self.difference else level]]), name))
            metadata[name] = dict(timestamp=str(stamp), reference_timestamp=str(reference), prior=prior, prior_timestamp=str(prior_stamp) if pd.notna(prior_stamp) else '', observed=level, predictor_start=str(start), predictor_cutoff=str(cutoff), training_through=str(pd.Timestamp(through)))
        return (windows, metadata)

    def context(self, stamps, features, variant, horizon=0, with_uncertainty=False):
        raw, _ = self.raw_windows(stamps, features, horizon, through=max(stamps), variant=variant)
        bounds = {}
        for name in features:
            values = np.array([w.inputs.attrs['prior'] for w in raw]) if name.endswith('_state') else np.concatenate([w.inputs[name].to_numpy(dtype=float) for w in raw])
            values = values[np.isfinite(values)]
            if not len(values):
                raise InsufficientCoverage(f'No training values for {name}')
            bounds[name] = dict(min=float(values.min()), max=float(values.max()))
        y = np.array([w.target[0, 0] for w in raw])
        y = y[np.isfinite(y)]
        if len(y) < 6:
            raise InsufficientCoverage('Insufficient finite training targets')
        output = self.target + ('_diff' if self.difference else '')
        bounds[output] = dict(min=float(y.min()), max=float(y.max()))
        errors, offsets = ({}, None)
        if with_uncertainty:raise ValueError('This study uses original measured inputs')
        return dict(bounds=bounds, output=output, offsets=offsets, errors=errors, training_through=str(max(stamps)))

"""Training-only chronological predictor, representation, family and tuning selection."""
from __future__ import annotations
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sparse_wq.native import NativeAnalysis
from sparse_wq.data.qualified import expanding_folds, evaluation_metrics
from sparse_wq.fitting import InsufficientCoverage, finite_json, write_json
from sparse_wq.data.native import DEEP_SUFFIX, digest
from sparse_wq.search.subsets import search_subsets

class Analysis(NativeAnalysis):

    def __init__(self, *args, protocol, uncertainty=False, sensor_history=None, **kwargs):
        self.protocol = copy.deepcopy(protocol)
        if uncertainty:raise ValueError('This study uses original measured inputs')
        self.uncertainty = False
        self._basis_contexts = {}
        self._vectors = OrderedDict()
        self._vector_bytes = 0
        self._raw_cache = OrderedDict()
        self._raw_bytes = 0
        self._gp_budgets = {}
        super().__init__(*args, **kwargs)
        marker = self.out / 'trial_protocol.json'
        record = dict(protocol=self.protocol, input_replication='one original window per observation')
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != record:
            raise ValueError('Trial protocol changed; use a fresh output series')
        write_json(marker, record)

    def compatible(self, features, variant):
        representation = self.representation(variant)
        return not (representation in ('none', 'latest') and any((name.endswith(DEEP_SUFFIX) for name in features)))

    def raw_windows(self, stamps, features, horizon=0, through=None, variant=None):
        from sparse_wq.models.samples import Window
        if through is None:
            raise ValueError('Explicit training cutoff required')
        representation = self.representation(variant)
        scope = 'full' if representation in ('mean', 'stats:12h', 'stats:24h') else representation
        key = (tuple(pd.DatetimeIndex(stamps)), scope, int(horizon), pd.Timestamp(through))
        if key not in self._raw_cache:
            basis = [name for name in self.features if self.compatible([name], variant)]
            windows, metadata = super().raw_windows(stamps, basis, horizon=horizon, through=through, variant=variant)
            size = sum((int(window.inputs.memory_usage(index=True, deep=True).sum()) for window in windows))
            self._raw_cache[key] = (windows, metadata, size)
            self._raw_bytes += size
            while self._raw_bytes > 512 * 1024 ** 2 and len(self._raw_cache) > 1:
                _, (_, _, evicted) = self._raw_cache.popitem(last=False)
                self._raw_bytes -= evicted
        self._raw_cache.move_to_end(key)
        windows, metadata, _ = self._raw_cache[key]
        environmental = [name for name in features if not name.endswith('_state')]
        return ([Window(window.inputs[environmental].copy(), window.target.copy(), window.name) for window in windows], metadata)



    def context(self, stamps, features, variant, horizon=0, with_uncertainty=False):
        if not self.compatible(features, variant):
            raise ValueError('Incompatible native representation')
        canonical = [name for name in self.features if name in features]
        if list(features) != canonical:
            raise ValueError('Trial predictors must retain their canonical offered order')
        basis = [name for name in self.features if self.compatible([name], variant)]
        mode = bool(with_uncertainty)
        key = (tuple(pd.DatetimeIndex(stamps)), variant, int(horizon), mode)
        if key not in self._basis_contexts:
            raw = self.raw_windows(stamps, basis, horizon=horizon, through=max(stamps), variant=variant)[0]
            available = []
            for name in basis:
                values = np.array([window.inputs.attrs['prior'] for window in raw]) if name.endswith('_state') else np.concatenate([window.inputs[name].to_numpy(dtype=float) for window in raw])
                if np.isfinite(values).any():
                    available.append(name)
            basis = available
            context = super().context(stamps, basis, variant, horizon, with_uncertainty=mode)
            context['defer_input_filtering'] = True
            self._basis_contexts[key] = (basis, context)
        _, context = self._basis_contexts[key]
        unavailable = [name for name in features if name not in context['bounds']]
        if unavailable:
            raise InsufficientCoverage(f'No training values for {unavailable}')
        output = context['output']
        selected = {key: value for key, value in context.items() if key not in ('bounds', 'errors', 'feature_metadata', 'perturbation_diagnostics', 'defer_input_filtering')}
        selected['bounds'] = {name: context['bounds'][name] for name in [*features, output]}
        selected['errors'] = {name: context['errors'][name] for name in features if name in context['errors']}
        selected['_basis_key'] = key
        return selected


    def transformed(self, raw, features, context, variant, training=False, seed=0, uncertainty=False):
        from sparse_wq.models.samples import Window, base_name
        family, _ = self.variant(variant)
        basis_key = context['_basis_key']
        basis, basis_context = self._basis_contexts[basis_key]
        stamps = tuple(pd.to_datetime([base_name(window.name).removeprefix('segment_').removesuffix('.csv') for window in raw], format='%Y%m%d%H'))
        data_seed = seed if training and family != 'gp' and uncertainty and basis_context['errors'] else 0
        key = (basis_key, stamps, bool(training), int(data_seed), bool(uncertainty))
        if key not in self._vectors:
            source = self.raw_windows(stamps, basis, horizon=basis_key[2], through=pd.Timestamp(context['training_through']), variant=variant)[0]
            encoded, layout, diagnostics = self._encode_basis(source, basis, basis_context, variant, training, data_seed, uncertainty)
            size = sum((window.inputs.nbytes + window.target.nbytes for window in encoded))
            self._vectors[key] = (encoded, layout, diagnostics, size)
            self._vector_bytes += size
            while self._vector_bytes > 512 * 1024 ** 2 and len(self._vectors) > 1:
                _, (_, _, _, evicted) = self._vectors.popitem(last=False)
                self._vector_bytes -= evicted
        self._vectors.move_to_end(key)
        encoded, layout, diagnostics, _ = self._vectors[key]
        positions = [i for i, row in enumerate(layout) if row['column'] in features]
        context['feature_metadata'] = [layout[i] for i in positions]
        result = []
        for window in encoded:
            values = window.inputs[:, positions]
            if family == 'gp' and (not np.isfinite(values).all()):
                continue
            if family == 'xgb' and np.mean(~np.isfinite(values)) > 0.8:
                continue
            result.append(Window(values, window.target, window.name))
        chosen = [diagnostics[name] for name in features if name in diagnostics]
        if training:
            context['perturbation_diagnostics'] = chosen
        return (result, sum((row['n_outside_training_range'] for row in chosen)))

    def _encode_basis(self, raw, features, context, variant, training, seed, uncertainty):
        from sparse_wq.models.samples import Window
        from sparse_wq.data.feature_plan import NativeFeaturePlan
        if uncertainty:raise ValueError('Only original measured inputs are supported')
        representation=self.representation(variant)
        environmental=[name for name in features if not name.endswith('_state')]
        minima=np.array([context['bounds'][name]['min'] for name in environmental])
        spans=np.array([context['bounds'][name]['max']-context['bounds'][name]['min'] or 1.0 for name in environmental])
        output=context['bounds'][context['output']]
        output_span=output['max']-output['min'] or 1.0
        result=[];layout=None
        for window in raw:
            observations=window.inputs
            start=observations.attrs['observation_start'] if representation=='last12mean' else observations.attrs['start']
            spec='mean' if representation=='last12mean' else 'none' if representation=='latest' else representation
            plan=NativeFeaturePlan(observations,environmental,start,observations.attrs['cutoff'],spec)
            current=copy.deepcopy(plan.metadata);held=[]
            for name in features:
                if name.endswith('_state'):
                    bound=context['bounds'][name]
                    held.append((observations.attrs['prior']-bound['min'])/(bound['max']-bound['min'] or 1.0))
                    current.append(dict(column=name,scope='window',block_index=-1,statistic='held_prior'))
            if layout is not None and layout!=current:raise ValueError('Feature layout changed across windows')
            layout=current
            normalized=(plan.original-minima)/spans
            target=(window.target-output['min'])/output_span
            if not np.isfinite(target).all():continue
            vector=np.r_[plan.encode(normalized.copy()),held]
            result.append(Window(vector.reshape(1,-1),target.copy(),window.name))
        return result,layout or [],{}


    def _prepared_folds(self, stamps, features, variant, *, uncertainty, seed, minimum_training):
        contexts = []
        coverage = []
        for fold, (training, validation) in enumerate(expanding_folds(stamps), 1):
            try:
                context = self.context(training, features, variant, with_uncertainty=uncertainty)
                raw_train = self.raw_windows(training, features, through=max(training), variant=variant)[0]
                raw_validation = self.raw_windows(validation, features, through=max(training), variant=variant)[0]
                tr, _ = self.transformed(raw_train, features, context, variant, True, seed, uncertainty)
                va, _ = self.transformed(raw_validation, features, context, variant, False, seed, uncertainty)
            except InsufficientCoverage as error:
                coverage.append(dict(fold=fold, status='unavailable', n_expected=len(validation), n_scored=0, reason=str(error)))
                continue
            if len(tr) < minimum_training or not va:
                coverage.append(dict(fold=fold, status='unavailable', n_expected=len(validation), n_scored=0, n_training=len(tr)))
                continue
            coverage.append(dict(fold=fold, status='scored', n_expected=len(validation), n_scored=len(va), n_training=len(tr)))
            contexts.append((tr, va, context))
        if not contexts:
            raise InsufficientCoverage('No usable nested training-validation folds')
        return (contexts, coverage)

    def tune(self, train_stamps, features, variant, horizon=0):
        import optuna
        import xgboost
        from sparse_wq.models import xgb
        from sparse_wq.search.tuning import constrain_space, _suggest, _sanitise
        if horizon:
            raise ValueError('Hyperparameter tuning uses zero-horizon inputs')
        pool = tuple((name for name in self.features if self.compatible([name], variant)))
        token = (tuple(train_stamps), pool, variant, self.uncertainty)
        name = hashlib.sha256(repr(token).encode()).hexdigest()[:20]
        path = self.out / 'tuning' / f'{variant}_{name}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))['hyperparameters']
        setting = copy.deepcopy(self.settings['models']['xgb']['hyperparameters'])
        contexts, coverage = self._prepared_folds(train_stamps, pool, variant, uncertainty=self.uncertainty, seed=0, minimum_training=6)
        space = constrain_space(setting['cv_tuning']['param_space'], len(train_stamps))

        def objective(trial):
            parameters = _sanitise({key: _suggest(trial, key, spec) for key, spec in space.items()})
            errors = []
            for training, validation, context in contexts:
                model, _ = xgb.fit({**setting, **parameters, 'early_stopping_rounds': None, 'random_state': 21}, training, self.device)
                prediction = xgb.predict(model, validation)[:, 0]
                y = np.array([window.target[0, 0] for window in validation])
                bound = context['bounds'][context['output']]
                span = bound['max'] - bound['min'] or 1.0
                errors.extend(np.square((prediction - y) * span).tolist())
            return float(np.sqrt(np.mean(errors)))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=21))
        study.optimize(objective, n_trials=self.trials, catch=(xgboost.core.XGBoostError,))
        result = {**setting, **_sanitise(study.best_params), 'early_stopping_rounds': None}
        write_json(path, finite_json(dict(hyperparameters=result, best_score=study.best_value, coverage=coverage, uncertainty=self.uncertainty, offered_pool=list(pool), trials=[dict(number=row.number, value=row.value, params=row.params) for row in study.trials])))
        return result

    def gp_budget(self, stamps, features, variant, seed):
        from sparse_wq.search.gp_budget import choose_steps
        token = (tuple(stamps), tuple(features), variant, seed, self.uncertainty)
        name = hashlib.sha256(repr(token).encode()).hexdigest()[:20]
        path = self.out / 'gp_budgets' / f'{name}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))['steps']
        contexts, coverage = self._prepared_folds(stamps, features, variant, uncertainty=self.uncertainty, seed=seed, minimum_training=6)
        folds = []
        for training, validation, context in contexts:
            bound = context['bounds'][context['output']]
            folds.append(dict(x_train=np.array([window.inputs.flatten() for window in training]), y_train=np.array([window.target[0, 0] for window in training]), x_validation=np.array([window.inputs.flatten() for window in validation]), y_validation=np.array([window.target[0, 0] for window in validation]), target_span=bound['max'] - bound['min'] or 1.0, deltas=None))
        hyper = copy.deepcopy(self.settings['models']['gp']['hyperparameters'])
        hyper.update(kernel=self.settings['models']['gp']['variants'][variant][0], num_epochs=self.epochs)
        steps, history = choose_steps(folds, hyper, self.device, seed)
        write_json(path, dict(steps=steps, physical_validation_rmse=history, coverage=coverage, uncertainty=self.uncertainty, preprocessing='independently fitted in every nested fold'))
        return steps

    def predict(self, train_stamps, eval_stamps, features, variant, seed=0, horizon=0, uncertainty=None, hyper_override=None):
        if uncertainty is not None and bool(uncertainty) != self.uncertainty:
            raise ValueError('Use a separate independently selected trial branch for another uncertainty method')
        if horizon and (not hyper_override):
            raise ValueError('Positive horizons require saved final hyperparameters')
        if self.variant(variant)[0] == 'gp' and hyper_override is None:
            hyper_override = copy.deepcopy(self.settings['models']['gp']['hyperparameters'])
            hyper_override['revision_fixed_steps'] = self.gp_budget(train_stamps, features, variant, seed)
        try:
            return super().predict(train_stamps, eval_stamps, features, variant, seed=seed, horizon=horizon, uncertainty=self.uncertainty, hyper_override=hyper_override)
        except ValueError as error:
            if variant.startswith('mlr') and str(error).startswith('Found array with 0 sample(s)'):
                raise InsufficientCoverage('No complete validation rows after MLR screening') from error
            raise

    def search(self):
        options = self.protocol['search']

        def measure(features, variant, seeds):
            _, info = self.score(features, variant, seeds=seeds, coverage_penalty=options['coverage_penalty'], minimum_observations=options['minimum_validation'], prediction_column=self.protocol['prediction_column'], uncertainty=self.uncertainty)
            print(f"Selection {self.slug}/{self.run}/uncertainty={self.uncertainty}: {variant}, {len(features)} predictors, objective={info['objective']}", flush=True)
            return info

        def seeds(features, variant):
            stochastic = variant.startswith('xgb')
            return tuple(range(self.protocol['finalist_seeds'])) if stochastic else (0,)
        selected, finalists, trace = search_subsets(self.features, list(self.settings['models']['xgb']['variants']), [*self.settings['models']['gp']['variants'], *self.settings['models']['xgb']['variants'], 'mlr', 'mlr_avg12', 'mlr_avgall'], measure, self.compatible, budget=options['subset_budget'], beam_width=options['beam_width'], rounds=max(options['rounds'], len(self.features) - options['min_features']), swaps=options['swaps'], min_features=options['min_features'], final_top_k=options['final_top_k'], seed=options['seed'], finalist_seeds=seeds, require_profiler=self.run.startswith('profiler_'))
        write_json(self.out / 'selected.json', finite_json(selected))
        write_json(self.out / 'finalists.json', finite_json(finalists))
        write_json(self.out / 'search_trace.json', finite_json(trace))
        return selected

    def score(self, features, variant, seeds=(0,), *, coverage_penalty, minimum_observations, prediction_column, uncertainty, hyper_override=None):
        """Score the observations predicted by every finalist seed."""
        from sparse_wq.evaluate.seed_coverage import score_common_seed_folds
        seeds = tuple(seeds)
        settings = dict(features=list(features), variant=variant, seeds=list(seeds), coverage_penalty=coverage_penalty, minimum_observations=minimum_observations, prediction_column=prediction_column, uncertainty=uncertainty, hyper_override=hyper_override)
        token = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        path = self.out / 'validation' / f'{token}.csv'
        metadata = path.with_suffix('.json')
        if path.exists() and metadata.exists():
            return (pd.read_csv(path), json.loads(metadata.read_text(encoding='utf-8')))
        folds=list(expanding_folds(self.train))
        raw_predictions=[]
        def predict(training, validation, seed):
            table,detail=self.predict(training,validation,features,variant,seed=seed,uncertainty=uncertainty,hyper_override=hyper_override)
            fold=next(i for i,(_,dates) in enumerate(folds,1) if dates.equals(validation))
            raw_predictions.append(table.assign(fold=fold,seed=seed))
            return table,detail
        table, info = score_common_seed_folds(folds, predict, seeds=seeds, coverage_penalty=coverage_penalty, minimum_observations=minimum_observations, prediction_column=prediction_column)
        info.update(settings, selection='training_validation_only', degenerate=bool(table.groupby('fold').prediction.std(ddof=0).max() <= max(1e-12, 0.0001 * table.observed.std(ddof=0))))
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        pd.concat(raw_predictions,ignore_index=True).to_csv(path.with_name(path.stem+'_seeds.csv'),index=False)
        write_json(metadata, finite_json(info))
        return (table, finite_json(info))

    def assess(self, selected):
        from sparse_wq.evaluate.seed_coverage import collect_seed_predictions
        aligned, raw, details, coverage = collect_seed_predictions(lambda seed: self.predict(self.train, self.test, selected['features'], selected['variant'], seed=seed), seeds=selected['seeds'], expected=self.test, prediction_columns=['prediction', 'prediction_unclipped'])
        destination = self.out / 'assessment'
        destination.mkdir(parents=True, exist_ok=True)
        raw.to_csv(destination / 'seed_predictions.csv', index=False)
        write_json(destination / 'coverage.json', coverage)
        if aligned.empty:
            raise InsufficientCoverage('No evaluation observations predicted by every requested seed')
        table = aligned.groupby('timestamp', as_index=False).agg(observed=('observed', 'first'), prediction=('prediction', 'mean'), prediction_unclipped=('prediction_unclipped', 'mean'), prior=('prior', 'first'))
        column = self.protocol['prediction_column']
        metrics = evaluation_metrics(table.observed.to_numpy(), table[column].to_numpy())
        table.to_csv(destination / 'predictions.csv', index=False)
        write_json(destination / 'fit.json', finite_json(dict(metrics=metrics, details=details, coverage=coverage, selected_by='training-only chronological validation', assessment_role=self.protocol['assessment_role'])))
        return (table, metrics)

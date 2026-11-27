"""Bounded checks of reported configurations; never search or tune a model.

DL substitution replays saved fold/final fitting settings. Influence checks
only rescore saved predictions. Primary study results are read-only inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_wq import paths, workflow
from sparse_wq.analysis import Analysis
from sparse_wq.data.native import NativeInputs, digest
from sparse_wq.data.qualified import evaluation_metrics, expanding_folds
from sparse_wq.evaluate.block_permutation import read_json, save_json
from sparse_wq.evaluate.forecasts import reference_forecasts
from sparse_wq.evaluate.seed_coverage import collect_seed_predictions
from sparse_wq.fitting import finite_json

CHECK_VERSION = 1
CENSORED_TARGETS = ('turbidity', 'arsenic', 'lead', 'cadmium', 'chromium', 'nickel')

def rmse(y, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(prediction)))))


def summarize(table):
    references = {name.removeprefix('reference: '): evaluation_metrics(table.observed, table[name])
                  for name in table if name.startswith('reference: ')}
    result = evaluation_metrics(table.observed, table.prediction)
    best = min(references, key=lambda name: (references[name]['rmse'], name))
    return dict(**result, strongest_reference=best, references=references,
                rmse_reduction_percent=100 * (1 - result['rmse'] / references[best]['rmse']))


def add_references(table, lab, train, rows, *, target):
    result = table.copy()
    for name, values in reference_forecasts(lab, train, table.index, rows, target=target, prior=table.prior).items():
        result['reference: ' + workflow.REFERENCE_NAMES[name]] = values
    if not np.isfinite(result.filter(regex='^reference: ').to_numpy()).all():
        raise ValueError('A prescribed comparison date lacks a reference')
    return result


def settings(detail, variant):
    hyper = copy.deepcopy(detail.get('hyperparameters', {}))
    if variant.startswith('gp_'):
        hyper['revision_fixed_steps'] = detail['steps']
    if variant.startswith('mlr'):
        hyper['revision_selected_features'] = detail['selected_features']
    if not hyper:
        raise ValueError('Saved fitting settings are required; tuning is prohibited')
    return hyper


def fit_phase(model, selected, training, evaluation, original_details, destination):
    """Save every seed; reject changed training or prediction memberships."""
    signature = dict(version=CHECK_VERSION, model_identity=model.identity,
                     features=selected['features'], variant=selected['variant'], seeds=selected['seeds'],
                     training=list(map(str, training)), evaluation=list(map(str, evaluation)),
                     settings=original_details)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'fit.json'
    if marker.exists():
        saved = read_json(marker)
        if saved['signature'] != finite_json(signature):
            raise ValueError('Cached fit differs from requested inputs/settings: ' + str(destination))
        table = pd.read_csv(destination / 'predictions.csv', parse_dates=['timestamp'],
                            float_precision='round_trip').set_index('timestamp')
        return table, saved['details']
    lookup = {detail['seed']: detail for detail in original_details}

    def predict(seed):
        original = lookup[seed]
        frame, detail = model.predict(training, evaluation, selected['features'], selected['variant'],
                                      seed=seed, hyper_override=settings(original, selected['variant']))
        if detail['training_observations'] != original['training_observations']:
            raise ValueError('Substitution changed fitted training observations')
        if selected['variant'].startswith('gp_') and detail['steps'] != original['steps']:
            raise ValueError('Saved GP fitting budget changed')
        return frame, detail

    aligned, raw, details, coverage = collect_seed_predictions(
        predict, seeds=selected['seeds'], expected=evaluation,
        prediction_columns=['prediction', 'prediction_unclipped'])
    if aligned.empty:
        raise ValueError('No predictions for this prescribed phase')
    pooled = aligned.groupby('timestamp').agg(observed=('observed', 'first'),
        prediction=('prediction', 'mean'), prior=('prior', 'first'))
    raw.to_csv(destination / 'seed_predictions.csv', index=False)
    pooled.to_csv(destination / 'predictions.csv')
    save_json(marker, finite_json(dict(signature=signature, details=details, coverage=coverage)))
    return pooled, details


def model_for(args, store, case, treatment):
    config = workflow.parser().parse_args(['--prepared', str(args.prepared), '--device', args.device])
    protocol = workflow.protocol(config, [case['target']])
    protocol.update(treatment=treatment, targeted_check_version=CHECK_VERSION)
    return Analysis(args.prepared, None, case['laboratory_target'], case['run'], treatment=treatment,
                    protocol=protocol, uncertainty=False, trials=250, epochs=250, device=args.device,
                    output=args.output / 'inputs' / case['method'], inputs=store)


def source_directory(args, case):
    return args.results / 'models' / case['method'] / 'half' / case['run'] / case['target']


def censoring(args, store, cases):
    summaries = []
    for case in cases:
        if case['target'] not in CENSORED_TARGETS:
            continue
        source = source_directory(args, case)
        selected = read_json(source / 'selected.json')
        original = pd.read_csv(source / 'validation.csv', parse_dates=['timestamp'],
                               float_precision='round_trip').set_index('timestamp')
        assessed = pd.read_csv(source / 'assessment/predictions.csv', parse_dates=['timestamp'],
                               float_precision='round_trip').set_index('timestamp')
        final_details = read_json(source / 'assessment/fit.json')['details']
        for treatment in ('half', 'sqrt2'):
            model = model_for(args, store, case, treatment)
            root = args.output / 'censoring' / case['target'] / treatment
            validation = []
            for fold, (training, dates) in enumerate(expanding_folds(model.train), 1):
                old = original.loc[original['fold'].eq(fold)]
                if old.empty:
                    continue
                details = [d for d in selected['info']['details'] if d['fold'] == fold]
                table, fitted = fit_phase(model, selected, training, dates, details, root / f'fold{fold:02d}')
                if not table.index.equals(old.index):
                    raise ValueError('Substitution changed validation prediction coverage')
                if treatment == 'half':
                    np.testing.assert_allclose(table.prediction, old.prediction, rtol=1e-6, atol=1e-10)
                    np.testing.assert_array_equal(table.observed, old.observed)
                table = add_references(table, model.lab.value, training, model.rows, target=model.target)
                validation.append(table.assign(fold=fold))
            validation = pd.concat(validation).sort_index()
            evaluation, fitted = fit_phase(model, selected, model.train, model.test, final_details, root / 'evaluation')
            if not evaluation.index.equals(assessed.index):
                raise ValueError('Substitution changed evaluation prediction coverage')
            if treatment == 'half':
                np.testing.assert_allclose(evaluation.prediction, assessed.prediction, rtol=1e-6, atol=1e-10)
                np.testing.assert_array_equal(evaluation.observed, assessed.observed)
            evaluation = add_references(evaluation, model.lab.value, model.train, model.rows, target=model.target)
            for period, table in [('validation', validation), ('evaluation', evaluation)]:
                dates = pd.to_datetime(case[period]['timestamps'])
                scored = table.loc[dates].rename_axis('timestamp')
                scored.to_csv(root / (period + '_scored.csv'))
                metrics = summarize(scored)
                if treatment == 'half':
                    np.testing.assert_allclose(metrics['rmse'], case[period]['scores']['rmse'], rtol=1e-6, atol=1e-10)
                summaries.append(dict(target=case['target'], target_name=case['target_name'],
                                      period=period, treatment=treatment, **metrics))
            save_json(root / 'summary.json', finite_json(summaries[-2:]))
            print(json.dumps(dict(check='censoring', target=case['target'], treatment=treatment,
                                  evaluation=summaries[-1]['rmse'], reproduced=treatment == 'half')), flush=True)
    save_json(args.output / 'censoring_summary.json', finite_json(summaries))
    pd.DataFrame([{k: v for k, v in row.items() if k != 'references'} for row in summaries]).to_csv(
        args.output / 'censoring_summary.csv', index=False)


def influence(args,cases):
    """Omit one evaluation date from scoring fixed primary-model predictions."""
    records=workflow.load_records(args.prepared)
    rows=[]
    for case in cases:
        source=source_directory(args,case)
        lab=records.loc[records.target.eq(case['laboratory_target'])].set_index('TIMESTAMP').value
        view=workflow.view(source,read_json(source/'selected.json'),lab,case['rows'],'evaluation',target=case['laboratory_target'])
        table=view['table'].loc[pd.DatetimeIndex(case['evaluation']['timestamps'])].copy()
        for name,values in view['references'].items():table['reference: '+name]=values
        names=[name for name in table if name.startswith('reference: ')]
        original=min(names,key=lambda n:rmse(table.observed,table[n]))
        for date in table.index:
            remaining=table.drop(date)
            best=min(names,key=lambda n:rmse(remaining.observed,remaining[n]))
            error=rmse(remaining.observed,remaining.prediction)
            rows.append(dict(target=case['target'],omitted_date=str(date),
                **evaluation_metrics(remaining.observed,remaining.prediction),
                original_reference=original,strongest_remaining_reference=best,
                rmse_reduction_percent=100*(1-error/rmse(remaining.observed,remaining[best])),
                rmse_reduction_original_reference_percent=100*(1-error/rmse(remaining.observed,remaining[original]))))
    pd.DataFrame(rows).to_csv(args.output/'evaluation_date_influence.csv',index=False)
    save_json(args.output/'influence_provenance.json',dict(scope='One-date omissions from fixed evaluation forecasts; references reranked on remaining dates; no refitting or selection'))


def render_targeted_tables(source, out, cases=None):
    """Render supplied diagnostics without fitting or changing primary scores.

    ``cases`` gives the display order (``report.publication.display_order``); it must hold
    the same targets as the results bundle. Without it, rows follow the bundle.
    """
    from sparse_wq.report.publication import number, tex, minus
    source, out = Path(source), Path(out)
    if not source.exists():
        return
    expected = read_json(source / 'source_models.json')
    fields = expected['case_fields']
    stored = read_json(out / 'results.json')['targets']
    for case in stored:
        if {field: case[field] for field in fields} != expected['cases'][case['target']]:
            raise ValueError('Supplied targeted checks do not match the reported model/dates')
    if cases is None:
        cases = stored
    elif sorted(c['target'] for c in cases) != sorted(c['target'] for c in stored):
        raise ValueError('Display order must contain exactly the reported targets')
    for name, sha in expected['files'].items():
        if digest(source.parent / name) != sha:
            raise ValueError('Targeted-check source result changed: ' + name)
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)

    def write(name, caption, columns, heading, rows):
        content = [r'\begin{table}[H]', r'\caption{' + caption + '}', r'\scriptsize',
                   r'\begin{tabularx}{\textwidth}{' + columns + '}', r'\toprule ' + heading + r'\\\midrule',
                   *map(minus, rows), r'\bottomrule', r'\end{tabularx}', r'\end{table}']
        (tables / name).write_text('\n'.join(content) + '\n', encoding='utf-8')

    summary = source / 'censoring_summary.json'
    if summary.exists():
        records = pd.DataFrame(read_json(summary))
        # Ordered by the share of qualified below-DL results, the quantity the
        # substitution acts on; ties follow the display order.
        laboratory = pd.read_csv(out / 'laboratory.csv')
        censored = {case['target']: 100 * laboratory.loc[laboratory.target.eq(case['laboratory_target']), 'is_left_censored'].mean()
                    for case in cases}
        position = {case['target']: i for i, case in enumerate(cases)}
        rows = []
        groups = dict(list(records.loc[records.period.eq('evaluation')].groupby('target', sort=False)))
        for target in sorted(groups, key=lambda t: (-censored[t], position[t])):
            group = groups[target]
            half, alternative = [group.loc[group.treatment.eq(policy)].iloc[0] for policy in ('half', 'sqrt2')]
            if half['n'] != alternative['n']:
                raise ValueError('Censoring sensitivity dates/counts must match')
            values = [tex(half.target_name), f'{censored[target]:.1f}', str(half['n']), number(half.r2), number(alternative.r2),
                      f'{half.rmse:.4g}', f'{alternative.rmse:.4g}',
                      number(half.rmse_reduction_percent, 1), number(alternative.rmse_reduction_percent, 1)]
            rows.append(' & '.join(values) + r'\\')
        write('targeted_censoring.tex',
              r'Sensitivity of the main-analysis models to the value substituted for results below the '
              r'detection limit (DL). Selected configurations are held fixed, models are refitted and scored '
              r'on the same evaluation dates; ordered by the share of measurements below DL for each target '
              r'(Table~\ref{tab:target_metadata}). '
              r'Subscripts 2 and $\sqrt{2}$ denote substitution by DL/2 and DL/$\sqrt{2}$. '
              r'RMSE uses the target units in Table~\ref{tab:targets}; SS is percentage RMSE reduction '
              r'against the strongest of six references recalculated under each substitution. '
              r'Both the training target values and the scored measurements change, so lower RMSE across '
              r'substitutions does not itself establish better prediction.\label{tab:censoring_sensitivity}',
              'Xrrrrrrrr', r'Target & Below DL (\%) & $n$ & $R^2_2$ & $R^2_{\sqrt{2}}$ & RMSE$_2$ & RMSE$_{\sqrt{2}}$ & SS$_2$ (\%) & SS$_{\sqrt{2}}$ (\%)', rows)

    evaluation=source/'evaluation_date_influence.csv'
    if evaluation.exists():
        data=pd.read_csv(evaluation)
        rows=[]
        for case in cases:
            group=data.loc[data.target.eq(case['target'])]
            rows.append(tex(case['target_name'])+' & '+str(len(group))+' & ['+number(group.rmse_reduction_percent.min(),1)+', '+number(group.rmse_reduction_percent.max(),1)+']'+r'\\')
        write('date_influence.tex',r'Influence of omitting one evaluation date when scoring the fixed forecasts of the main-analysis models. The strongest reference is selected again after each omission. No fitting or model selection is repeated.\label{tab:date_influence}',
              'Xrr',r'Target & Evaluation $n$ & RMSE reduction range (\%)',rows)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=paths.DERIVED)
    parser.add_argument('--results', type=Path, default=paths.RESULTS)
    parser.add_argument('--output', type=Path, default=paths.OUTPUTS / 'targeted_checks')
    parser.add_argument('--check', choices=['censoring', 'influence', 'all'], default='all')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workflow.verify_supplied_results(args.results, args.prepared)
    cases = read_json(args.results / 'bundle/results.json')['targets']
    fields = ['target', 'run', 'method', 'variant', 'features', 'seeds', 'rows', 'validation', 'evaluation']
    files = []
    for case in cases:
        directory = source_directory(args, case)
        files.extend(directory / name for name in ['selected.json', 'validation.csv', 'assessment/fit.json', 'assessment/predictions.csv'])
    provenance = dict(case_fields=fields, cases={case['target']: {field: case[field] for field in fields} for case in cases},
                      files={file.relative_to(args.results).as_posix(): digest(file) for file in files})
    marker = args.output / 'source_models.json'
    if marker.exists() and read_json(marker) != provenance:
        raise ValueError('Source models changed; use a fresh targeted-check output directory')
    save_json(marker, provenance)
    if args.check in ('influence', 'all'):
        influence(args, cases)
    if args.check in ('censoring', 'all'):
        import torch
        torch.set_num_threads(1)
        store = NativeInputs(args.prepared, weather_cache=args.output / 'weather_cache')
        if args.check in ('censoring', 'all'):
            censoring(args, store, cases)
    save_json(args.output / ('completed_' + args.check + '.json'), dict(version=CHECK_VERSION,
        scope='No reselection or tuning; primary identities and numerical results are read-only.',
        source_bundle_sha256=digest(args.results / 'bundle/results.json'),
        implementation_sha256=digest(__file__)))


if __name__ == '__main__':
    main()

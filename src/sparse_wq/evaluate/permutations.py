"""Full-selection block permutations within specified target arms.

Original and rearranged records use the same complete chronological selection
procedure. Only the named arm is searched, on fixed dates and original budgets.
"""
from __future__ import annotations
import copy
import gc
import io
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .block_permutation import (atomic_text, digest, metrics, read_json,
    rearrange_records, run_lock, save_json, score_predictions)


def folder(out, target, repetition):
    return Path(out)/'jobs'/target/('original' if repetition == 0 else f'{repetition:06d}')


def paired_result(original, scored, dates, references):
    """Compare on dates predicted by both; keep lost coverage explicit."""
    fixed = pd.DatetimeIndex(dates)
    original, scored = original.reindex(fixed), scored.reindex(fixed)
    required = ['observed', 'prediction'] + ['reference: '+name for name in references]
    valid = np.isfinite(original[required]).all(axis=1) & np.isfinite(scored[required]).all(axis=1)
    return dict(n_matched=int(valid.sum()), n_expected=len(fixed), fixed_set_complete=bool(valid.all()),
        timestamps=list(map(str, fixed[valid])), missing_timestamps=list(map(str, fixed[~valid])),
        observed_matched_scores=metrics(original.loc[valid], references),
        matched_scores=metrics(scored.loc[valid], references))


def same_configuration(left, right):
    return (left.get('variant') == right.get('variant') and
            list(left.get('features', [])) == list(right.get('features', [])) and
            list(left.get('seeds', [])) == list(right.get('seeds', [])))


def cases(args):
    available = read_json(args.results/'bundle/results.json')['targets']
    if args.case:
        requested = []
        from ..design import RUNS
        for value in args.case:
            parts = value.split(':')
            if len(parts) != 2 or parts[1] not in RUNS:
                raise ValueError('Use target:run for an explicitly specified arm')
            target, run = parts
            base = next(row for row in available if row['target'] == target)
            requested.append(dict(base, run=run, method='off' if run.startswith('profiler_') else 'shared'))
        available = requested
    else:
        # Follow up the explicitly chosen detailed case study; --case permits
        # other target/predictor-set experiments without a numerical skill gate.
        available = [row for row in available if row['eligible_for_followup']]
    if args.target: available = [row for row in available if row['target'] in args.target]
    if len({row['target'] for row in available}) != len(available):
        raise ValueError('Use a separate output directory for multiple arms of one target')
    return available


def run(args, store):
    """Resume a fixed repetition count; no score-dependent stopping or allocation."""
    import torch
    from .. import fitting
    from ..data.native import NativeInputs
    from ..data.qualified import substitute
    from ..fitting import InsufficientCoverage
    from ..workflow import analysis, model_path, run_signature, configuration_name
    selected_cases = cases(args)
    if not selected_cases or args.repetitions < 1:
        raise ValueError('Specify at least one case and a positive repetition count')
    out = args.output/'permutations'
    out.mkdir(parents=True, exist_ok=True)
    fields = ['target','target_name','laboratory_target','run','method','rows',
              'evaluation','configuration','features','variant','scores']
    plan = dict(version=1, seed=4101, block_length=3, repetitions=args.repetitions,
        original_reselection=True, cases=[{key: row[key] for key in fields} for row in selected_cases],
        scientific_budgets=dict(subset_candidates=args.budget, tuning_trials=args.trials,
                                max_gp_epochs=args.epochs, finalist_seeds=6),
        signature=run_signature(args))
    marker = out/'experiment.json'
    if marker.exists() and read_json(marker) != plan:
        raise ValueError('Experiment settings changed; use a fresh output directory')
    save_json(marker, plan)
    fitting.write_json = save_json
    torch.set_num_threads(2)
    base = args.prepared
    records = pd.read_csv(base/'qualified_observations.csv', parse_dates=['TIMESTAMP'], float_precision='round_trip')
    with run_lock(out):
        # Original data first, then equal numbered repetitions across targets.
        for repetition in range(args.repetitions+1):
            for case in selected_cases:
                target = case['target']; destination = folder(out, target, repetition)
                destination.mkdir(parents=True, exist_ok=True)
                if (destination/'result.json').exists(): continue
                prepared = destination/'prepared'
                if not prepared.exists(): shutil.copytree(base, prepared)
                changed, mapping, order = (records, None, None)
                if repetition:
                    changed, mapping, order = rearrange_records(records, case['laboratory_target'], 3, plan['seed'], repetition)
                    changed.to_csv(prepared/'qualified_observations.csv', index=False)
                    mapping.to_csv(destination/'rearrangement.csv', index=False)
                    save_json(destination/'rearrangement.json', dict(seed=plan['seed'], repetition=repetition, block_order=order))
                # Match the published experiment's resource settings without a
                # background process controller or machine-specific snapshots.
                import yaml
                settings = yaml.safe_load((prepared/'search.yml').read_text(encoding='utf-8'))
                settings['models']['xgb']['hyperparameters']['n_jobs'] = 2
                (prepared/'search.yml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
                provenance = read_json(base/'preparation.json')
                for name in ['qualified_observations.csv','search.yml']:
                    provenance['outputs_sha256'][name] = digest(prepared/name)
                if repetition: provenance['block_permutation'] = dict(seed=plan['seed'], repetition=repetition, block_order=order)
                save_json(prepared/'preparation.json', provenance)
                local = copy.copy(args); local.prepared = prepared
                inputs = NativeInputs(prepared, weather_cache=destination/'weather_cache')
                model = analysis(local, inputs, destination/'models', case['laboratory_target'], case['run'], case['method'])
                published = model_path(args.results/'models', target, case['run'], case['method'])
                identity = read_json(published/'identity.json')
                if (model.rows != case['rows'] or list(map(str, model.train)) != identity['train']
                        or list(map(str, model.test)) != identity['test']):
                    raise ValueError('Chronological memberships or input-window length changed')
                result = dict(target=target, repetition=repetition, run=case['run'], fit_status='insufficient_coverage')
                try:
                    selected = read_json(model.out/'selected.json') if (model.out/'selected.json').exists() else model.search()
                    result.update(selected={key: selected[key] for key in ['features','variant','seeds','objective']},
                                  configuration=configuration_name(selected['variant'], model.settings))
                    assessment = model.out/'assessment'
                    if (assessment/'fit.json').exists() and (assessment/'predictions.csv').exists():
                        predictions = pd.read_csv(assessment/'predictions.csv', float_precision='round_trip')
                    else: predictions, _ = model.assess(selected)
                    scored = score_predictions(predictions, read_json(assessment/'fit.json'), model.lab.value, model.rows, target=model.target, nominal_training=model.train)
                    scored.to_csv(destination/'predictions_and_references.csv', index_label='timestamp')
                    dates, references = case['evaluation']['timestamps'], list(case['scores']['baselines'])
                    if repetition:
                        original = pd.read_csv(folder(out,target,0)/'predictions_and_references.csv',
                            parse_dates=['timestamp'], float_precision='round_trip').set_index('timestamp')
                    else:
                        original = scored
                        lab = substitute(records,'half')
                        lab = lab.loc[lab.target.eq(case['laboratory_target'])].set_index('TIMESTAMP').value
                        previous = score_predictions(pd.read_csv(published/'assessment/predictions.csv',float_precision='round_trip'),
                            read_json(published/'assessment/fit.json'), lab, model.rows, target=case['laboratory_target'], nominal_training=identity['train'])
                        result['reported_model_comparison'] = paired_result(previous, scored, dates, references)
                        result['same_reported_configuration'] = same_configuration(selected,read_json(published/'selected.json'))
                    result.update(paired_result(original,scored,dates,references),fit_status='complete')
                except InsufficientCoverage as error:
                    if repetition == 0: raise
                    result.update(reason=str(error),n_matched=0,n_expected=len(case['evaluation']['timestamps']),
                        fixed_set_complete=False,observed_matched_scores=None,matched_scores=None)
                save_json(destination/'result.json',result)
                refresh_report(out)
                del model, inputs
                gc.collect()
    refresh_report(out)


def refresh_report(out, plan=None):
    """A reader lock on a derived report must not stop later expensive fits."""
    try: report(out)
    except PermissionError as error:
        print('Report refresh deferred: '+str(error),flush=True)
        return str(error)
    return None


def report(out):
    """Recompute summaries/tables from retained forecast sets; no model fitting."""
    out=Path(out); plan=read_json(out/'experiment.json')
    rows=[];references=[];summaries=[];tables=[]
    completed={case['target']:{int(p.parent.name):read_json(p)
        for p in (out/'jobs'/case['target']).glob('[0-9]*/result.json')} for case in plan['cases']}
    balanced=0
    while balanced < plan['repetitions'] and all(balanced+1 in jobs for jobs in completed.values()): balanced+=1
    lines=['# Block-permutation experiment','',
        'Original and rearranged records repeat complete selection within each specified predictor set. '
        'Three consecutive laboratory observations move together across fixed dates; environmental records remain fixed. '
        'Laboratory priors, differential responses and six references are rebuilt.','',
        f'{balanced} completed numbered rearrangements per target; {plan["repetitions"]} specified independently of scores.','',
        'Each comparison uses matching primary evaluation dates. Improvement (%) is '
        '100 × (1 − model RMSE / lowest RMSE among six references), computed separately for each record. '
        'These are descriptive comparisons conditional on the specified predictor sets and exchangeable observation blocks.','']
    for case in plan['cases']:
        target=case['target']; jobs=completed[target]
        original_path=folder(out,target,0)/'result.json'
        original=read_json(original_path) if original_path.exists() else None
        eligible=[jobs[n] for n in range(1,balanced+1)]
        finite=[r for r in eligible if (r.get('matched_scores') or {}).get('r2') is not None
                and (r.get('observed_matched_scores') or {}).get('r2') is not None]
        skilled=[r for r in eligible if (r.get('matched_scores') or {}).get('skill') is not None
                 and (r.get('observed_matched_scores') or {}).get('skill') is not None]
        higher_r2=sum(r['observed_matched_scores']['r2']>r['matched_scores']['r2'] for r in finite)
        higher_skill=sum(r['observed_matched_scores']['skill']>r['matched_scores']['skill'] for r in skilled)
        summaries.append(dict(target=target,name=case['target_name'],completed=len(jobs),balanced=balanced,
            assessed=len(finite),original_r2_higher=higher_r2,skill_assessed=len(skilled),original_skill_higher=higher_skill,original=original))
        lines += ['## '+case['target_name'],'',case['run'],'',
            f'Original R² higher in {higher_r2}/{len(finite)} comparisons; original improvement against references higher in {higher_skill}/{len(skilled)}.','',
            '| Repetition | Shared / primary dates | Original R² | Rearranged R² | Original RMSE | Rearranged RMSE | Original improvement (%) | Rearranged improvement (%) | Selected model |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        def number(value):return 'undefined' if value is None else f'{value:.4g}'
        for repetition,result in sorted(jobs.items()):
            a,b=result.get('observed_matched_scores') or {},result.get('matched_scores') or {}
            row=dict(target=target,repetition=repetition,balanced=repetition<=balanced,n=result.get('n_matched',0),
                n_expected=result.get('n_expected'),fixed_set_complete=result.get('fixed_set_complete'),fit_status=result['fit_status'],
                original_r2=a.get('r2'),rearranged_r2=b.get('r2'),original_rmse=a.get('rmse'),rearranged_rmse=b.get('rmse'),
                original_skill_percent=100*a['skill'] if a.get('skill') is not None else None,
                rearranged_skill_percent=100*b['skill'] if b.get('skill') is not None else None,
                original_best_reference=a.get('best_baseline'),rearranged_best_reference=b.get('best_baseline'),
                configuration=result.get('configuration'),predictors='; '.join(result.get('selected',{}).get('features',[])))
            rows.append(row)
            lines.append(f'| {repetition} | {row["n"]}/{row["n_expected"]} | '+
                ' | '.join(number(row[key]) for key in ['original_r2','rearranged_r2','original_rmse','rearranged_rmse','original_skill_percent','rearranged_skill_percent'])+
                ' | '+str(row['configuration'])+' |')
            for label,scores in [('original',a),('rearranged',b)]:
                for name,score in scores.get('baselines',{}).items():
                    references.append(dict(target=target,repetition=repetition,record=label,reference=name,**score))
        def span(key,collection):
            values=[r[collection][key] for r in eligible if r.get(collection) and r[collection].get(key) is not None]
            if not values:return '--'
            scale,precision=(100,1) if key=='skill' else (1,3)
            def f(v):
                text=f'{v*scale:.{precision}f}'
                return text.lstrip('-') if float(text)==0 else text
            return f'${f(values[0])}$' if min(values) == max(values) else f'$[{f(min(values))}, {f(max(values))}]$'
        if eligible:
            tables.append(f"{case['target_name']} & {balanced} & {span('r2','observed_matched_scores')} & {span('r2','matched_scores')} & {span('skill','observed_matched_scores')} & {span('skill','matched_scores')}\\\\")
    destination=out/'report';destination.mkdir(exist_ok=True)
    atomic_text(destination/'REPORT.md','\n'.join(lines)+'\n')
    save_json(destination/'summary.json',dict(balanced_repetitions=balanced,targets=summaries,
        inferential_status='descriptive; conditional-arm and changing coverage'))
    for name,records in [('comparisons.csv',rows),('references.csv',references)]:
        pd.DataFrame(records).to_csv(destination/name,index=False)
    tex=(r'\begin{table}[!htbp]'+'\n'+
        r'\caption{Block-permutation results. Both original and rearranged records repeat selection within the specified predictor set. SS is percentage RMSE reduction against the strongest reference, computed separately for each record. Ranges use matching dates within each comparison. These comparisons are descriptive and conditional on that predictor set.\label{tab:block_permutation}}'+'\n'+
        r'\small\begin{tabularx}{\textwidth}{Xrrrrr}\toprule Target & Permutations & \makecell{$R^2$\\Original} & \makecell{$R^2$\\Permuted} & \makecell{SS (\%)\\Original} & \makecell{SS (\%)\\Permuted}\\\midrule'+'\n'+
        '\n'.join(tables)+'\n'+r'\bottomrule\end{tabularx}\end{table}'+'\n')
    atomic_text(destination/'block_permutation.tex',tex)

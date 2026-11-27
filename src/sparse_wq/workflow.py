"""One workflow: native preparation, chronological selection, refitting and reporting."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from . import paths
from .analysis import Analysis
from .data.prepare import prepare
from .data.native import NativeInputs,digest
from .data.qualified import substitute,evaluation_metrics
from .evaluate.forecasts import reference_forecasts
from .evaluate.seed_coverage import collect_seed_predictions
from .evaluate.block_permutation import atomic_text,read_json,save_json,REFERENCE_NAMES
from .evaluate import evidence
from .report.publication import interval_and_tests,render,r2_interval,fitted_predictors


from .design import ARM_KEYS
RUNS=[run for run,_ in ARM_KEYS]
# Detailed case study chosen during exploratory development, not a skill threshold.
CASE_STUDY_TARGETS = frozenset({'total_coliforms_37c'})


def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level',type=int,choices=[1,2,3])
    ap.add_argument('--run',nargs='+',choices=RUNS,default=RUNS)
    ap.add_argument('--target',nargs='+')
    ap.add_argument('--prepared',type=Path,default=paths.DERIVED)
    ap.add_argument('--results',type=Path,default=paths.RESULTS)
    ap.add_argument('--output',type=Path,default=paths.OUTPUTS)
    ap.add_argument('--trials',type=int,default=250)
    ap.add_argument('--epochs',type=int,default=250)
    ap.add_argument('--budget',type=int,default=240)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--repetitions',type=int,default=1)
    ap.add_argument('--case',action='append',help='Permutation override: target:run; repeat for more cases')
    return ap


def protocol(args,targets):
    return dict(targets=targets,assessment_role='chronological_later_evaluation',
        input_policy='original measurements without simulated sensor error',
        tuning_trials=args.trials,gp_epochs=args.epochs,finalist_seeds=6,device=args.device,
        prediction_column='prediction',treatment='half',horizon=0,
        search=dict(subset_budget=args.budget,beam_width=6,rounds=10,swaps=60,min_features=4,
            final_top_k=4,seed=42,coverage_penalty=.25,minimum_validation=5),
        search_allocation='elimination depths and reserved swaps',
        input_replication='one original window per laboratory observation', output_constraint='nonnegative-measured-level-v1',
        budget_class='study' if (args.trials,args.epochs,args.budget)==(250,250,240) else 'development check')


def run_signature(args):
    import importlib.metadata
    package=Path(__file__).parent
    return dict(preparation_sha256=digest(args.prepared/'preparation.json'),
        code_sha256={str(file.relative_to(package)):digest(file) for file in package.rglob('*.py')},
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','gpytorch','xgboost','scikit-learn','optuna']},
        protocol=protocol(args,[]))


def verify_supplied_results(results,prepared):
    manifest=Path(results)/'configuration_hashes.json'
    if manifest.exists():
        for case in read_json(manifest):
            directory=model_path(Path(results)/'models',case['target'],case['run'],case['method'])
            for name,key in [('selected.json','selected_sha256'),('assessment/fit.json','fit_sha256'),('assessment/predictions.csv','predictions_sha256')]:
                if digest(directory/name)!=case[key]:raise ValueError('Supplied scientific result changed: '+str(directory/name))
    expected=Path(results)/'reference_input_hashes.json'
    if expected.exists():
        for name,sha in read_json(expected).items():
            # CSV/YAML text is identical across platforms after LF normalization.
            # Raw sources and supplied forecasts retain strict byte checksums.
            data=(Path(prepared)/name).read_bytes().replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=sha:
                raise ValueError('Prepared data do not match the supplied fits: '+name)


def slots(args,store):
    targets=store.manifest['study']['targets']
    unknown=set(args.target or [])-set(targets.values())
    if unknown:raise ValueError('Unknown targets: '+', '.join(sorted(unknown)))
    for name,target in targets.items():
        if args.target and target not in args.target:continue
        for run in args.run:
            method=dict(ARM_KEYS)[run]
            yield name,target,run,method


def analysis(args,store,root,name,run,method):
    return Analysis(args.prepared,None,name,run,
        protocol=protocol(args,list(store.manifest['study']['targets'].values())),uncertainty=False,
        trials=args.trials,epochs=args.epochs,
        device=args.device,output=Path(root)/method,inputs=store)


def model_path(root,target,run,method):return Path(root)/method/'half'/run/target


def configuration_name(variant,settings):
    names={'none':'flattened hourly inputs','mean':'full-window means','stats:12h':'12-hour summaries','stats:24h':'24-hour summaries'}
    if variant.startswith('xgb'):return 'XGBoost ('+names[settings['models']['xgb']['variants'][variant]]+')'
    if variant.startswith('gp'):
        kernel,representation=settings['models']['gp']['variants'][variant]
        return 'Gaussian process ('+('Matérn 5/2 + linear' if kernel=='matern52+linear' else 'Matérn 5/2')+'; '+names[representation]+')'
    return 'Multiple linear regression ('+{'mlr':'latest input','mlr_avg12':'preceding 12-hour mean','mlr_avgall':'full-window means'}[variant]+')'


def validation_path(directory,selected):
    if (directory/'validation.csv').exists():return directory/'validation.csv'
    info=selected['info']
    settings={name:info[name] for name in ['coverage_penalty','minimum_observations','prediction_column','uncertainty','hyper_override']}
    settings.update(features=selected['features'],variant=selected['variant'],seeds=selected['seeds'])
    token=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:16]
    return directory/'validation'/(token+'.csv')


def view(directory,selected,lab,rows,period,*,target):
    if period=='validation':
        table=pd.read_csv(validation_path(directory,selected),parse_dates=['timestamp'],float_precision='round_trip').set_index('timestamp').sort_index()
        references=pd.DataFrame(index=table.index,columns=list(REFERENCE_NAMES.values()),dtype=float)
        details=selected['info']['details']
        from .data.qualified import expanding_folds
        nominal=pd.DatetimeIndex(read_json(directory/'identity.json')['train'])
        folds=list(expanding_folds(nominal))
        for fold,part in table.groupby('fold'):
            training=folds[int(fold)-1][0]
            forecasts=reference_forecasts(lab,training,part.index,rows,target=target)
            table.loc[part.index,'prior']=forecasts['Naive']
            for name,value in forecasts.items():references.loc[part.index,REFERENCE_NAMES[name]]=value
    else:
        table=pd.read_csv(directory/'assessment/predictions.csv',parse_dates=['timestamp'],float_precision='round_trip').set_index('timestamp').sort_index()
        details=read_json(directory/'assessment/fit.json')['details']
        training=pd.DatetimeIndex(read_json(directory/'identity.json')['train'])
        forecasts=reference_forecasts(lab,training,table.index,rows,target=target,prior=table.prior)
        references=pd.DataFrame(forecasts,index=table.index).rename(columns=REFERENCE_NAMES)
    eligible=[name for name in references if np.isfinite(references[name]).any()]
    valid=np.isfinite(table[['observed','prediction']]).all(axis=1)&np.isfinite(references[eligible]).all(axis=1)
    return dict(table=table,references=references,eligible=eligible,available=table.index[valid],details=details)


def member_dates(members):
    return pd.to_datetime([member.removeprefix('segment_').removesuffix('.csv') for member in sorted(members)],format='%Y%m%d%H')


def score(view,dates):
    if len(dates)<2:return None
    frame=view['table'].loc[dates];y,p=frame.observed.to_numpy(),frame.prediction.to_numpy()
    result=evaluation_metrics(y,p)
    refs={name:evaluation_metrics(y,view['references'].loc[dates,name].to_numpy()) for name in view['eligible']}
    best=min(refs,key=lambda name:(refs[name]['rmse'],name))
    percent=100*(result['rmse']/refs[best]['rmse']-1) if refs[best]['rmse']>0 else None
    return dict(**result,baselines=refs,best_baseline=best,baseline_error_percent=percent,
        eligible=len(refs),beaten=sum(result['rmse']<value['rmse'] for value in refs.values()))


def compare(views):
    common=None
    for item in views:
        common=item['available'] if common is None else common.intersection(item['available'])
    for item in views:
        item['common']=score(item,common);item['own']=score(item,item['available'])
    return common


def choices(args,root,store,destination=None):
    """Primary predictor set is fixed; model selection occurs within each arm."""
    groups={}
    for name,target,run,method in slots(args,store):
        if run!='noprofiler_differential':continue
        directory=model_path(root,target,run,method)
        selected=read_json(directory/'selected.json')
        path=validation_path(directory,selected)
        table=pd.read_csv(path,parse_dates=['timestamp'])
        groups[target]=dict(run=run,method=method,timestamps=list(map(str,table.timestamp)),
                           r2=selected['info']['metrics']['r2'])
    if destination is not None:save_json(destination,groups)
    return groups


def load_records(prepared):
    return substitute(pd.read_csv(Path(prepared)/'qualified_observations.csv',parse_dates=['TIMESTAMP'],float_precision='round_trip'),'half')


def fixed_fit(model,selected,saved_fit,horizon=0):
    from .evaluate.targeted_checks import settings
    parameters={detail['seed']:settings(detail,selected['variant']) for detail in saved_fit['details']}
    if not set(selected['seeds'])<=parameters.keys():raise ValueError('Missing saved final hyperparameters')
    aligned,raw,details,coverage=collect_seed_predictions(
        lambda seed:model.predict(model.train,model.test,selected['features'],selected['variant'],
            seed=seed,horizon=horizon,hyper_override=copy.deepcopy(parameters[seed])),
        seeds=selected['seeds'],expected=model.test,prediction_columns=['prediction','prediction_unclipped'])
    if aligned.empty:raise ValueError('No common seed predictions')
    table=aligned.groupby('timestamp',as_index=False).agg(observed=('observed','first'),prediction=('prediction','mean'),
        prediction_unclipped=('prediction_unclipped','mean'),prior=('prior','first'))
    destination=model.out/'assessment' if horizon==0 else model.out/'horizons'/f'h{horizon:03d}'
    destination.mkdir(parents=True,exist_ok=True)
    raw.to_csv(destination/'seed_predictions.csv',index=False);table.to_csv(destination/'predictions.csv',index=False)
    save_json(destination/'fit.json',dict(details=details,coverage=coverage))
    return table,details


def assemble(args,root,store):
    """Rebuild the derived bundle from saved predictions and independent records."""
    root=Path(root);destination=args.output/'analysis';destination.mkdir(parents=True,exist_ok=True)
    chosen=choices(args,root,store,destination/'reporting_choices.json');records=load_records(args.prepared)
    display=read_json(paths.CONFIG/'targets.json')
    settings=yaml.safe_load((args.prepared/'search.yml').read_text(encoding='utf-8'))
    groups={};memberships=[];all_rows=[];all_scores=[];cases=[]
    for name,target,run,method in slots(args,store):
        directory=model_path(root,target,run,method);selected=read_json(directory/'selected.json')
        identity=read_json(directory/'identity.json');lab=records.loc[records.target.eq(name)].set_index('TIMESTAMP').sort_index()
        models={period:view(directory,selected,lab.value,identity['rows'],period,target=name) for period in ['validation','evaluation']}
        for item in models.values():item.update(run=run,method=method,selected=selected,directory=directory,identity=identity,name=name)
        groups.setdefault(target,[]).append(models)
    for target,models in groups.items():
        dates={period:compare([model[period] for model in models]) for period in ['validation','evaluation']}
        primary=next(model for model in models if (model['validation']['run'],model['validation']['method'])==(chosen[target]['run'],chosen[target]['method']))
        for period in ['validation','evaluation']:
            for model in models:
                item=model[period];table=item['table'].copy()
                for name in item['references']:table['reference: '+name]=item['references'][name]
                table['primary_comparison']=table.index.isin(dates[period])
                table['target'],table['run'],table['method'],table['period']=target,item['run'],item['method'],period
                all_rows.append(table.reset_index())
                all_scores.append(dict(target=target,period=period,run=item['run'],method=item['method'],
                    selected=item is primary[period],common=item['common'],all_available=item['own'],
                    r2_interval=r2_interval(item['table'].loc[dates[period],'observed'].to_numpy(),item['table'].loc[dates[period],'prediction'].to_numpy())))
        selected=primary['evaluation'];spec=selected['selected'];identity=selected['identity'];scores=selected['own']
        frame=selected['table'].loc[selected['available']];refs=selected['references'].loc[frame.index]
        laboratory=records.loc[records.target.eq(selected['name'])].set_index('TIMESTAMP').sort_index()
        eligible=target in CASE_STUDY_TARGETS
        details=selected['details'];members={member for detail in details for member in detail['training_observations']}
        gain=np.square(frame.observed.to_numpy()-refs[scores['best_baseline']].to_numpy())-np.square(frame.observed.to_numpy()-frame.prediction.to_numpy())
        configuration=configuration_name(spec['variant'],settings)
        case=dict(target=target,target_name=display[target]['target_name'],laboratory_target=selected['name'],unit=display[target]['unit'],
            run=selected['run'],method=selected['method'],features=spec['features'],fitted_features=fitted_predictors(details),variant=spec['variant'],seeds=spec['seeds'],configuration=configuration,
            rows=identity['rows'],scores=scores,all_available_scores=selected['own'],eligible_for_followup=eligible,
            validation=dict(timestamps=list(map(str,primary['validation']['available'])),scores=primary['validation']['own']),
            evaluation=dict(timestamps=list(map(str,selected['available'])),scores=scores),
            counts=dict(n_train=len(members),n_train_min=min(len(d['training_observations']) for d in details),n_train_max=max(len(d['training_observations']) for d in details),
                n_nominal_train=len(identity['train']),n_nominal_evaluation=len(identity['test']),n_common_validation=len(primary['validation']['available']),n_paired_validation=len(dates['validation']),n_paired_evaluation=len(dates['evaluation']),
                n_available_validation=len(primary['validation']['table']),n_common_evaluation=len(frame),n_available_evaluation=len(selected['available']),
                n_censored_evaluation=int(laboratory.loc[frame.index,'is_left_censored'].sum()),censored_percent=100*laboratory.loc[frame.index,'is_left_censored'].mean()),
            support=interval_and_tests(frame,refs,scores,evidence),
            squared_error_gain=dict(net=gain.sum(),top_three=np.sort(gain)[-3:].sum(),top_three_fraction=np.sort(gain)[-3:].sum()/gain.sum() if gain.sum() else None))
        cases.append(case);folder=destination/'cases'/target;folder.mkdir(parents=True,exist_ok=True)
        frame=frame.copy()
        for name in refs:frame['reference: '+name]=refs[name]
        for name in ['is_left_censored','reported_number','limit','source_row']:frame[name]=laboratory.loc[frame.index,name]
        frame.to_csv(folder/'evaluation.csv',index_label='timestamp')
        for filename in ['selected.json','assessment/fit.json']:shutil.copyfile(selected['directory']/filename,folder/Path(filename).name)
        for phase,values in [('final',details),('validation',primary['validation']['details'])]:
            memberships.extend(dict(target=target,phase=phase,fold=detail.get('fold'),seed=detail['seed'],member=member) for detail in values for member in detail['training_observations'])
    predictors=store.manifest['study']['predictors']
    save_json(destination/'results.json',dict(selection_rule='Primary no-profiler arm fixed; family, predictors and representation selected within each arm by chronological training validation',targets=cases,
        eligible_targets=[case['target'] for case in cases if case['eligible_for_followup']],predictor_channels=predictors['always']+predictors['profiler']))
    save_json(destination/'all_arm_scores.json',all_scores)
    pd.concat(all_rows,ignore_index=True).to_csv(destination/'all_arm_predictions.csv',index=False)
    pd.DataFrame(memberships).to_csv(destination/'training_memberships.csv',index=False)
    records.to_csv(destination/'laboratory.csv',index=False)
    # Stored horizon fits are included only when their zero-hour selected identity
    # is the same; new searches do not inherit unrelated horizon conclusions.
    from .evaluate.compatibility import copy_horizon_scores
    copy_horizon_scores(args.results,root,destination,cases)
    render(destination, results=args.results, models=root)
    return destination


def stage_main(stage=None):
    import torch
    torch.set_num_threads(1)
    ap=parser();args=ap.parse_args()
    if stage is None and args.level is None:ap.error('--level is required')
    args.output.mkdir(parents=True,exist_ok=True)
    if stage=='report' or (stage is None and args.level==3):
        if set(args.run)!=set(RUNS):ap.error('Reporting requires both predictor sets; --run restricts fitting runs only')
        prepare(paths.RAW,paths.CONFIG,args.prepared)
        verify_supplied_results(args.results,args.prepared)
        store=NativeInputs(args.prepared,weather_cache=args.output/'weather_cache')
        bundle=assemble(args,args.results/'models',store)
        from .report.records import environmental
        environmental(paths.RAW,args.prepared,bundle)
        return
    prepare(paths.RAW,paths.CONFIG,args.prepared)
    if stage in ['prepare','windows']:return
    store=NativeInputs(args.prepared,weather_cache=args.output/'weather_cache')
    if stage in ['refit','horizons','permutations','diagnostics'] or args.level==2:
        verify_supplied_results(args.results,args.prepared)
    if stage=='record-figures':
        # Record plots/tables are rendered from the same supplied qualified/QC
        # observations; no model fitting or fitting-dependent decisions occur.
        destination=assemble(args,args.results/'models',store)
        from .report.records import environmental
        environmental(paths.RAW,args.prepared,destination)
        return
    if stage=='permutations':
        from .evaluate.permutations import run
        run(args,store);return
    if stage=='horizons':
        from .evaluate.horizons import run
        run(args,store);return
    if stage=='diagnostics':
        import matplotlib.pyplot as plt
        for name,target,run,method in slots(args,store):
            source=model_path(args.results/'models',target,run,method)
            frame=pd.read_csv(source/'assessment/predictions.csv',parse_dates=['timestamp'])
            folder=args.output/'diagnostics'/target/run/method;folder.mkdir(parents=True,exist_ok=True)
            fig,ax=plt.subplots(figsize=(8,3))
            ax.plot(frame.timestamp,frame.observed,'o-',label='Measured')
            ax.plot(frame.timestamp,frame.prediction,'s-',label='Selected prediction')
            ax.set_title(target+'; '+run+'; '+method);ax.legend();fig.tight_layout()
            fig.savefig(folder/'predictions.png',dpi=180);plt.close(fig)
        return
    root=args.output/'models' if stage=='search' or args.level==1 else args.output/'refit'
    for name,target,run,method in slots(args,store):
        model=analysis(args,store,root,name,run,method)
        if stage=='search' or args.level==1:
            selected=read_json(model.out/'selected.json') if (model.out/'selected.json').exists() else model.search()
        else:
            source=model_path(args.results/'models',target,run,method)
            selected=read_json(source/'selected.json');shutil.copyfile(source/'selected.json',model.out/'selected.json')
            shutil.copyfile(validation_path(source,selected),model.out/'validation.csv')
    if len(args.run)==2:choices(args,root,store,root/'reporting_choices.json')
    if stage=='search':return
    for name,target,run,method in slots(args,store):
        model=analysis(args,store,root,name,run,method);selected=read_json(model.out/'selected.json')
        if (model.out/'assessment/fit.json').exists() and (model.out/'assessment/predictions.csv').exists():continue
        if stage=='refit' or args.level==2:
            source=model_path(args.results/'models',target,run,method)
            fixed_fit(model,selected,read_json(source/'assessment/fit.json'))
        else:model.assess(selected)
    if len(args.run)==2:
        bundle=assemble(args,root,store)
        from .report.records import environmental
        environmental(paths.RAW,args.prepared,bundle)

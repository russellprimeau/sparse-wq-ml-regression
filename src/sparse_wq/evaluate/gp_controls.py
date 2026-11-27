"""Fixed Matérn-5/2 GP controls: held prior, with optional calendar terms.

Chronological nested folds select optimization duration only. Both controls
use exactly the environmental GP's fitted training/evaluation observations.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy
import numpy as np
import pandas as pd
from .block_permutation import read_json as read,save_json as save
def now(): return datetime.now(timezone.utc).isoformat()

def reference_table(table, lab, training, rows, target, horizon=0):
    from sparse_wq.evaluate.forecasts import reference_forecasts
    from sparse_wq.evaluate.block_permutation import REFERENCE_NAMES
    result = table.copy()
    for name, values in reference_forecasts(lab, training, result.index, rows,
            target=target, prior=result.prior, horizon=horizon).items():
        result['reference: ' + REFERENCE_NAMES[name]] = values
    return result

def run(args):
    import torch
    torch.set_num_threads(2)
    from sparse_wq import workflow
    out=args.output/'gp_controls'
    out.mkdir(parents=True,exist_ok=True)
    case=next(c for c in read(args.results/'bundle/results.json')['targets'] if c['target']=='total_coliforms_37c')
    source=workflow.model_path(args.results/'models',case['target'],case['run'],case['method'])
    identity=read(source/'identity.json')
    entry=case
    records=workflow.load_records(args.prepared)
    lab=records.loc[records.target.eq(case['laboratory_target'])].set_index('TIMESTAMP').value.sort_index()
    import yaml
    from sparse_wq.models import gp
    from sparse_wq.models.samples import Window
    from sparse_wq.data.qualified import expanding_folds
    from sparse_wq.search.gp_budget import choose_steps
    from sparse_wq.evaluate.forecasts import prior_values, calendar_columns
    from sparse_wq.evaluate.block_permutation import metrics
    training, evaluation = pd.DatetimeIndex(identity['train']), pd.DatetimeIndex(identity['test'])
    settings = yaml.safe_load((args.prepared / 'search.yml').read_text(encoding='utf-8'))
    base_hyper = copy.deepcopy(settings['models']['gp']['hyperparameters'])
    base_hyper.update(kernel='matern52', num_epochs=250, random_state=0)

    def data(stamps, calendar):
        stamps = pd.DatetimeIndex(stamps)
        prior = prior_values(lab, stamps, identity['rows'])
        valid = np.isfinite(prior)
        stamps, prior = stamps[valid], prior[valid]
        x = prior[:, None]
        if calendar:
            x = np.column_stack((x, calendar_columns(stamps)))
        return stamps, x, lab.loc[stamps].to_numpy() - prior, prior

    def select_steps(stamps, calendar):
        folds = []
        for tr, va in expanding_folds(pd.DatetimeIndex(stamps)):
            _, xt, yt, _ = data(tr, calendar)
            _, xv, yv, _ = data(va, calendar)
            if len(xt) < 6 or not len(xv):
                continue
            folds.append(dict(x_train=xt,y_train=yt,x_validation=xv,y_validation=yv,
                              target_span=1.,deltas=None))
        steps, history = choose_steps(folds, base_hyper, args.device, seed=0)
        return steps, history

    def predict(tr, va, calendar, steps):
        ti, xt, yt, _ = data(tr, calendar)
        vi, xv, yv, prior = data(va, calendar)
        names = ['held_prior'] + (['calendar_sine','calendar_cosine'] if calendar else [])
        windows = [Window(x.reshape(1,-1),np.array([[y]]),f'segment_{t:%Y%m%d%H}.csv') for t,x,y in zip(ti,xt,yt)]
        validation = [Window(x.reshape(1,-1),np.array([[y]]),f'segment_{t:%Y%m%d%H}.csv') for t,x,y in zip(vi,xv,yv)]
        hyper = dict(base_hyper, revision_fixed_steps=steps)
        fitted = gp.fit(hyper, windows, names, 1, 'mean', {}, None, args.device)
        delta = gp.predict(fitted, windows, validation)[0][:,0]
        unbounded = delta + prior
        result = pd.DataFrame(dict(observed=lab.loc[vi].to_numpy(),prediction=np.maximum(unbounded,0.),
                                  prediction_unclipped=unbounded,prior=prior),index=vi)
        result.index.name = 'timestamp'
        detail = dict(steps=steps, hyperparameters=hyper, features=names, seed=0,
                      training_observations=list(map(str,ti)),evaluation_observations=list(map(str,vi)),
                      preprocessing='GP input/response standardization fitted on training observations only',
                      response='change from window-start prior', output_constraint='zero floor after reconstructing level')
        return result, detail

    results=[]
    for calendar, name in [(False,'prior_only'),(True,'prior_and_calendar')]:
        folder=out/'controls'/name
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists():
            results.append(read(folder/'result.json'));continue
        save(out/'controls_status.json',dict(status='running',control=name,started_utc=now()))
        validations=[];details=[]
        for fold,(tr,va) in enumerate(expanding_folds(training),1):
            steps,history=select_steps(tr,calendar)
            table,detail=predict(tr,va,calendar,steps)
            detail.update(fold=fold,budget_history=history)
            validations.append(table.assign(fold=fold));details.append(detail)
            table.to_csv(folder/f'validation_fold_{fold}.csv')
            save(folder/f'validation_fold_{fold}.json',detail)
        validation=pd.concat(validations)
        validation.to_csv(folder/'validation.csv')
        steps,history=select_steps(training,calendar)
        table,detail=predict(training,evaluation,calendar,steps)
        expected=read(source/'assessment/fit.json')['details'][0]['training_observations']
        actual=[f'segment_{pd.Timestamp(t):%Y%m%d%H}.csv' for t in detail['training_observations']]
        if sorted(actual)!=sorted(expected):
            raise ValueError('Control and environmental GP training observations differ')
        if not table.index.equals(evaluation):
            raise ValueError('Control must predict all original evaluation dates')
        table=reference_table(table,lab,training,identity['rows'],entry['laboratory_target'])
        refs=[c.removeprefix('reference: ') for c in table if c.startswith('reference: ')]
        result=dict(control=name,completed_utc=now(),scores=metrics(table,refs),
                    validation_r2=1-float(np.square(validation.observed-validation.prediction).sum()/np.square(validation.observed-validation.observed.mean()).sum()),
                    n_validation=len(validation),training=len(actual),n_evaluation=len(table),steps=steps,
                    selection='Nested chronological training-only optimization-duration selection; fixed Matern-5/2 GP; no subset or family search')
        table.to_csv(folder/'predictions_and_references.csv')
        save(folder/'fit.json',dict(details=[detail],budget_history=history,validation_details=details))
        save(folder/'result.json',result);results.append(result)
        print(name,result,flush=True)
    save(out/'controls.json',results)
    save(out/'controls_status.json',dict(status='complete',completed_utc=now(),controls=len(results)))

def main():
    import argparse
    from sparse_wq import paths
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=paths.DERIVED)
    parser.add_argument('--results',type=Path,default=paths.RESULTS)
    parser.add_argument('--output',type=Path,default=paths.OUTPUTS)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    run(parser.parse_args())

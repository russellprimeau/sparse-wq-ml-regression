"""Refit qualifying reported configurations at fixed nominal input horizons."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .block_permutation import read_json,save_json,metrics,REFERENCE_NAMES
from .forecasts import reference_forecasts


HOURS=[0,12,24,48,96,144,168]


def run(args,store):
    from ..workflow import analysis,model_path,fixed_fit,load_records,member_dates,run_signature
    signature=run_signature(args)
    marker=args.output/'horizon_plan.json'
    plan=dict(signature=signature,targets=args.target,hours=HOURS)
    if marker.exists() and read_json(marker)!=plan:raise ValueError('Horizon source/configuration changed; use new output')
    save_json(marker,plan)
    cases=read_json(args.results/'bundle/results.json')['targets']
    records=load_records(args.prepared);scores=[]
    for case in cases:
        if not case['eligible_for_followup'] or args.target and case['target'] not in args.target:continue
        source=model_path(args.results/'models',case['target'],case['run'],case['method'])
        selected=read_json(source/'selected.json');saved=read_json(source/'assessment/fit.json')
        original=pd.read_csv(source/'assessment/predictions.csv',parse_dates=['timestamp'],float_precision='round_trip').set_index('timestamp').sort_index()
        lab=records.loc[records.target.eq(case['laboratory_target'])].set_index('TIMESTAMP').value.sort_index()
        for horizon in HOURS:
            folder=args.output/'horizons'/case['target']/f'h{horizon:03d}';folder.mkdir(parents=True,exist_ok=True)
            marker=folder/'result.json'
            if marker.exists():scores.append(read_json(marker));continue
            model=analysis(args,store,folder/'models',case['laboratory_target'],case['run'],case['method'])
            predictions,details=fixed_fit(model,selected,saved,horizon)
            table=predictions.set_index('timestamp').sort_index()
            if horizon==0:
                if not table.index.equals(original.index):raise AssertionError('Zero-hour membership differs')
                columns=['observed','prediction','prediction_unclipped','prior']
                if not np.allclose(table[columns],original[columns],rtol=1e-6,atol=1e-8,equal_nan=True):
                    raise AssertionError('Zero-hour refit does not reproduce supplied predictions/priors')
            training=pd.DatetimeIndex(read_json(source/'identity.json')['train'])
            refs=reference_forecasts(lab,training,table.index,model.rows,target=case['laboratory_target'],prior=table.prior,horizon=horizon)
            for name,values in refs.items():table['reference: '+REFERENCE_NAMES[name]]=values
            fixed=pd.DatetimeIndex(case['evaluation']['timestamps']);aligned=table.reindex(fixed)
            references=list(case['scores']['baselines'])
            valid=np.isfinite(aligned[['observed','prediction']+['reference: '+name for name in references]]).all(axis=1)
            original_primary=original.reindex(fixed[valid]).copy()
            result=dict(target=case['target'],horizon=horizon,n_expected=len(fixed),n=int(valid.sum()),
                fixed_set_complete=bool(valid.all()),scores=metrics(aligned.loc[valid],references),
                zero_hour_identity='passed' if horizon==0 else None)
            # A partial horizon retains a matched zero-hour score instead of
            # comparing different observed values without saying so.
            result['matched_zero_hour_r2']=(1-np.square(original_primary.observed-original_primary.prediction).mean()/original_primary.observed.var(ddof=0)) if len(original_primary)>1 and original_primary.observed.nunique()>1 else None
            table.to_csv(folder/'predictions_and_references.csv',index_label='timestamp')
            save_json(marker,result);scores.append(result)
    rows=[dict(target=r['target'],horizon=r['horizon'],n=r['n'],n_expected=r['n_expected'],fixed_set_complete=r['fixed_set_complete'],
        **{k:(r['scores'] or {}).get(k) for k in ['r2','rmse','mae','skill','best_baseline']}) for r in scores]
    pd.DataFrame(rows).to_csv(args.output/'horizon_scores.csv',index=False)
    save_json(args.output/'horizons.json',scores)

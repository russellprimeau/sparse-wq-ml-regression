"""Verify saved horizon sources before inheriting their numerical results."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from ..data.native import digest


def copy_horizon_scores(results, models, output, cases):
    results,models,output=map(Path,[results,models,output])
    target=output/'horizon_scores.csv'
    manifest=results/'horizons/source_models.json'
    source=results/'bundle/horizon_scores.csv'
    if not manifest.exists() or not source.exists():
        target.unlink(missing_ok=True)
        return False
    entries=json.loads(manifest.read_text(encoding='utf-8'))
    required=[case for case in cases if case['eligible_for_followup']]
    table=pd.read_csv(source)
    for case in required:
        entry=entries.get(case['target'])
        if entry is None or any(case[key]!=entry[key] for key in ['run','method','variant','features','seeds']):
            target.unlink(missing_ok=True);return False
        if case['evaluation']['timestamps']!=entry['dates']:
            target.unlink(missing_ok=True);return False
        directory=models/case['method']/'half'/case['run']/case['target']
        if any(not (directory/name).exists() or digest(directory/name)!=sha for name,sha in entry['files'].items()):
            target.unlink(missing_ok=True);return False
        for name,sha in entry.get('horizon_files',{}).items():
            saved=results/'horizons'/case['target']/name
            if not saved.exists() or digest(saved)!=sha:raise ValueError('Stored horizon product changed: '+name)
        zero=results/'horizons'/case['target']/'h000/predictions.csv'
        if not zero.exists():target.unlink(missing_ok=True);return False
        before=pd.read_csv(directory/'assessment/predictions.csv',index_col='timestamp',float_precision='round_trip')
        after=pd.read_csv(zero,index_col='timestamp',float_precision='round_trip')
        columns=['observed','prediction','prior']
        if not before.index.equals(after.index) or not np.allclose(before[columns],after[columns],rtol=1e-6,atol=1e-8,equal_nan=True):
            raise ValueError('Stored zero-hour horizon differs from selected fit: '+case['target'])
        frame=table.loc[table.target.eq(case['target'])]
        if set(frame.horizon)!={0,12,24,48,96,144,168}:
            raise ValueError('Incomplete stored horizon series: '+case['target'])
        np.testing.assert_allclose(frame.loc[frame.horizon.eq(0),'r2'].iloc[0],case['scores']['r2'],atol=1e-10)
    table.loc[table.target.isin([case['target'] for case in required])].to_csv(target,index=False)
    return True

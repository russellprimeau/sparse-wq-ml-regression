"""Prepare native profiler, qualified laboratory and historical weather inputs."""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from .native import digest,SHALLOW_COLUMNS,DEEP_COLUMNS
from .profiler_qc import configuration,screen_channels,shallow_selection,screening_summary
from .profiler_depth import select_cast_depth,configuration as depth_configuration
from .summaries import configuration as summary_configuration
from .qualified import qualified_records,historical_reconciliation,laboratory_summary,chronological_split,expanding_folds
from .weather_sources import run as weather_validation,source_records


def read_profiler(path):
    frame=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    frame['TIMESTAMP']=pd.to_datetime(frame.TIMESTAMP,errors='raise')
    frame['RECORD']=pd.to_numeric(frame.RECORD,errors='raise').astype('int64')
    frame=frame.set_index(['TIMESTAMP','RECORD'])
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Profiler row keys must be unique and chronologically sorted')
    return frame


def prepare(raw,config,out):
    raw,config,out=map(lambda p:Path(p).resolve(),[raw,config,out])
    if out.exists():
        marker=out/'preparation.json'
        if not marker.exists():raise FileExistsError('Incomplete preparation; use a fresh --prepared directory')
        manifest=json.loads(marker.read_text(encoding='utf-8'))
        for path,expected in manifest['source_files'].items():
            if digest(path)!=expected:raise ValueError('Preparation source changed: '+path)
        for name,expected in manifest['outputs_sha256'].items():
            if digest(out/name)!=expected:raise ValueError('Prepared input changed: '+name)
        return manifest
    sensors=raw/'sensors'
    source_manifest=json.loads((sensors/'profiler_source.json').read_text(encoding='utf-8'))
    for name,expected in source_manifest['outputs_sha256'].items():
        if digest(sensors/name)!=expected:raise ValueError('Corrected logger source changed: '+name)
    start,end=map(pd.Timestamp,['2021-12-13 00:00:00','2025-02-10 12:00:00'])
    hourly=read_profiler(sensors/'FullHourly_source_corrected.csv')
    profiles=read_profiler(sensors/'Profiles_source_corrected.csv')
    shallow,shallow_flags,_=screen_channels(hourly)
    eligible=shallow_selection(hourly).eligible
    shallow=shallow.loc[eligible];shallow_flags=shallow_flags.loc[eligible]
    screened,profile_flags,_=screen_channels(profiles)
    deep,casts,_=select_cast_depth(screened);deep_flags=profile_flags.loc[deep.index]
    out.mkdir(parents=True)
    summary=[]
    for series,frame,flags in [('shallow',shallow,shallow_flags),('near35',deep,deep_flags)]:
        frame.to_csv(out/('FullHourly_shallow_qc.csv' if series=='shallow' else 'Profiles_35m_qc.csv'),
            date_format='%Y-%m-%dT%H:%M:%S',lineterminator='\n')
        for scope in ['all_dates','study']:
            times=flags.index.get_level_values('TIMESTAMP')
            keep=times.to_series(index=flags.index).between(start,end) if scope=='study' else np.ones(len(flags),dtype=bool)
            summary.extend(dict(series=series,scope=scope,**row) for row in screening_summary(flags.loc[keep]))
    pd.DataFrame(summary).to_csv(out/'profiler_screening_summary.csv',index=False)
    casts.to_csv(out/'cast_depth_selection.csv',index=False)
    records,provenance=qualified_records(sensors/'Laboratory.csv')
    records=historical_reconciliation(records,sensors/'Laboratory_numeric.csv')
    records['numeric_value_source']='qualified_export'
    for index,row in records.loc[records.historical_match&records.qualifier.eq('=')].iterrows():
        text=row.original_result.replace(',','.');decimals=len(text.split('.')[-1]) if '.' in text else 0
        if not math.isclose(round(float(row.historical_value),decimals),float(row.reported_number),abs_tol=1e-12):
            raise ValueError('Conflicting laboratory precision sources')
        records.loc[index,'reported_number']=row.historical_value
        records.loc[index,'numeric_value_source']='finer_historical_numeric_agrees_at_source_precision'
    records.to_csv(out/'qualified_observations.csv',index=False)
    laboratory_summary(records,'all').to_csv(out/'laboratory_summary.csv',index=False)
    shutil.copyfile(sensors/'SCADA.csv',out/'SCADA.csv')
    study=yaml.safe_load((config/'study.yml').read_text(encoding='utf-8'))
    if study['predictors']['profiler']!=SHALLOW_COLUMNS+DEEP_COLUMNS:raise ValueError('Expected both profiler depths')
    for name in ['study.yml','search.yml']:shutil.copyfile(config/name,out/name)
    partitions=[]
    for target,group in records.groupby('target',sort=False):
        train,test=chronological_split(group.TIMESTAMP)
        for phase,dates in [('training',train),('evaluation',test)]:
            partitions.extend(dict(target=target,TIMESTAMP=t,split=phase) for t in dates)
        for fold,(training,validation) in enumerate(expanding_folds(train),1):
            for phase,dates in [('training',training),('validation',validation)]:
                partitions.extend(dict(target=target,TIMESTAMP=t,split=f'fold{fold}_{phase}') for t in dates)
    pd.DataFrame(partitions).to_csv(out/'partitions.csv',index=False)
    station=sensors/'Weather_raw.csv';archive=raw/'met_nordic'
    weather_validation(out/'weather_validation',station,archive)
    _,qc,weather_hashes=source_records(station,archive)
    qc.to_csv(out/'weather_source_qc.csv',index=False)
    sources=[sensors/name for name in ['profiler_source.json','FullHourly_source_corrected.csv','Profiles_source_corrected.csv','Laboratory.csv','Laboratory_numeric.csv','SCADA.csv']]
    sources.extend([config/'study.yml',config/'search.yml'])
    provenance.update(stage='native_rerun_inputs',study_start=str(start),study_end_inclusive=str(end),study=study,
        scope='all',interpolation='none',timestamp_policy='logger wall-clock; laboratory noon convention',
        sensor_configuration=configuration(),depth_configuration=depth_configuration(),summary_configuration=summary_configuration(),
        weather=dict(station=str(station),archive=str(archive),pilot=str(out/'weather_validation'),
            source_sha256=weather_hashes,validation_sha256=digest(out/'weather_validation/historical_validation.csv'),
            selection_end_exclusive='2021-12-13 00:00:00'),
        source_files={str(p):digest(p) for p in sources}|weather_hashes,
        outputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'preparation.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return provenance

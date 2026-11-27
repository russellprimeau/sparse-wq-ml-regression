"""Block mapping, paired scoring, atomic files and single-writer locking."""
from __future__ import annotations
from contextlib import contextmanager
import copy
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
ARMS = {'profiler_differential', 'profiler_absolute', 'profiler_absolute_noprior', 'noprofiler_differential', 'noprofiler_absolute', 'noprofiler_absolute_noprior'}
REFERENCE_NAMES = {'Training mean': 'Training mean', 'Calendar': 'Calendar', 'Calendar + prior': 'Calendar + prior', 'Naive': 'Persistence', 'Linear': 'Recent linear trend', 'Seasonal': 'Earlier-years seasonal mean'}
FIXED_COLUMNS = {'TIMESTAMP', 'target', 'parameter'}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def atomic_text(path, text):
    """Replace complete files, retrying transient Windows reader locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pending', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text.encode('utf-8'))
    try:
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(0.02 * 2 ** attempt, 0.5))
    finally:
        temporary.unlink(missing_ok=True)

def save_json(path, value):
    atomic_text(path, json.dumps(clean_json(value), indent=2, allow_nan=False, default=str) + '\n')

def block_order(length, block_length, seed, repetition, target):
    """Uniformly reorder consecutive observation blocks; retain within-block order.

    Include identity arrangements. Both arms of a target receive the same
    rearrangement, independently of case ordering and model-fitting seeds.
    The last block may be shorter. Blocks represent observations, not durations.
    """
    if min(length, block_length) < 1 or repetition < 1 or seed < 0:
        raise ValueError('Positive lengths/repetition and nonnegative seed required')
    blocks = [np.arange(start, min(start + block_length, length)) for start in range(0, length, block_length)]
    if len(blocks) < 2:
        raise ValueError('At least two laboratory blocks are required')
    target_seed = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition, target_seed]))
    order = rng.permutation(len(blocks))
    return (np.concatenate([blocks[number] for number in order]), order.tolist())

def rearrange_records(records, target, block_length, seed, repetition):
    """Move whole qualified assay payloads, leaving sampling dates/other assays fixed."""
    if not {'TIMESTAMP', 'target'} <= set(records):
        raise ValueError('Qualified dated laboratory records required')
    selected = records.loc[records.target.eq(target)].sort_values('TIMESTAMP', kind='stable')
    if selected.empty or selected.TIMESTAMP.duplicated().any():
        raise ValueError('Target must have unique laboratory timestamps')
    order, blocks = block_order(len(selected), block_length, seed, repetition, target)
    payload = [name for name in records if name not in FIXED_COLUMNS]
    result = records.copy(deep=True)
    for name in payload:
        result.loc[selected.index, name] = selected.iloc[order][name].to_numpy()
    mapping = pd.DataFrame({'destination_timestamp': selected.TIMESTAMP.to_numpy(), 'source_timestamp': selected.iloc[order].TIMESTAMP.to_numpy(), 'destination_position': np.arange(len(selected)), 'source_position': order})
    return (result, mapping, blocks)

def score_predictions(table, fit, lab, rows, *, target, nominal_training):
    from sparse_wq.evaluate.forecasts import reference_forecasts
    table = table.copy()
    table['timestamp'] = pd.to_datetime(table.timestamp)
    if table.timestamp.duplicated().any():
        raise ValueError('Duplicate prediction timestamps')
    table = table.set_index('timestamp').sort_index()
    if not np.array_equal(table.observed.to_numpy(), lab.loc[table.index].to_numpy()):
        raise ValueError('Predictions do not describe the rearranged laboratory observations')
    members = sorted({name for detail in fit['details'] for name in detail.get('training_observations', [])})
    if not members:
        raise ValueError('Actual fitted training memberships are required for references')
    training = pd.to_datetime([name.removeprefix('segment_').removesuffix('.csv') for name in members], format='%Y%m%d%H')
    references = reference_forecasts(lab, pd.DatetimeIndex(nominal_training), table.index, rows, target=target, prior=table.prior)
    for name, values in references.items():
        table['reference: ' + REFERENCE_NAMES[name]] = values
    return table

def metrics(table, references):
    from sparse_wq.data.qualified import evaluation_metrics
    if not len(table):
        return None
    model = evaluation_metrics(table.observed.to_numpy(), table.prediction.to_numpy())
    scores = {name: evaluation_metrics(table.observed.to_numpy(), table['reference: ' + name].to_numpy()) for name in references}
    best = min(scores, key=lambda name: (scores[name]['rmse'], name)) if scores else None
    scale = scores[best]['rmse'] if best else 0
    return clean_json(dict(**model, baselines=scores, best_baseline=best, skill=1 - model['rmse'] / scale if scale > 0 else None, baselines_beaten=sum((model['rmse'] < score['rmse'] for score in scores.values()))))



@contextmanager
def run_lock(out):
    import psutil
    lock = Path(out) / 'run_lock.json'
    if lock.exists():
        old = read_json(lock)
        try:
            alive = abs(psutil.Process(old['pid']).create_time() - old['created']) < 1
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError('This experiment already has a running controller')
        lock.unlink()
    process = psutil.Process()
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=process.pid, created=process.create_time()), stream)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)

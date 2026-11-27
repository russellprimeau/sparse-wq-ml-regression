"""Identify logger casts and select one native observation near intake depth."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sparse_wq.data.profiler_qc import numbers
REFERENCE_M, LOWER_M, UPPER_M = (35.0, 30.0, 40.0)

def configuration() -> dict:
    return dict(reference_m=REFERENCE_M, lower_m=LOWER_M, upper_m=UPPER_M, inclusivity='both', selection='nearest depth per cast; one complete source row', ties='earlier timestamp, then lower RECORD', interpolation='none', cast_boundary='PFL_Counter changes or RECORD resets in timestamp order', missing_channels='remain missing in the selected row; no substitution from another depth')

def select_cast_depth(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select nearest 35 m row per cast within the approved 30--40 m envelope.

    Logger counters can reset; a counter value is not a globally unique cast.
    Preserve original row keys and every selected source value. Selection uses
    measured depth rather than the rounded RS232Dpt metadata field. Missing
    sensor channels do not cause selection of a different depth.
    """
    if list(frame.index.names) != ['TIMESTAMP', 'RECORD'] or not frame.index.is_unique:
        raise ValueError('Cast selection requires unique TIMESTAMP/RECORD keys')
    if not frame.index.is_monotonic_increasing:
        raise ValueError('Cast selection requires timestamp/record order')
    if frame.empty:
        raise ValueError('No profile records supplied')
    counter = numbers(frame['PFL_Counter'])
    if counter.isna().any() or (~np.isfinite(counter)).any() or (counter <= 0).any() or (counter % 1 != 0).any():
        raise ValueError('Invalid PFL_Counter; cast membership is unresolved')
    record = pd.Series(frame.index.get_level_values('RECORD'), index=frame.index)
    boundary = counter.ne(counter.shift()) | record.le(record.shift())
    cast = boundary.cumsum().astype('int64')
    depth = numbers(frame['sensorParms(9)'])
    choices = pd.DataFrame(dict(cast_id=cast, depth_m=depth, distance_m=(depth - REFERENCE_M).abs(), eligible=depth.between(LOWER_M, UPPER_M, inclusive='both')))
    candidates = choices.loc[choices.eligible].reset_index()
    candidates = candidates.sort_values(['cast_id', 'distance_m', 'TIMESTAMP', 'RECORD'], kind='stable')
    winners = candidates.drop_duplicates('cast_id').set_index(['TIMESTAMP', 'RECORD'])
    choices['selected'] = choices.index.isin(winners.index)
    choices['reason'] = 'outside_depth_envelope'
    choices.loc[choices.depth_m.isna() | ~np.isfinite(choices.depth_m), 'reason'] = 'missing_or_invalid_depth'
    choices.loc[choices.eligible, 'reason'] = 'farther_from_reference_or_later_tie'
    choices.loc[choices.selected, 'reason'] = 'nearest_admissible_depth'
    selected = frame.loc[choices.selected].copy()
    records = []
    for cast_id, group in choices.groupby('cast_id', sort=False):
        positions = group.index
        chosen = group.loc[group.selected]
        info = dict(cast_id=int(cast_id), PFL_Counter=int(counter.loc[positions].iloc[0]), start=positions.get_level_values('TIMESTAMP').min(), end=positions.get_level_values('TIMESTAMP').max(), rows=len(group), candidates=int(group.eligible.sum()), min_depth_m=float(group.depth_m.min()) if group.depth_m.notna().any() else None, max_depth_m=float(group.depth_m.max()) if group.depth_m.notna().any() else None, selection_status='selected' if len(chosen) else 'no_admissible_depth', selected_timestamp=None, selected_record=None, selected_depth_m=None, distance_m=None)
        if len(chosen):
            key = chosen.index[0]
            info.update(selected_timestamp=key[0], selected_record=int(key[1]), selected_depth_m=float(chosen.depth_m.iloc[0]), distance_m=float(chosen.distance_m.iloc[0]))
        records.append(info)
    return (selected, pd.DataFrame(records), choices)

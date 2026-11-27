"""Native-time summaries and representation rules for the corrected rerun.

Blocks are explicit time intervals, not arrays of interpolated/repeated hourly
values. Callers supply their actual window boundaries; no calendar alignment
or laboratory timestamp convention is inferred here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
MIN_OBSERVATIONS = 2
DEEP_MIN_OBSERVATIONS = 1
BLOCK_HOURS = (12, 24)
STATISTICS = ('mean', 'min', 'max', 'std')

def configuration() -> dict:
    return dict(block_hours=list(BLOCK_HOURS), minimum_block_observations=dict(deeper=DEEP_MIN_OBSERVATIONS, other=MIN_OBSERVATIONS), deeper_standard_deviation=dict(scope='predictor_window', minimum_observations=MIN_OBSERVATIONS, model_features_per_series=1, ddof=0), other_standard_deviation=dict(scope='block', minimum_observations=MIN_OBSERVATIONS, ddof=0), empty_blocks='missing; no interpolation or repetition', deeper_representations=['mean', 'stats:12h', 'stats:24h'], other_representations=['none', 'mean', 'stats:12h', 'stats:24h'])

def representation_allowed(spec: str, *, includes_deeper: bool) -> bool:
    """Both depths can use 12h/24h/full-window summaries; deeper inputs stay native."""
    text = str(spec).strip().lower()
    allowed = {'mean', 'stats:12h', 'stats:24h'}
    if not includes_deeper:
        allowed.add('none')
    return text in allowed

def summarise_native(frame: pd.DataFrame, edges: pd.DatetimeIndex, *, deeper_columns=()) -> pd.DataFrame:
    """Mean/min/max/population SD per predictor in [left, right) intervals.

    Other series require two observations per block. Deeper mean/min/max allow
    one; their SD uses all valid observations inside this predictor window,
    requiring two in that larger population. Empty blocks remain missing.
    summary_features() emits the shared window SD once, not once per block.
    Each row must be one observation, identified by its timestamp (or by a
    TIMESTAMP/RECORD MultiIndex). Repeated values do not count as multiple
    observations unless they were separate source records.
    """
    if not frame.index.is_unique:
        raise ValueError('Summary inputs must have unique observation identities')
    if not frame.columns.is_unique:
        raise ValueError('Summary predictor names must be unique')
    deeper_columns = set(deeper_columns)
    if not deeper_columns <= set(frame.columns):
        raise ValueError('Deeper predictor names must be present in the input frame')
    times = frame.index.get_level_values('TIMESTAMP') if isinstance(frame.index, pd.MultiIndex) else frame.index
    if not isinstance(times, pd.DatetimeIndex):
        raise ValueError('Summary inputs require native datetime timestamps')
    edges = pd.DatetimeIndex(edges)
    if len(edges) < 2 or edges.hasnans or (not edges.is_monotonic_increasing) or (not edges.is_unique):
        raise ValueError('Supply at least two strictly increasing time boundaries')
    if times.tz != edges.tz:
        raise ValueError('Observation and block clocks must match')
    if times.hasnans:
        raise ValueError('Observation timestamps cannot be missing')
    values = frame.apply(pd.to_numeric, errors='raise')
    values = values.where(np.isfinite(values))
    ids = np.searchsorted(edges.as_unit('ns').asi8, times.as_unit('ns').asi8, side='right') - 1
    keep = (ids >= 0) & (ids < len(edges) - 1)
    selected = values.iloc[np.flatnonzero(keep)].copy()
    selected.index = ids[keep]
    groups = selected.groupby(level=0)
    window_counts = selected.count()
    window_std = selected.std(ddof=0).where(window_counts >= MIN_OBSERVATIONS)
    block_index = pd.RangeIndex(len(edges) - 1)
    counts = groups.count().reindex(block_index, fill_value=0)
    stats = {'mean': groups.mean(), 'min': groups.min(), 'max': groups.max(), 'std': groups.std(ddof=0)}
    rows = []
    for column in values:
        deep = column in deeper_columns
        minimum = DEEP_MIN_OBSERVATIONS if deep else MIN_OBSERVATIONS
        for block in block_index:
            n = int(counts.at[block, column])
            row = dict(block_start=edges[block], block_end=edges[block + 1], column=column, count=n, sufficient=n >= minimum, minimum_observations=minimum, window_count=int(window_counts[column]), window_std=float(window_std[column]), std_scope='window' if deep else 'block')
            for name, data in stats.items():
                if name == 'std' and deep:
                    row[name] = float(window_std[column]) if n >= minimum else np.nan
                else:
                    row[name] = float(data.at[block, column]) if n >= minimum else np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=['block_start', 'block_end', 'column', 'count', 'sufficient', 'minimum_observations', 'window_count', 'window_std', 'std_scope', *STATISTICS])

def summary_features(summary: pd.DataFrame, statistics=STATISTICS) -> pd.DataFrame:
    """Model feature values and their sensor/statistic metadata for one window.

    Deeper SD is a single window-level feature, avoiding repeated copies that
    could change its weight in a distance-based model. Missing block statistics
    stay missing; the separately computed window SD is not gap filling.
    """
    if not set(statistics) <= set(STATISTICS):
        raise ValueError('Unsupported summary statistic')
    rows = []
    for column, group in summary.groupby('column', sort=False):
        if group.std_scope.nunique() != 1:
            raise ValueError('Cannot combine different SD scopes for one predictor')
        deep = group.std_scope.iloc[0] == 'window'
        for block, row in enumerate(group.to_dict('records')):
            for statistic in statistics:
                if statistic == 'std' and deep:
                    continue
                rows.append(dict(column=column, scope='block', block_index=block, block_start=row['block_start'], block_end=row['block_end'], statistic=statistic, count=row['count'], value=row[statistic]))
        if deep and 'std' in statistics:
            first = group.iloc[0]
            rows.append(dict(column=column, scope='window', block_index=-1, block_start=first.block_start, block_end=group.iloc[-1].block_end, statistic='std', count=int(first.window_count), value=float(first.window_std)))
    return pd.DataFrame(rows, columns=['column', 'scope', 'block_index', 'block_start', 'block_end', 'statistic', 'count', 'value'])

"""Native measurements and historical cutoff-specific meteorological correction."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sparse_wq.data.summaries import summarise_native, summary_features, representation_allowed
from sparse_wq.data.profiler_qc import RULES
from sparse_wq.data.weather_correction import HistoricalWeatherCorrection, select_correction_forms, specifications_for_forms
DEEP_SUFFIX = ' [35 m]'
SHALLOW_COLUMNS = ['Pfl - ' + rule.label for rule in RULES.values()]
DEEP_COLUMNS = [name + DEEP_SUFFIX for name in SHALLOW_COLUMNS]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def window_bounds(stamp, rows: int, horizon: int=0):
    """Inclusive boundaries: the last observation is at least h+1 hours old.

    Preserve the existing 167/671-hourly-row convention. The laboratory prior
    cutoff moves with the environmental-window start at positive horizons.
    The nanosecond added by block_edges implements an inclusive cutoff without
    accepting observations later in that final hour.
    """
    stamp = pd.Timestamp(stamp)
    if stamp.tz is not None or stamp != stamp.floor('h'):
        raise ValueError('Target timestamps must be whole hours in the source clock')
    if rows < 1 or horizon < 0:
        raise ValueError('Positive row count and nonnegative horizon required')
    start = stamp - pd.Timedelta(hours=int(rows) + int(horizon))
    cutoff = stamp - pd.Timedelta(hours=int(horizon) + 1)
    return (start, cutoff, start)

def hourly_values(observations, features, start, cutoff):
    """Latest actual observation in each (hour-1h, hour] source interval.

    Empty intervals stay missing, including when the latest source row itself
    has a missing channel. No older valid value is substituted. Simultaneous
    source rows tie by the higher logger record number, then input order.
    """
    features = list(features)
    if any((name.endswith(DEEP_SUFFIX) for name in features)):
        raise ValueError('Deeper predictors cannot be converted to hourly rows')
    hours = pd.date_range(start, cutoff, freq='h', name='TIMESTAMP')
    result = pd.DataFrame(np.nan, index=hours, columns=features)
    grouped = observations.groupby(level='series', sort=False) if isinstance(observations.index, pd.MultiIndex) else [('source', observations)]
    for _, frame in grouped:
        times = frame.index.get_level_values('TIMESTAMP') if isinstance(frame.index, pd.MultiIndex) else frame.index
        times = pd.DatetimeIndex(times)
        records = frame.index.get_level_values('RECORD').to_numpy() if isinstance(frame.index, pd.MultiIndex) and 'RECORD' in frame.index.names else np.arange(len(frame))
        identity = pd.DataFrame(dict(time=times, record=records, position=np.arange(len(frame)), hour=times.ceil('h')))
        identity = identity.loc[identity.hour.isin(hours)].sort_values(['time', 'record', 'position'], kind='stable')
        chosen = identity.groupby('hour', sort=False).tail(1)
        selected = frame.iloc[chosen.position.to_numpy()][features].copy()
        selected.index = pd.DatetimeIndex(chosen.hour, name='TIMESTAMP')
        selected = selected.reindex(hours)
        if (result.notna() & selected.notna()).any().any():
            raise ValueError('Multiple source series supply the same hourly predictor')
        result = result.combine_first(selected).reindex(columns=features)
    return result

def hourly_features(observations, features, start, cutoff):
    values = hourly_values(observations, features, start, cutoff)
    rows = []
    for block, (stamp, row) in enumerate(values.iterrows()):
        for name in features:
            value = float(row[name])
            rows.append(dict(column=name, scope='hour', block_index=block, block_start=stamp - pd.Timedelta(hours=1), block_end=stamp, statistic='value', count=int(np.isfinite(value)), value=value))
    return pd.DataFrame(rows, columns=['column', 'scope', 'block_index', 'block_start', 'block_end', 'statistic', 'count', 'value'])

def block_edges(start, cutoff, representation):
    start, cutoff = (pd.Timestamp(start), pd.Timestamp(cutoff))
    end = cutoff + pd.Timedelta(nanoseconds=1)
    if representation == 'mean':
        return pd.DatetimeIndex([start, end])
    if representation not in ('stats:12h', 'stats:24h'):
        raise ValueError('This path requires a native summary representation')
    hours = int(representation.removeprefix('stats:').removesuffix('h'))
    return pd.date_range(start, end, freq=pd.Timedelta(hours=hours)).union(pd.DatetimeIndex([end]))

def encode_window(observations, features, start, cutoff, representation):
    """Physical-unit features, counts and their explicit source/statistic layout.

    Laboratory priors are supplied separately by the fitting adapter. This
    function accepts environmental channels only.
    """
    features = list(features)
    deep = [name for name in features if name.endswith(DEEP_SUFFIX)]
    if not representation_allowed(representation, includes_deeper=bool(deep)):
        raise ValueError('Representation incompatible with the selected predictors')
    if representation == 'none':
        return hourly_features(observations, features, start, cutoff)
    summary = summarise_native(observations[features], block_edges(start, cutoff, representation), deeper_columns=deep)
    statistics = ('mean',) if representation == 'mean' else ('mean', 'min', 'max', 'std')
    return summary_features(summary, statistics=statistics)

def predictor_weather(correction, through, start, end):
    """Same physical weather inputs as prepare_weather_predictors.py, in memory."""
    from sparse_wq.data.weather_sources import PREDICTOR_NAMES
    first = pd.Timestamp(start) - pd.Timedelta(hours=23)
    values, flags = correction.fill_station_gaps(first, end, through)
    index = pd.date_range(first, pd.Timestamp(end) - pd.Timedelta(hours=1), freq='h', name='TIMESTAMP')
    values = values.reindex(index)
    flags = flags.reindex(index, fill_value='missing')
    predictors = values[list(PREDICTOR_NAMES)].rename(columns=PREDICTOR_NAMES)
    predictors['24hr precipitation total (mm)'] = values.precipitation_amount.rolling(24, min_periods=24).sum()
    return (predictors.loc[start:], flags.loc[start:])

class NativeInputs:
    """Verified native profiler + hourly intake + weather fitted at each cutoff.

    Weather parameters and resulting series are shared across all windows with
    the same training cutoff. Source identities prevent simultaneous readings
    in different tables from being counted as repeated measurements.
    """

    def __init__(self, prepared, *, weather_cache=None):
        self.prepared = Path(prepared)
        self.manifest = json.loads((self.prepared / 'preparation.json').read_text(encoding='utf-8'))
        if self.manifest.get('stage') != 'native_rerun_inputs':
            raise ValueError('Expected prepared native rerun inputs')
        for name, expected in self.manifest['outputs_sha256'].items():
            if digest(self.prepared / name) != expected:
                raise ValueError(f'Prepared input changed: {name}')
        self.start = pd.Timestamp(self.manifest['study_start'])
        self.end = pd.Timestamp(self.manifest['study_end_inclusive'])
        sources = {}
        for series, filename in [('shallow', 'FullHourly_shallow_qc.csv'), ('near35', 'Profiles_35m_qc.csv')]:
            frame = pd.read_csv(self.prepared / filename, parse_dates=['TIMESTAMP'], na_values=['NAN'], float_precision='round_trip')
            mapping = {raw: 'Pfl - ' + rule.label + (DEEP_SUFFIX if series == 'near35' else '') for raw, rule in RULES.items()}
            sources[series] = frame.set_index(['TIMESTAMP', 'RECORD'])[list(mapping)].rename(columns=mapping)
        scada = pd.read_csv(self.prepared / 'SCADA.csv', sep=';', parse_dates=['Time'], float_precision='round_trip')
        if scada.Time.duplicated().any() or (scada.Time != scada.Time.dt.floor('h')).any():
            raise ValueError('Intake records must have unique, unchanged hourly timestamps')
        scada = scada.rename(columns={'Time': 'TIMESTAMP'})
        scada['RECORD'] = np.arange(len(scada))
        sources['intake'] = scada.set_index(['TIMESTAMP', 'RECORD'])
        self.fixed = pd.concat(sources, names=['series'])
        self.fixed = self.fixed.iloc[np.argsort(self.fixed.index.get_level_values('TIMESTAMP').asi8, kind='stable')]
        if not self.fixed.index.is_unique:
            raise ValueError('Native observation identities are not unique')
        self.fixed = self.fixed.where(np.isfinite(self.fixed))
        self._times = pd.DatetimeIndex(self.fixed.index.get_level_values('TIMESTAMP'))
        self._weather = {}
        self._weather_flags = {}
        weather = self.manifest['weather']
        from sparse_wq.data.weather_sources import source_records
        clean, self.weather_qc, sources = source_records(Path(weather['station']), Path(weather['archive']))
        if sources != weather['source_sha256']:
            raise ValueError('Weather sources differ from the prepared snapshot')
        pilot = Path(weather['pilot'])
        score_path = pilot / 'historical_validation.csv'
        if digest(score_path) != weather['validation_sha256']:
            raise ValueError('Weather form-selection scores changed')
        self.selected_forms = select_correction_forms(pd.read_csv(score_path))
        self.correction = HistoricalWeatherCorrection(clean['station'], clean['grid'], specifications_for_forms(self.selected_forms), cache_dir=weather_cache)

    def weather_at(self, through):
        through = pd.Timestamp(through)
        if pd.Timestamp(self.manifest['weather']['selection_end_exclusive']) > through:
            raise ValueError('Weather form selection extends beyond the training cutoff')
        if through not in self._weather:
            first = min(self.start, self.correction.station.index.min(), self.correction.grid.index.min())
            values, flags = predictor_weather(self.correction, through, first, self.end + pd.Timedelta(hours=1))
            self._weather[through], self._weather_flags[through] = (values, flags)
        return self._weather[through]

    def observations(self, start, cutoff, through, features):
        start, cutoff = (pd.Timestamp(start), pd.Timestamp(cutoff))
        if start > cutoff:
            raise ValueError('Predictor start exceeds its cutoff')
        times = self._times.as_unit('ns')
        left = times.searchsorted(start.as_unit('ns'), side='left')
        right = times.searchsorted(cutoff.as_unit('ns'), side='right')
        fixed_columns = [col for col in features if col in self.fixed]
        pieces = [self.fixed.iloc[left:right][fixed_columns]] if fixed_columns else []
        weather_columns = [col for col in features if col not in fixed_columns]
        if weather_columns:
            weather = self.weather_at(through).loc[start:cutoff, weather_columns].copy()
            weather.index.name = 'TIMESTAMP'
            weather['RECORD'] = np.arange(len(weather))
            weather = weather.reset_index().set_index(['TIMESTAMP', 'RECORD'])
            pieces.append(pd.concat({'weather': weather}, names=['series']))
        if not pieces:
            raise ValueError('At least one environmental predictor required')
        result = pd.concat(pieces).reindex(columns=features)
        if not result.index.is_unique:
            raise ValueError('Window observation identities are not unique')
        return result

    def window(self, stamp, rows, features, through, representation, horizon=0):
        start, cutoff, reference = window_bounds(stamp, rows, horizon)
        observation_start = start - pd.Timedelta(hours=1) + pd.Timedelta(nanoseconds=1) if representation == 'none' else start
        observations = self.observations(observation_start, cutoff, through, features)
        encoded = encode_window(observations, features, start, cutoff, representation)
        metadata = dict(timestamp=str(pd.Timestamp(stamp)), reference_timestamp=str(reference), predictor_start=str(start), predictor_cutoff=str(cutoff), training_through=str(pd.Timestamp(through)))
        return (encoded, metadata)

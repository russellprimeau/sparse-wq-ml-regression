"""Simple frozen station-reference corrections in physical units.

Fit on historical paired observations only. Applying a saved correction never
updates it or estimates parameters from the incoming window.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

def select_correction_forms(validation: pd.DataFrame) -> pd.DataFrame:
    """Lowest historical-validation RMSE on matched pairs; ties retain raw.

    Mean-bias reduction does not override an RMSE increase. Bias and MAE stay
    in the report so that the effect of the selected form remains visible.
    """
    required = {'variable', 'method', 'n_pairs', 'bias', 'rmse', 'mae'}
    if not required <= set(validation):
        raise ValueError('Incomplete weather validation metrics')
    rows = []
    for variable, group in validation.groupby('variable', sort=False):
        if group.method.duplicated().any():
            raise ValueError('Duplicate correction candidates')
        raw = group.loc[group.method.eq('raw')]
        if len(raw) != 1:
            raise ValueError('Each variable requires one uncorrected baseline')
        raw = raw.iloc[0]
        if not np.isfinite(group[['bias', 'rmse', 'mae', 'n_pairs']].to_numpy(dtype=float)).all():
            raise ValueError('Weather validation metrics must be finite')
        if (group[['rmse', 'mae']] < 0).any().any() or raw.n_pairs < 2:
            raise ValueError('Invalid weather validation metrics')
        if not group.n_pairs.eq(raw.n_pairs).all():
            raise ValueError('Compare correction forms on the same paired observations')
        candidates = group.loc[group.rmse < raw.rmse].sort_values(['rmse', 'method'], kind='stable')
        winner = candidates.iloc[0] if len(candidates) else raw
        rows.append(dict(variable=variable, selected_method=winner.method, n_pairs=int(raw.n_pairs), raw_bias=float(raw.bias), selected_bias=float(winner.bias), raw_rmse=float(raw.rmse), selected_rmse=float(winner.rmse), raw_mae=float(raw.mae), selected_mae=float(winner.mae), decision='lower historical-validation RMSE' if winner.method != 'raw' else 'no correction lowers historical-validation RMSE'))
    return pd.DataFrame(rows)

def specifications_for_forms(selected: pd.DataFrame) -> dict:
    specifications = {}
    for row in selected.to_dict('records'):
        name = row['variable']
        rain = name == 'precipitation_amount'
        solar = name.startswith('integral_of_surface_downwelling_shortwave')
        radiation = name.startswith('integral_')
        humidity = name == 'relative_humidity_2m'
        specifications[name] = dict(method=row['selected_method'], lower=0 if rain or radiation or humidity else None, upper=100 if humidity else None, preserve_zero=rain or solar)
    return specifications

def fit_correction(station: pd.Series, grid: pd.Series, method: str, *, before, lower=None, upper=None, preserve_zero=False) -> dict:
    pairs = pd.concat([station.rename('station'), grid.rename('grid')], axis=1, sort=True)
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    cutoff = pd.Timestamp(before)
    pairs = pairs.loc[pairs.index < cutoff]
    if len(pairs) < 2:
        raise ValueError('Correction requires two historical paired observations')
    x, y = (pairs.grid.to_numpy(), pairs.station.to_numpy())
    if method == 'raw':
        slope, intercept = (1.0, 0.0)
    elif method == 'additive':
        slope, intercept = (1.0, float(np.mean(y - x)))
    elif method == 'mean_ratio':
        if np.mean(x) <= 0:
            raise ValueError('A positive grid mean is required for ratio correction')
        slope, intercept = (float(np.mean(y) / np.mean(x)), 0.0)
    elif method == 'affine':
        if np.std(x) == 0:
            raise ValueError('Affine calibration requires variable grid observations')
        slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
        slope, intercept = (float(slope), float(intercept))
    else:
        raise ValueError(f'Unsupported correction method: {method}')
    return dict(method=method, slope=slope, intercept=intercept, lower=lower, upper=upper, preserve_zero=preserve_zero, n_pairs=len(pairs), first=str(pairs.index.min()), last=str(pairs.index.max()), before_exclusive=str(cutoff))

def apply_correction(grid: pd.Series, correction: dict) -> pd.Series:
    value = pd.to_numeric(grid, errors='raise').replace([np.inf, -np.inf], np.nan)
    result = correction['slope'] * value + correction['intercept']
    if correction['preserve_zero']:
        result = result.mask(value.eq(0), 0.0)
    return result.clip(lower=correction['lower'], upper=correction['upper'])

class HistoricalWeatherCorrection:
    """Fit once per training cutoff using all earlier available paired weather.

    The reference is deliberately not restricted to dates inside the water-
    quality training fold. Validation windows reuse its saved parameters.
    Callers explicitly choose correction forms and any fixed physical bounds.
    """

    def __init__(self, station: pd.DataFrame, grid: pd.DataFrame, specifications: dict, cache_dir: Path | None=None):
        if not specifications or not set(specifications) <= set(station) or (not set(specifications) <= set(grid)):
            raise ValueError('Explicit correction specifications must match source variables')
        for frame in [station, grid]:
            if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique or frame.index.hasnans:
                raise ValueError('Weather sources require unique, nonmissing datetime indices')
        if station.index.tz != grid.index.tz:
            raise ValueError('Weather source clocks must match')
        self.specifications = copy.deepcopy(specifications)
        self.station = station[list(specifications)].copy(deep=True).sort_index()
        self.grid = grid[list(specifications)].copy(deep=True).sort_index()
        identity = hashlib.sha256(json.dumps(self.specifications, sort_keys=True).encode())
        for frame in [self.station, self.grid]:
            identity.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        identity.update(Path(__file__).read_bytes())
        self.source_identity = identity.hexdigest()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._references = {}

    def fit_at(self, through) -> dict:
        """Include paired weather at or before the fold's training cutoff."""
        through = pd.Timestamp(through)
        if pd.isna(through) or through.tz != self.station.index.tz:
            raise ValueError('Training cutoff must use the weather source clock')
        token = hashlib.sha256((self.source_identity + through.isoformat()).encode()).hexdigest()
        if token in self._references:
            return copy.deepcopy(self._references[token])
        path = self.cache_dir / (token + '.json') if self.cache_dir is not None else None
        if path is not None and path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('identity') != token:
                raise ValueError('Weather cache identity mismatch')
        else:
            models = {name: fit_correction(self.station[name], self.grid[name], before=through + pd.Timedelta(nanoseconds=1), **spec) for name, spec in self.specifications.items()}
            result = dict(identity=token, through_inclusive=str(through), models=models, reference='all available paired history through the training cutoff; includes observations before fold start')
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='weather_', suffix='.tmp', delete=False) as handle:
                        temporary = Path(handle.name)
                        json.dump(result, handle, indent=2, allow_nan=False)
                        handle.write('\n')
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        self._references[token] = result
        return copy.deepcopy(result)

    def apply(self, grid_window: pd.DataFrame, through) -> pd.DataFrame:
        """Use a cached cutoff reference, without learning from this window."""
        reference = self.fit_at(through)
        return pd.DataFrame({name: apply_correction(grid_window[name], model) for name, model in reference['models'].items()}, index=grid_window.index)

    def fill_station_gaps(self, start, end, through) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return [start, end) values and provenance, preferring station data."""
        start, end = (pd.Timestamp(start), pd.Timestamp(end))
        if start.tz != self.station.index.tz or end.tz != self.station.index.tz or start >= end:
            raise ValueError('Weather window requires increasing boundaries in the source clock')
        index = self.station.index.union(self.grid.index).sort_values()
        index = index[(index >= start) & (index < end)]
        observed = self.station.reindex(index)
        corrected = self.apply(self.grid.reindex(index), through)
        filled = observed.combine_first(corrected).reindex(columns=self.station.columns)
        origin = pd.DataFrame('missing', index=index, columns=filled.columns)
        origin = origin.mask(corrected.notna(), 'met_nordic').mask(observed.notna(), 'station')
        return (filled, origin)

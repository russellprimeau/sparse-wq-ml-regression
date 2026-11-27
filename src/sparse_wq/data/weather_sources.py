"""Screened station/grid source records and pre-study weather-form validation."""
from __future__ import annotations
from sparse_wq.data.weather_correction import fit_correction, apply_correction
VARIABLES = {'air_pressure_at_sea_level': ('PR Trykk redusert', 'Pressure', 'mbar', 0.01, 0.0), 'air_temperature_2m': ('TA Middel', 'Air temperature', 'degC', 1.0, -273.15), 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time': ('QSI Kort', 'Shortwave radiation', 'W/m2', 1 / 3600, 0.0), 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time': ('QLI Lang', 'Longwave radiation', 'W/m2', 1 / 3600, 0.0), 'precipitation_amount': ('RR_1', 'Precipitation', 'mm/hour', 1.0, 0.0), 'relative_humidity_2m': ('UU Luftfuktighet', 'Relative humidity', 'percentage points', 100.0, 0.0), 'wind_direction_10m': ('DX_l', 'Wind direction', 'degrees', 1.0, 0.0), 'wind_speed_10m': ('FF Hastighet', 'Wind speed', 'm/s', 1.0, 0.0)}
WEATHER_BOUNDS: dict[str, tuple[float | None, float | None]] = {'Hourly average wind direction (°)': (0, 360), 'Average wind speed (m/s)': (0, 100), 'Maximum sustained wind speed, 3-second span (m/s)': (0, 100), 'Maximum sustained wind speed, 10-minute span (m/s)': (0, 100), 'Atmospheric pressure (mBar)': (860, 1080), 'Maximum pressure differential, 3-hour span (mBar)': (0, 50), 'Longwave (IR) radiation (W/m2)': (0, 750), 'Shortwave (solar) radiation (W/m2)': (0, 900), 'Precipitation (mm/hr)': (0, 50), '24hr precipitation total (mm)': (0, 24 * 50), 'Maximum temperature (°C)': (-40, 40), 'Minimum temperature (°C)': (-40, 40), 'Air temperature (°C)': (-40, 40), 'Humidity (%)': (0, 100)}
PREDICTOR_NAMES = {'air_temperature_2m': 'Air temperature (°C)', 'air_pressure_at_sea_level': 'Atmospheric pressure (mBar)', 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time': 'Shortwave (solar) radiation (W/m2)', 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time': 'Longwave (IR) radiation (W/m2)', 'relative_humidity_2m': 'Humidity (%)', 'wind_x': 'Wind speed x (m/s)', 'wind_y': 'Wind speed y (m/s)'}
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
LABELS = {'air_pressure_at_sea_level': 'Atmospheric pressure (mBar)', 'air_temperature_2m': 'Air temperature (°C)', 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time': 'Shortwave (solar) radiation (W/m2)', 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time': 'Longwave (IR) radiation (W/m2)', 'precipitation_amount': 'Precipitation (mm/hr)', 'relative_humidity_2m': 'Humidity (%)', 'wind_direction_10m': 'Hourly average wind direction (°)', 'wind_speed_10m': 'Average wind speed (m/s)'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_records(station_path: Path, archive: Path):
    manifest_path = archive / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    point = archive / 'met_nordic_ltc_hourly.csv'
    if digest(station_path) != manifest['station_sha256'] or digest(point) != manifest['point_data_sha256']:
        raise ValueError('Weather inputs changed since their full-record verification')
    station = pd.read_csv(station_path, sep=';', decimal=',', low_memory=False, parse_dates=['Time']).set_index('Time')
    grid = pd.read_csv(point, parse_dates=['Time'], float_precision='round_trip').set_index('Time')
    if not station.index.is_unique or not grid.index.is_unique:
        raise ValueError('Duplicate weather timestamps')
    columns = {}
    for variable, (prefix, _, _, _, _) in VARIABLES.items():
        match = [c for c in station if prefix.lower() in c.lower()]
        if len(match) != 1:
            raise ValueError(f'Ambiguous station variable: {variable}')
        columns[match[0]] = variable
    station = station[list(columns)].rename(columns=columns)
    records = []
    clean = {}
    for source, frame in [('station', station), ('grid', grid)]:
        frame = frame.sort_index().apply(pd.to_numeric, errors='raise').replace([-9999, -999, -99.9, -99, np.inf, -np.inf], np.nan)
        for variable, label in LABELS.items():
            value = frame[variable]
            negative_solar = pd.Series(False, index=value.index)
            if variable.startswith('integral_of_surface_downwelling_shortwave'):
                negative_solar = value.lt(0)
                value = value.clip(lower=0)
            lo, hi = WEATHER_BOUNDS[label]
            failed = value.lt(lo) | value.gt(hi)
            records.append(dict(source=source, variable=variable, missing=int(value.isna().sum()), excluded_by_existing_weather_bounds=int(failed.sum()), negative_solar_reset_to_zero=int(negative_solar.sum()), lower=lo, upper=hi))
            frame[variable] = value.mask(failed)
        direction = frame.wind_direction_10m
        speed = frame.wind_speed_10m
        frame['wind_x'] = np.cos(np.deg2rad(direction)) * speed
        frame['wind_y'] = np.sin(np.deg2rad(direction)) * speed
        clean[source] = frame.drop(columns=['wind_speed_10m', 'wind_direction_10m'])
    return (clean, pd.DataFrame(records), {str(station_path): digest(station_path), str(point): digest(point), str(manifest_path): digest(manifest_path)})

def score(y, pred):
    pairs = pd.concat([y.rename('station'), pred.rename('prediction')], axis=1, sort=True).dropna()
    if pairs.empty:
        raise ValueError('No paired weather observations to score')
    err = pairs.prediction - pairs.station
    return dict(n_pairs=len(pairs), bias=float(err.mean()), rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(abs(err))))

def run(output: Path, station_path: Path, archive: Path):
    if output.exists():
        raise FileExistsError(f'Use a fresh pilot output directory: {output}')
    clean, qc, sources = source_records(station_path, archive)
    fit_end = pd.Timestamp('2021-01-01')
    reference_end = pd.Timestamp('2021-12-13')
    rows = []
    later = []
    parameters = {}
    recommendations = []
    for variable in clean['station']:
        y, x = (clean['station'][variable], clean['grid'][variable])
        rain = variable == 'precipitation_amount'
        solar = variable.startswith('integral_of_surface_downwelling_shortwave')
        positive = rain or variable.startswith('integral_')
        humidity = variable == 'relative_humidity_2m'
        methods = ['raw', 'mean_ratio'] if rain else ['raw', 'additive', 'affine']
        if solar:
            methods.insert(2, 'mean_ratio')
        parameters[variable] = {}
        candidates = []
        for method in methods:
            model = fit_correction(y, x, method, before=fit_end, lower=0 if positive or humidity else None, upper=100 if humidity else None, preserve_zero=rain or solar)
            valid = (x.index >= fit_end) & (x.index < reference_end)
            metrics = score(y.loc[(y.index >= fit_end) & (y.index < reference_end)], apply_correction(x.loc[valid], model))
            row = dict(variable=variable, method=method, **metrics)
            rows.append(row)
            candidates.append(row)
            full = fit_correction(y, x, method, before=reference_end, lower=model['lower'], upper=model['upper'], preserve_zero=model['preserve_zero'])
            parameters[variable][method] = full
            pred = apply_correction(x.loc[x.index >= reference_end], full)
            observed = y.loc[y.index >= reference_end]
            paired = pd.concat([observed.rename('station'), pred.rename('prediction')], axis=1, sort=True).dropna()
            for year, group in paired.groupby(paired.index.year):
                later.append(dict(variable=variable, method=method, year=int(year), **score(group.station, group.prediction)))
        winner = min(candidates, key=lambda row: row['rmse'])
        recommendations.append(dict(variable=variable, lowest_historical_validation_rmse=winner['method'], validation_rmse=winner['rmse'], validation_bias=winner['bias']))
    output.mkdir(parents=True)
    tables = {'historical_validation.csv': pd.DataFrame(rows), 'later_weather_stability.csv': pd.DataFrame(later), 'source_qc_counts.csv': qc, 'historical_rmse_candidates.csv': pd.DataFrame(recommendations)}
    hashes = {}
    for name, table in tables.items():
        path = output / name
        table.to_csv(path, index=False)
        hashes[name] = digest(path)
    parameter_path = output / 'candidate_parameters.json'
    parameter_path.write_text(json.dumps(parameters, indent=2) + '\n', encoding='utf-8')
    hashes[parameter_path.name] = digest(parameter_path)
    manifest = dict(stage='historical_weather_form_validation', operational_correction_selected=True, form_comparison_reference_end_exclusive=str(reference_end), fit_end_exclusive=str(fit_end), validation_end_exclusive=str(reference_end), recommendation_basis='historical validation RMSE only; later stability checks are descriptive', preprocessing='existing weather bounds on both sources; negative solar set to zero as in the pipeline; no gap filling', fixed_output_bounds='nonnegative precipitation/radiation, humidity 0-100%; rain and solar zeros retained', raw_fig5c='unchanged', station_location='unchanged (author confirmed)', source_sha256=sources, output_sha256=hashes, code_sha256={str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('weather_correction.py')]}, operational_reference='all paired history from 16 January 2020 through the applicable training cutoff, including pre-fold history; cached and reused within that fold', note='Candidate parameters at the example December 2021 cutoff are not installed in model inputs. No water-quality targets or scores are read.')
    (output / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return pd.DataFrame(recommendations)

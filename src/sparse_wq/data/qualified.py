"""Qualified laboratory ingestion and audited preparation for the review revision.

Historical Eurofins.csv and manuscript-source outputs are never modified here.
The export's midnight timestamps are preserved as source metadata. Modelling uses
the existing noon convention, explicitly recorded rather than called a known
collection time. Units remain the laboratory's assay-specific units.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd
TARGETS = {'01-Farge': 'Color', '04-Turbiditet': 'Turbidity (FNU)', '44-pH, surhetsgrad': 'pH', '06-E.coli': 'E.coli (CFU/100mL)', '07-Intestinale enterokokker': 'Intestinal enterococci (CFU/100mL)', '08-Kimtall 22°C': 'Colony Count 22°C (CFU/mL)', '09-Koliforme bakterier 37°C': 'Total coliforms 37°C (CFU/100mL)', '21-Arsen': 'Arsenic (µg/L)', '24-Bly': 'Lead (µg/L)', '32-Kadmium': 'Cadmium (µg/L)', '36-Kopper filtrert': 'Copper filtered (mg/L)', '37-Krom': 'Chromium (µg/L)', '41-Nikkel': 'Nickel (µg/L)', 'Sink (Zn)': 'Zinc (µg/L)'}
EXPECTED_UNITS = {'01-Farge': 'mg/l Pt', '04-Turbiditet': 'FNU', '44-pH, surhetsgrad': '', '06-E.coli': '/100 ml', '07-Intestinale enterokokker': '/100 ml', '08-Kimtall 22°C': '/ml', '09-Koliforme bakterier 37°C': '/100 ml', '21-Arsen': 'µg/l As', '24-Bly': 'µg/l Pb', '32-Kadmium': 'µg/l Cd', '36-Kopper filtrert': 'mg/l', '37-Krom': 'µg/l Cr', '41-Nikkel': 'µg/l Ni', 'Sink (Zn)': 'µg/l'}
RESULT = re.compile('^\\s*(?P<qualifier><|=)?\\s*(?P<value>[+-]?(?:\\d+(?:[.,]\\d*)?|[.,]\\d+)(?:[eE][+-]?\\d+)?)\\s*$')

def parse_result(text: str) -> tuple[str, float]:
    """Return explicit '=' or '<' and the value/limit. Never guess a qualifier."""
    match = RESULT.fullmatch(str(text))
    if match is None:
        raise ValueError(f'Unsupported laboratory result: {text!r}')
    number = float(match['value'].replace(',', '.'))
    qualifier = match['qualifier'] or '='
    if not math.isfinite(number) or number < 0 or (qualifier == '<' and number <= 0):
        raise ValueError(f'Invalid laboratory value/limit: {text!r}')
    return (qualifier, number)

def qualified_records(path: Path) -> tuple[pd.DataFrame, dict]:
    """Selected assays, deduplicated only where the same sample/result/unit repeats.

    A '<' value is a reported upper bound. The export does not identify whether
    this is an analytical detection, reporting, or quantitation limit; retain
    limit_type='unspecified_reported_limit' until source documentation establishes it.
    """
    frame = pd.read_csv(path, sep=';', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    required = {'Dato', 'Resultat', 'Parameter', 'Enhet', 'Prøvenummer'}
    if not required <= set(frame):
        raise ValueError(f'{path}: missing columns {required - set(frame)}')
    frame['source_row'] = np.arange(2, len(frame) + 2)
    frame = frame.loc[frame.Parameter.isin(TARGETS)].copy()
    if set(frame.Parameter) != set(TARGETS):
        raise ValueError(f'Original export missing study assays: {set(TARGETS) - set(frame.Parameter)}')
    records = []
    source_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for row in frame.to_dict('records'):
        parameter = row['Parameter']
        if row['Enhet'] != EXPECTED_UNITS[parameter]:
            raise ValueError(f"Unexpected units for {parameter}: {row['Enhet']!r}")
        qualifier, number = parse_result(row['Resultat'])
        source_time = pd.to_datetime(row['Dato'], format='%d.%m.%Y %H:%M', errors='raise')
        records.append(dict(source_row=int(row['source_row']), sample_id=row['Prøvenummer'], parameter=parameter, target=TARGETS[parameter], source_timestamp=source_time, TIMESTAMP=source_time.normalize() + pd.Timedelta(hours=12), time_basis='historical_noon_convention', original_result=row['Resultat'], unit=row['Enhet'], qualifier=qualifier, reported_number=number, is_left_censored=qualifier == '<', limit=number if qualifier == '<' else np.nan, limit_type='unspecified_reported_limit' if qualifier == '<' else '', source_sha256=source_hash))
    out = pd.DataFrame(records)
    identity = ['TIMESTAMP', 'parameter']
    duplicates = out[out.duplicated(identity, keep=False)]
    for _, block in duplicates.groupby(identity):
        if len(block[['sample_id', 'original_result', 'unit']].drop_duplicates()) != 1:
            raise ValueError(f"Conflicting observations at one target/date: {block.to_dict('records')}")
    kept = out.drop_duplicates(identity).sort_values(['TIMESTAMP', 'parameter']).reset_index(drop=True)
    return (kept, {'source': str(Path(path).resolve()), 'sha256': records[0]['source_sha256'], 'study_assay_rows': len(out), 'duplicate_rows_removed': len(out) - len(kept), 'unique_study_observations': len(kept), 'time_basis': 'existing noon convention; source timestamps retained separately'})

def substitute(records: pd.DataFrame, treatment: str) -> pd.DataFrame:
    if treatment not in ('half', 'sqrt2'):
        raise ValueError('Treatment must be half or sqrt2')
    out = records.copy()
    out['value'] = out.reported_number.astype(float)
    mask = out.is_left_censored
    out.loc[mask, 'value'] = out.loc[mask, 'limit'] / (2.0 if treatment == 'half' else math.sqrt(2))
    out['treatment'] = treatment
    return out

def historical_reconciliation(records: pd.DataFrame, exported_csv: Path) -> pd.DataFrame:
    """Match by assay and calendar day; expose added dates and value discrepancies."""
    legacy = pd.read_csv(exported_csv, sep=';', na_values=['#N/A'], dtype=str)
    legacy['TIMESTAMP'] = pd.to_datetime(legacy.Time).dt.normalize() + pd.Timedelta(hours=12)
    old = legacy.melt(id_vars=['TIMESTAMP'], value_vars=list(TARGETS), var_name='parameter', value_name='historical_value')
    old['historical_value'] = pd.to_numeric(old.historical_value.str.replace(',', '.', regex=False), errors='raise')
    old = old.loc[old.historical_value.notna()]
    out = records.merge(old, on=['TIMESTAMP', 'parameter'], how='outer', indicator=True, validate='one_to_one')
    if (out['_merge'] == 'right_only').any():
        raise ValueError('Historical observations missing from qualified source')

    def old_substitution(row):
        if row.qualifier != '<':
            return row.reported_number
        limit = float(row.limit)
        power = 10.0 ** math.floor(math.log10(limit))
        return limit / 2.0 if math.isclose(limit, power, rel_tol=1e-12) else power
    out['legacy_rule_value'] = out.apply(old_substitution, axis=1)
    out['historical_match'] = out.historical_value.notna()
    out['legacy_rule_matches_export'] = np.isclose(out.legacy_rule_value, out.historical_value, rtol=1e-10, atol=1e-12)
    return out.drop(columns=['_merge'])

def laboratory_summary(records: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for parameter, g in records.groupby('parameter', sort=False):
        limits = sorted(g.loc[g.is_left_censored, 'limit'].dropna().unique().tolist())
        rows.append(dict(parameter=parameter, target=TARGETS[parameter], scope=scope, n=len(g), n_numeric=int((g.qualifier == '=').sum()), n_below_limit=int(g.is_left_censored.sum()), below_limit_percent=float(100 * g.is_left_censored.mean()), n_observed_zero=int(((g.qualifier == '=') & (g.reported_number == 0)).sum()), limits=';'.join((f'{v:g}' for v in limits)), first=g.TIMESTAMP.min(), last=g.TIMESTAMP.max(), unit=g.unit.iloc[0]))
    return pd.DataFrame(rows)

def chronological_split(times, fraction: float=0.7) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Fixed per-target split; predictor coverage and outcome values never set the boundary."""
    stamps = pd.DatetimeIndex(times).unique().sort_values()
    if not 0 < fraction < 1 or len(stamps) < 2:
        raise ValueError('Need at least two timestamps and a training fraction in (0,1)')
    cut = int(math.floor(len(stamps) * fraction))
    if cut < 1 or cut >= len(stamps):
        raise ValueError('Empty chronological partition')
    return (stamps[:cut], stamps[cut:])

def expanding_folds(times, folds: int=3):
    """Initial half for training, remaining observations in chronological validation blocks."""
    stamps = pd.DatetimeIndex(times).unique().sort_values()
    start = len(stamps) // 2
    if start < 3 or len(stamps) - start < folds:
        raise ValueError('Insufficient training observations for expanding validation')
    for validation in np.array_split(np.arange(start, len(stamps)), folds):
        yield (stamps[:int(validation[0])], stamps[validation])

def evaluation_metrics(y, prediction):
    y, prediction = (np.asarray(y, dtype=float), np.asarray(prediction, dtype=float))
    if y.shape != prediction.shape or y.ndim != 1 or (not len(y)):
        raise ValueError('Nonempty aligned one-dimensional observations/predictions required')
    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
        raise ValueError('Choose and record the finite comparison set before scoring')
    error = prediction - y
    rmse = float(np.sqrt(np.mean(error ** 2)))
    sigma = float(np.std(y, ddof=0)) if np.ptp(y) > 0 else 0.0
    return dict(n=len(y), rmse=rmse, mae=float(np.mean(np.abs(error))), sigma_evaluation=sigma, nrmse=rmse / sigma if sigma > 0 else float('nan'), r2=1 - float(np.mean(error ** 2)) / sigma ** 2 if sigma > 0 else float('nan'))

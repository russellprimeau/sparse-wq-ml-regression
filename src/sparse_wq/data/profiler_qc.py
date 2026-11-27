"""Native-record profiler QC: sensor failures, review flags and shallow depth.

Fixed gross-range rules implement manuscript Table 3. A review flag never
automatically removes an observation. No learned scaling or gap filling occurs.
The conductivity sensor is wiped (calibration serial 19F100067); its specified
span applies to measured conductivity, not directly to calculated specific
conductance. The latter has a nonnegative physical check and inherits failures
of its measured conductivity input.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
PASS, NOT_EVALUATED, REVIEW, FAIL, MISSING = (1, 2, 3, 4, 9)
MISSING_TOKENS = ('', 'NAN', 'NaN', 'nan')
SHALLOW_REFERENCE_M = 2.94
SHALLOW_TOLERANCE_M = 0.4
SPECIFICATION_URL = 'https://www.xylem.com/siteassets/brand/ysi/resources/brochure/ysi-exo-brochure.pdf'

@dataclass(frozen=True)
class Rule:
    label: str
    sensor_min: float | None
    sensor_max: float | None
    review_min: float | None
    review_max: float | None
    companion_column: str | None = None
    companion_min: float | None = None
    companion_max: float | None = None
RULES = {'sensorParms(1)': Rule('Water temperature (°C)', -5, 50, 1, 25), 'sensorParms(3)': Rule('Sp Cond (microS_cm)', 0, None, 1, None, 'sensorParms(2)', 0, 100000), 'sensorParms(5)': Rule('pH', 0, 14, 2, 12), 'sensorParms(6)': Rule('DO (% Sat)', 0, 500, 10, 120), 'sensorParms(8)': Rule('Turbidity (FNU)', 0, 4000, 0, None), 'sensorParms(10)': Rule('fDOM (RFU)', 0, None, 0, 100), 'sensorParms(11)': Rule('fDOM (QSU)', 0, 300, 0, 300)}

def configuration() -> dict:
    return dict(rules={col: asdict(rule) for col, rule in RULES.items()}, flag_codes=dict(pass_=PASS, not_evaluated=NOT_EVALUATED, review=REVIEW, fail=FAIL, missing=MISSING), shallow_reference_m=SHALLOW_REFERENCE_M, shallow_tolerance_m=SHALLOW_TOLERANCE_M, sensor_specification=SPECIFICATION_URL, conductivity_evidence='Wiped Conductivity And Temperature, sensor serial 19F100067; calibration exports', policy='mask sensor failures; retain review flags; retain original timestamps and precision; no interpolation')

def numbers(series: pd.Series) -> pd.Series:
    """Reject unrecognized tokens; recognize logger missingness and sentinel."""
    return pd.to_numeric(series.replace(dict.fromkeys(MISSING_TOKENS, np.nan)), errors='raise').astype(float).replace(-9999, np.nan)

def outside(values: pd.Series, lower, upper) -> pd.Series:
    bad = values.notna() & ~np.isfinite(values)
    if lower is not None:
        bad |= values.lt(lower)
    if upper is not None:
        bad |= values.gt(upper)
    return bad

def screen_channels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen seven predictors per cell, preserving auxiliary source columns.

    Returns masked source-schema records, seven QC-code columns and an audit
    of all non-pass cells. Depth eligibility is handled separately. Missing
    companion conductivity prevents evaluation of that span test, rather than
    being treated as a successful test. Such a value is retained with code 2
    unless another test produces a review or failure flag.
    """
    if not frame.index.is_unique:
        raise ValueError('QC requires unique logger keys')
    result, flags, events = (frame.copy(deep=True), pd.DataFrame(index=frame.index), [])
    for column, rule in RULES.items():
        value = numbers(frame[column])
        missing = value.isna()
        bad = outside(value, rule.sensor_min, rule.sensor_max)
        reason = pd.Series('', index=frame.index, dtype=object)
        unevaluated = pd.Series(False, index=frame.index)
        if rule.companion_column:
            companion = numbers(frame[rule.companion_column])
            companion_bad = outside(companion, rule.companion_min, rule.companion_max) & ~missing
            unevaluated = companion.isna() & ~missing & ~bad
            bad |= companion_bad
            reason.loc[unevaluated] = 'conductivity_span_not_evaluated'
            reason.loc[companion_bad] = 'measured_conductivity_outside_sensor_range'
        own_bad = outside(value, rule.sensor_min, rule.sensor_max)
        reason.loc[own_bad] = 'outside_sensor_or_physical_range'
        review = outside(value, rule.review_min, rule.review_max) & ~bad & ~missing
        reason.loc[review] = 'outside_review_range'
        reason.loc[missing] = 'missing_value_or_sentinel'
        codes = pd.Series(PASS, index=frame.index, dtype='int8')
        codes.loc[unevaluated] = NOT_EVALUATED
        codes.loc[review] = REVIEW
        codes.loc[bad] = FAIL
        codes.loc[missing] = MISSING
        flags[column] = codes
        result.loc[codes.isin([FAIL, MISSING]), column] = 'NAN'
        for key in reason.index[reason.ne('')]:
            event = dict(TIMESTAMP=key[0], RECORD=key[1], column=column, channel=rule.label, original_token=frame.at[key, column], qc_flag=int(codes.at[key]), reason=reason.at[key])
            if rule.companion_column:
                event.update(companion_column=rule.companion_column, companion_token=frame.at[key, rule.companion_column])
            events.append(event)
    columns = ['TIMESTAMP', 'RECORD', 'column', 'channel', 'original_token', 'qc_flag', 'reason', 'companion_column', 'companion_token']
    return (result, flags, pd.DataFrame(events, columns=columns))

def shallow_selection(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed 2.54--3.34 m inclusive band around the documented nominal depth.

    The fixed deployment band is independent of model partitions and of later
    data supplied to this function. The observed median is reported for audit,
    but never refitted here to change eligibility.
    """
    depth = numbers(frame['sensorParms(9)'])
    lower = round(SHALLOW_REFERENCE_M - SHALLOW_TOLERANCE_M, 9)
    upper = round(SHALLOW_REFERENCE_M + SHALLOW_TOLERANCE_M, 9)
    keep = depth.between(lower, upper, inclusive='both')
    flags = pd.Series(PASS, index=frame.index, dtype='int8')
    flags.loc[~keep] = FAIL
    flags.loc[depth.isna()] = MISSING
    return pd.DataFrame(dict(depth_m=depth, depth_qc_flag=flags, eligible=keep), index=frame.index)

def screening_summary(flags: pd.DataFrame) -> list[dict]:
    rows = []
    for column, rule in RULES.items():
        codes = flags[column]
        recorded = int(codes.ne(MISSING).sum())
        n_failed, n_review = (int(codes.eq(FAIL).sum()), int(codes.eq(REVIEW).sum()))
        rows.append(dict(column=column, channel=rule.label, rows=len(codes), recorded=recorded, missing=int(codes.eq(MISSING).sum()), excluded=n_failed, review_flagged=n_review, not_evaluated=int(codes.eq(NOT_EVALUATED).sum()), retained=int(codes.isin([PASS, NOT_EVALUATED, REVIEW]).sum()), excluded_percent=100 * n_failed / recorded if recorded else None, review_percent=100 * n_review / recorded if recorded else None))
    return rows

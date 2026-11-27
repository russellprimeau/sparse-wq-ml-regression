"""Seasonal day/time averaging of explicitly supplied eligible earlier-year history."""
from __future__ import annotations
import numpy as np
import pandas as pd
SEASONAL_DAY_WINDOWS = (4, 15)
SEASONAL_DIURNAL_WINDOW_H = 1

def add_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add YEAR, DAYOFYEAR and fractional HOUR from TIMESTAMP (in place)."""
    frame['YEAR'] = frame['TIMESTAMP'].dt.year
    frame['DAYOFYEAR'] = frame['TIMESTAMP'].dt.dayofyear
    frame['HOUR'] = frame['TIMESTAMP'].dt.hour + frame['TIMESTAMP'].dt.minute / 60.0
    return frame

def seasonal_window_mean(days: np.ndarray, hours: np.ndarray, values: np.ndarray, target_day: int, target_hour: float, diurnal_window: float=SEASONAL_DIURNAL_WINDOW_H, day_windows: tuple[int, int]=SEASONAL_DAY_WINDOWS) -> float:
    """Seasonal forecast for one day of year and hour, from the supplied (other-year) values.

    Day differences wrap around the year end (day 365 is one day from day 1).
    """
    day_diff = np.abs(days - target_day)
    day_diff = np.minimum(day_diff, 365 - day_diff)
    hour_diff = np.abs(hours - target_hour)
    for mask in ((day_diff <= day_windows[0]) & (hour_diff <= diurnal_window), day_diff <= day_windows[0], day_diff <= day_windows[1]):
        found = values[mask]
        found = found[np.isfinite(found)]
        if found.size >= 3:
            return float(np.mean(found))
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan

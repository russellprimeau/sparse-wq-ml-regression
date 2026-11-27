"""One predictor window per independent laboratory observation."""
from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
import numpy as np


class Window(NamedTuple):
    inputs: np.ndarray   # (rows, features) after any summary
    target: np.ndarray   # (output rows, output columns)
    name: str            # file name of the window



def base_name(name: str) -> str:
    """Observation identifier used for chronological memberships."""
    return Path(str(name)).name



def flatten(windows: list[Window], dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """(inputs, targets): one row per window, the input block flattened hour by hour."""
    x = np.array([w.inputs.flatten() for w in windows], dtype=dtype)
    y = np.array([w.target.flatten() for w in windows], dtype=dtype)
    return x, (y.reshape(-1, 1) if y.ndim == 1 else y)

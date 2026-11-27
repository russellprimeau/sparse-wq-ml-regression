"""Physical constraints applied to reconstructed measured-level predictions."""
from __future__ import annotations
import numpy as np
from .data.qualified import TARGETS

POLICY = 'nonnegative-measured-level-v1'
NONNEGATIVE_TARGETS = frozenset(TARGETS.values()) - {'pH'}


def constrain_predictions(values, target):
    """Preserve pH; floor other registered nonnegative assays at zero.

    Apply after inverse scaling and differential reconstruction, separately to
    each seeded forecast before averaging. Historical training minima/maxima
    are scaling parameters only, never prediction bounds.
    """
    values = np.asarray(values, dtype=float)
    if target == 'pH':
        return values.copy()
    if target not in NONNEGATIVE_TARGETS:
        raise ValueError('No physical output constraint registered for '+str(target))
    return np.maximum(values, 0.0)

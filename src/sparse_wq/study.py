"""The study definition in config/study.yml, and what it implies for each run.

    targets          the 14 laboratory targets and their directory names
    predictors(run)  the hourly predictors a run offers, before the target's prior value
    output_column    the column a run's models predict for a target
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from sparse_wq import design, paths


@lru_cache(maxsize=1)
def load() -> dict:
    with open(paths.STUDY_YML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def targets() -> dict[str, str]:
    """Target name in the consolidated record -> short name used for its directories."""
    return dict(load()["targets"])


def predictors(run: str) -> list[str]:
    """The hourly predictors `run` offers, in the order they are written to each window."""
    groups = load()["predictors"]
    return list(groups["always"]) + (list(groups["profiler"]) if design.offers_profiler(run) else [])


def output_column(target: str, run: str) -> str:
    """The measured value in an absolute run; its change over the window in a differential one."""
    return target if design.is_absolute(run) else f"{target}_diff"


def prior_value_column(target: str) -> str:
    """The most recent laboratory value of `target` at each hour (Section 2.3)."""
    return f"{target}_state"


def offers_prior_value(run: str) -> bool:
    return design.RUNS[run][2] == "offered"


def window_settings() -> dict:
    return dict(load()["windows"])

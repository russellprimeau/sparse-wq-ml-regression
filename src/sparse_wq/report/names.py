"""Display labels for the study's 31 variables.

The column names in the consolidated record are the data contract: windows, model
configurations and stored results all refer to them, so they are never renamed. Some carry
logger conventions that should not appear in a printed figure (``microS_cm``, ``W/m2``,
``mBar``). Figures therefore ask this registry for a label instead of editing column names.

Two rules:

* A label carries its unit when the figure plots values of the variable, and drops it when
  the figure plots a statistic about a model of it (an R², an inclusion flag): see
  ``with_unit``.
* The source qualifier (Surface / SCADA / lab) is added only for the three quantities that
  more than one dataset measures: pH, turbidity and water temperature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SURFACE = "Surface"
SCADA = "SCADA"
WEATHER = "Weather"
SAMPLES = "Samples"


@dataclass(frozen=True)
class Param:
    """Display metadata for one stored column.

    ``short`` is for ticks and legends, ``long`` for axis titles. ``unit`` is matplotlib
    mathtext, or None for dimensionless quantities. ``unit_is_identity`` marks the two fDOM
    channels, which differ only in their unit, so it is shown even when units are dropped.
    """

    short: str
    long: str
    unit: str | None
    group: str
    qualifier: str | None = None
    unit_is_identity: bool = False


def _p(short, unit, group, long=None, qualifier=None, unit_is_identity=False):
    return Param(short=short, long=long or short, unit=unit, group=group,
                 qualifier=qualifier, unit_is_identity=unit_is_identity)


# Surface sonde on the vertical profiler (about 2 m depth); stored with a "Pfl - " prefix.
_SURFACE: dict[str, Param] = {
    "Pfl - Water temperature (°C)": _p("Water temp.", "°C", SURFACE, "Water temperature", qualifier=SURFACE),
    "Pfl - Sp Cond (microS_cm)": _p("Specific conductance", r"$\mu$S/cm", SURFACE),
    "Pfl - pH": _p("pH", None, SURFACE, qualifier=SURFACE),
    "Pfl - DO (% Sat)": _p("Dissolved oxygen", "% sat.", SURFACE),
    "Pfl - Turbidity (FNU)": _p("Turbidity", "FNU", SURFACE, qualifier=SURFACE),
    "Pfl - fDOM (RFU)": _p("fDOM", "RFU", SURFACE, unit_is_identity=True),
    "Pfl - fDOM (QSU)": _p("fDOM", "QSU", SURFACE, unit_is_identity=True),
    # In the sonde export but not among the predictors.
    "Pfl - Cond (microS_cm)": _p("Conductivity", r"$\mu$S/cm", SURFACE),
    "Pfl - Salinity (ppt)": _p("Salinity", "ppt", SURFACE),
    "Pfl - Turbidity (NTU)": _p("Turbidity, NTU", "NTU", SURFACE, qualifier=SURFACE),
    "Pfl - Vertical position (m)": _p("Sonde depth", "m", SURFACE),
}

# Treatment-plant raw-water intake.
_SCADA: dict[str, Param] = {
    "SCADA - pH": _p("pH", None, SCADA, qualifier=SCADA),
    "SCADA - Temperature (°C)": _p("Water temp.", "°C", SCADA, "Water temperature", qualifier=SCADA),
}

# Local weather station, gaps filled from MET Nordic.
_WEATHER: dict[str, Param] = {
    # x = cos(bearing) * speed and y = sin(bearing) * speed on the meteorological "from"
    # bearing; not the usual eastward/northward components, so named neutrally.
    "Wind speed x (m/s)": _p("Wind speed, x", "m/s", WEATHER, "Wind speed, x-component"),
    "Wind speed y (m/s)": _p("Wind speed, y", "m/s", WEATHER, "Wind speed, y-component"),
    "Atmospheric pressure (mBar)": _p("Atmospheric pressure", "mbar", WEATHER),
    "Longwave (IR) radiation (W/m2)": _p("Longwave (IR) irradiance", r"W/m$^2$", WEATHER),
    "Shortwave (solar) radiation (W/m2)": _p("Shortwave (solar) irradiance", r"W/m$^2$", WEATHER),
    "24hr precipitation total (mm)": _p("Precipitation, 24 h total", "mm", WEATHER),
    "Precipitation (mm/hr)": _p("Precipitation rate", "mm/h", WEATHER),
    "Air temperature (°C)": _p("Air temperature", "°C", WEATHER),
    "Humidity (%)": _p("Relative humidity", "%", WEATHER),
    "Maximum 3s wind gust (m/s)": _p("Wind gust, 3 s", "m/s", WEATHER),
}

# Laboratory analyses (the targets). Incubation temperatures distinguish the two culture
# counts and are part of the name; a non-breaking space keeps value and unit together.
_SAMPLES: dict[str, Param] = {
    "Color": _p("Color", "mg Pt/L", SAMPLES),
    "Turbidity (FNU)": _p("Turbidity", "FNU", SAMPLES, qualifier="lab"),
    "pH": _p("pH", None, SAMPLES, qualifier="lab"),
    "E.coli (CFU/100mL)": _p("$\\it{E.\\ coli}$", "CFU/100 mL", SAMPLES),
    "Intestinal enterococci (CFU/100mL)": _p("Intestinal enterococci", "CFU/100 mL", SAMPLES),
    "Colony Count 22°C (CFU/mL)": _p("Colony count, 22 °C", "CFU/mL", SAMPLES),
    "Total coliforms 37°C (CFU/100mL)": _p("Total coliforms, 37 °C", "CFU/100 mL", SAMPLES),
    "Arsenic (µg/L)": _p("Arsenic", r"$\mu$g/L", SAMPLES),
    "Lead (µg/L)": _p("Lead", r"$\mu$g/L", SAMPLES),
    "Cadmium (µg/L)": _p("Cadmium", r"$\mu$g/L", SAMPLES),
    "Copper filtered (mg/L)": _p("Copper, filtered", "mg/L", SAMPLES),  # mg/L, unlike the other metals
    "Chromium (µg/L)": _p("Chromium", r"$\mu$g/L", SAMPLES),
    "Nickel (µg/L)": _p("Nickel", r"$\mu$g/L", SAMPLES),
    "Zinc (µg/L)": _p("Zinc", r"$\mu$g/L", SAMPLES),
}

PARAMS: dict[str, Param] = {**_SURFACE, **_SCADA, **_WEATHER, **_SAMPLES}

# A target column may be the prior value (_state), the change since the window start
# (_diff) or the change since the previous sample (_res); each keeps the base unit.
_SUFFIX_TEMPLATES = {"_state": "{}, previous value", "_diff": "Δ{}", "_res": "{}, residual"}


def _split_suffix(col: str) -> tuple[str, str | None]:
    for sfx in _SUFFIX_TEMPLATES:
        if col.endswith(sfx):
            return col[: -len(sfx)], sfx
    return col, None


def slug(col: str) -> str:
    """File-safe form of a column name: every non-word character becomes "_"."""
    return re.sub(r"[^\w]", "_", col)


_SLUG_TO_COL: dict[str, str] = {slug(c): c for c in PARAMS}


def _lookup(col: str) -> tuple[Param | None, str | None]:
    base, sfx = _split_suffix(str(col))
    param = PARAMS.get(base) or PARAMS.get(_SLUG_TO_COL.get(slug(base), ""))
    return param, sfx


def unit(col: str) -> str | None:
    """Unit of a column, or None for a dimensionless quantity or an unknown column."""
    param, _ = _lookup(col)
    return None if param is None else param.unit


def label(col: str, *, with_unit: bool = True, long: bool = False, qualified: bool = True,
          with_suffix: bool = True) -> str:
    """Display label for a stored column name; unknown columns are returned unchanged."""
    param, sfx = _lookup(col)
    if param is None:
        return str(col)
    text = param.long if long else param.short
    if qualified and param.qualifier:
        text = f"{text} ({param.qualifier})"
    if with_suffix and sfx is not None:
        text = _SUFFIX_TEMPLATES[sfx].format(text)
    if with_unit or param.unit_is_identity:
        u = unit(col)
        if u:
            text = f"{text} ({u})"
    return text


def axis_label(col: str, *, with_unit: bool = True, qualified: bool = True) -> str:
    """Long-form label, for axis titles."""
    return label(col, with_unit=with_unit, long=True, qualified=qualified)


def clean_target_label(dataset_name: str, prefix: str = "MC", *, with_suffix: bool = False,
                       qualified: bool = False) -> str:
    """Target label (no unit) from a per-target dataset name such as ``MC_Lead__µg_L__diff``."""
    name = str(dataset_name)
    bare = prefix.rstrip("_")
    if bare and name.startswith(bare + "_"):
        name = name[len(bare) + 1:]
    if name.startswith("ex"):
        name = name[2:]
    sfx = None
    for candidate in _SUFFIX_TEMPLATES:
        if name.endswith(candidate):
            name, sfx = name[: -len(candidate)], candidate
            break
    col = _SLUG_TO_COL.get(name)
    if col is None:
        squashed = re.sub(r"_+", "_", name).strip("_")
        col = next((c for s, c in _SLUG_TO_COL.items() if re.sub(r"_+", "_", s).strip("_") == squashed), None)
    if col is None:
        return re.sub(r"_+", " ", name).strip()
    param = PARAMS[col]
    text = param.short
    if qualified and param.qualifier:
        text = f"{text} ({param.qualifier})"
    if with_suffix and sfx is not None:
        text = _SUFFIX_TEMPLATES[sfx].format(text)
    return text

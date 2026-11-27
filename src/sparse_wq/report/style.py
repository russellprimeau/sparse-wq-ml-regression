"""Shared figure style.

Figures are drawn at their printed size where possible, so the text sizes set here are the
sizes on the page. Titles are never drawn: captions belong to the article.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Width of the single-column figures (inches) and output resolution.
PAGE_WIDTH_IN = 6.5
FIGURE_DPI = 300

# Clear gap between the top of the plotting area and the bottom of a legend above it.
LEGEND_GAP_IN = 0.12

_BASE_FONT_PT = 8
_BASE_LABEL_PT = 9


def apply_paper_style() -> None:
    """Install the shared matplotlib settings."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": _BASE_FONT_PT,
        "axes.labelsize": _BASE_LABEL_PT,
        "axes.titlesize": _BASE_LABEL_PT,
        "xtick.labelsize": _BASE_FONT_PT,
        "ytick.labelsize": _BASE_FONT_PT,
        "legend.fontsize": _BASE_FONT_PT,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "axes.grid": False,
        "figure.autolayout": False,
    })


def clear_figure_titles(fig) -> None:
    """Remove the figure and axes titles."""
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")


def legend_above(target, handles=None, labels=None, *, ncol: int | None = None, fontsize=None,
                 gap_in: float = LEGEND_GAP_IN, frameon: bool = False, **kwargs):
    """Legend in the margin above the plotting area, anchored by its bottom edge.

    Anchoring the bottom edge a fixed distance above the topmost axes makes the legend grow
    away from the plot as it gains rows. Call after the layout is final.
    """
    fig = getattr(target, "figure", target)
    tops = [ax.get_position().y1 for ax in fig.axes if ax.get_visible()]
    top = max(tops) if tops else 1.0
    y_anchor = top + gap_in / (float(fig.get_size_inches()[1]) or 1.0)

    if handles is None:
        source = target if target is not fig else (fig.axes[0] if fig.axes else fig)
        handles, auto_labels = source.get_legend_handles_labels()
        if labels is None:
            labels = auto_labels
    if ncol is None:
        ncol = max(1, len(handles))
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    kwargs["handles"] = handles
    if labels is not None:
        kwargs["labels"] = labels
    return fig.legend(loc="lower center", bbox_to_anchor=(0.5, y_anchor), ncol=ncol,
                      frameon=frameon, borderaxespad=0.0, **kwargs)


def save_figure(fig, path, *, dpi: int = FIGURE_DPI, **kwargs) -> Path:
    """Remove titles and save tightly cropped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clear_figure_titles(fig)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.02)
    fig.savefig(path, dpi=dpi, **kwargs)
    return path

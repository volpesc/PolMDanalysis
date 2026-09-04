"""
mdplot.style
============
Shared visual style, colour palette, and figure/axes factory functions.

All observable modules import from here so that changing one value propagates
everywhere. Nothing in this module depends on simulation data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple


# ── Colour palette (colour-blind friendly, print-safe) ────────────────────────
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}

# Ordered list for cycling
COLOR_CYCLE = [
    PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
    PALETTE["red"],  PALETTE["purple"], PALETTE["sky"],
]

# ── rcParams ─────────────────────────────────────────────────────────────────
RC = {
    "font.family":        "serif",
    "font.size":          12,
    "axes.labelsize":     13,
    "axes.titlesize":     13,
    "axes.prop_cycle":    plt.cycler("color", COLOR_CYCLE),
    "legend.fontsize":    11,
    "legend.framealpha":  0.85,
    "lines.linewidth":    2.0,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}


def apply() -> None:
    """Apply the shared style globally. Call once at the start of a session."""
    plt.rcParams.update(RC)


# ── Figure / axes factory ─────────────────────────────────────────────────────

def figure(
    ncols: int = 1,
    nrows: int = 1,
    width: float = 7.0,
    aspect: float = 0.714,   # golden ratio ≈ 1/1.4
    **kwargs,
) -> Tuple[Figure, any]:
    """
    Create a styled figure.

    Parameters
    ----------
    ncols, nrows : int
        Grid dimensions.
    width : float
        Total figure width in inches.
    aspect : float
        Height/width ratio per panel.
    **kwargs
        Forwarded to ``plt.subplots``.

    Returns
    -------
    fig, axes
        Same as ``plt.subplots``.
    """
    apply()
    h = width * aspect * nrows / ncols if ncols > 1 else width * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, h), **kwargs)
    return fig, axes


def label_axes(ax: Axes, xlabel: str, ylabel: str,
               title: Optional[str] = None) -> None:
    """Set axis labels and optional title in one call."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)


def save(fig: Figure, path: str) -> None:
    """Save figure and print confirmation."""
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")


def reference_line(ax: Axes, slope: float, x, y_anchor, **kwargs) -> None:
    """
    Draw a power-law reference line y ∝ x^slope through the point (x_mid, y_anchor).

    Parameters
    ----------
    ax : Axes
    slope : float
        Exponent of the reference line.
    x : array-like
        x values to plot over.
    y_anchor : float
        y value at x[len(x)//2].
    """
    import numpy as np
    x = np.asarray(x)
    x = x[x > 0]
    mid = x[len(x) // 2]
    defaults = dict(color="gray", lw=1.2, zorder=1)
    defaults.update(kwargs)
    label = kwargs.pop("label", rf"$\sim t^{{{slope}}}$")
    ax.plot(x, y_anchor * (x / mid) ** slope, label=label, **defaults)

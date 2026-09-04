"""
mdplot.observables.ppa
======================
Plotting and parsing for Primitive Path Analysis (PPA) output.

PPA output uses a key=value format rather than columns, so this module
provides both a dedicated loader and a bar-chart summary plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from matplotlib.axes import Axes
import numpy as np

from mdplot import style


def load_ppa(path: str | Path) -> Dict[str, float]:
    """
    Parse a PPA key=value output file.

    Returns
    -------
    dict with keys ``bpp``, ``app``, ``Ne``.
    """
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = float(val.strip())
    return result


def plot_ppa(
    data: Dict[str, float],
    ax: Optional[Axes] = None,
    *,
    label: str = "",
) -> Axes:
    """
    Bar-chart summary of PPA results: bpp, app, Ne.

    Parameters
    ----------
    data : dict
        Output of ``load_ppa``.
    ax : Axes, optional
    label : str
        Optional title suffix.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure(width=5, aspect=0.9)

    keys   = ["bpp", "app", "Ne"]
    values = [data.get(k, 0.0) for k in keys]
    xlabels = [
        r"$b_{pp}$ [$\sigma$]",
        r"$a_{pp}$ [$\sigma$]",
        r"$N_e$ [monomers]",
    ]
    colors = list(style.COLOR_CYCLE[:3])

    bars = ax.bar(xlabels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    title = "Primitive Path Analysis"
    if label:
        title += f" — {label}"
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y")
    return ax

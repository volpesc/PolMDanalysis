"""
mdplot.observables.brush_length
================================
Plotting functions for the brush penetration length distribution P(L).
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes
from mdplot import style


def plot_brush_length(
    datasets: Dict[str, np.ndarray] | List[Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    *,
    fill: bool = True,
) -> Axes:
    """
    Plot the brush penetration length distribution P(L).

    Parameters
    ----------
    datasets : dict or list of dicts
        One or more outputs of ``io.load_brush_length``.
        Keys: ``L``, ``prob``.
    labels : list of str, optional
    ax : Axes, optional
    fill : bool
        Fill area under curves.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    if isinstance(datasets, dict):
        datasets = [datasets]
    if labels is None:
        labels = [r"$P(L)$"] * len(datasets)

    for data, lbl in zip(datasets, labels):
        ax.plot(data["L"], data["prob"], label=lbl)
        if fill:
            ax.fill_between(data["L"], data["prob"], alpha=0.12)

    ax.set_ylim(bottom=0)
    style.label_axes(ax,
                     xlabel=r"$L$ [$\sigma$]",
                     ylabel=r"$P(L)$",
                     title="Brush Penetration Length Distribution")
    ax.legend()
    return ax

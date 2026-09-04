"""
mdplot.observables.sq
=====================
Plotting functions for the static structure factor S(q).
"""

from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_sq(
    datasets: Dict[str, np.ndarray] | List[Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    *,
    loglog: bool = False,
) -> Axes:
    """
    Plot one or more S(q) curves on the same axes.

    Parameters
    ----------
    datasets : dict or list of dicts
        One or more outputs of ``io.load_sq``.
    labels : list of str, optional
        Legend labels (one per dataset).
    ax : Axes, optional
        Target axes.  Created if not provided.
    loglog : bool
        Use log-log scaling.

    Returns
    -------
    Axes

    Examples
    --------
    Plot a single curve::

        ax = plot_sq(load_sq("Sc_q.dat"), labels=["Sc(q)"])

    Compare two curves::

        ax = plot_sq(
            [load_sq("Sc_q.dat"), load_sq("Stot_q.dat")],
            labels=["Sc(q)", "S(q)"],
        )
    """
    if ax is None:
        _, ax = style.figure()

    # Normalise to a list
    if isinstance(datasets, dict):
        datasets = [datasets]
    if labels is None:
        labels = [r"$S(q)$"] * len(datasets)

    for data, label in zip(datasets, labels):
        ax.plot(data["q"], data["sq"], label=label)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    style.label_axes(ax,
                     xlabel=r"$q$ [$\sigma^{-1}$]",
                     ylabel=r"$S(q)$",
                     title="Static Structure Factor")
    ax.legend()
    return ax

"""
mdplot.observables.entanglement
================================
Plotting functions for spatial entanglement / kink density.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_entanglement(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    quantity: str = "ent_per_chain",
    label: str = "",
    fill: bool = True,
) -> Axes:
    """
    Plot entanglement density as a function of position x.

    Parameters
    ----------
    data : dict
        Output of ``io.load_entanglement``.
        Keys: ``x``, ``kink_count``, ``chains_in_bin``, ``ent_per_chain``.
    ax : Axes, optional
    quantity : str
        Which column to plot: ``"ent_per_chain"`` (default),
        ``"kink_count"``, or ``"chains_in_bin"``.
    label : str
        Legend label.
    fill : bool
        Fill area under the curve.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ylabels = {
        "ent_per_chain":  r"Entanglements per chain",
        "kink_count":     r"Kink count",
        "chains_in_bin":  r"Chains in bin",
    }
    y = data[quantity]
    lbl = label or ylabels.get(quantity, quantity)

    ax.plot(data["x"], y, label=lbl)
    if fill:
        ax.fill_between(data["x"], y, alpha=0.15)

    style.label_axes(ax,
                     xlabel=r"$x$ [$\sigma$]",
                     ylabel=ylabels.get(quantity, quantity),
                     title="Spatial Entanglement Density")
    ax.set_ylim(bottom=0)
    ax.legend()
    return ax

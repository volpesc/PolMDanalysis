"""
mdplot.observables.backbone
============================
Plotting functions for the FENE backbone bond-force profile.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes
from mdplot import style


def plot_backbone(
    datasets: Dict[str, np.ndarray] | List[Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    *,
    zero_line: bool = True,
) -> Axes:
    """
    Plot projected FENE bond force as a function of bond index.

    Parameters
    ----------
    datasets : dict or list of dicts
        One or more outputs of ``io.load_backbone``.
        Keys: ``bond_index``, ``F_proj``.
    labels : list of str, optional
        Legend labels.
    ax : Axes, optional
    zero_line : bool
        Draw a horizontal line at F=0.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    if isinstance(datasets, dict):
        datasets = [datasets]
    if labels is None:
        labels = [r"$F_\mathrm{proj}$"] * len(datasets)

    for data, lbl in zip(datasets, labels):
        ax.plot(data["bond_index"], data["F_proj"], label=lbl)

    if zero_line:
        ax.axhline(0, color="gray", lw=0.8, ls="--")

    style.label_axes(ax,
                     xlabel="Bond index $i$",
                     ylabel=r"$F_\mathrm{proj}$ [$\varepsilon/\sigma$]",
                     title="Backbone Bond-Force Profile")
    ax.legend()
    return ax

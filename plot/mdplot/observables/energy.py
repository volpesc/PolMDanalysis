"""
mdplot.observables.energy
==========================
Plotting functions for polymer-solvent interaction energy vs lambda.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes
from mdplot import style


def plot_energy(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    label: str = r"$E(\lambda)$",
    marker: str = "o",
) -> Axes:
    """
    Plot total interaction energy as a function of coupling lambda.

    Parameters
    ----------
    data : dict
        Output of ``io.load_energy``.  Keys: ``lam``, ``energy``.
    ax : Axes, optional
    label : str
        Legend label.
    marker : str
        Matplotlib marker style.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["lam"], data["energy"], marker=marker,
            ms=5, label=label)
    ax.axhline(0, color="gray", lw=0.8, ls="--")

    style.label_axes(ax,
                     xlabel=r"$\lambda$",
                     ylabel=r"$E$ [$\varepsilon$]",
                     title="Polymer–Solvent Interaction Energy")
    ax.legend()
    return ax

"""
mdplot.observables.spec_density
================================
Plotting functions for species radial density profiles.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from matplotlib.axes import Axes
from mdplot import style


def plot_spec_density(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    fill: bool = True,
) -> Axes:
    """
    Plot polymer and solvent volume-fraction profiles rho(r).

    Parameters
    ----------
    data : dict
        Output of ``io.load_spec_density``.
        Keys: ``r``, ``rho_poly``, ``rho_solv``.
    ax : Axes, optional
    fill : bool
        Fill area under curves.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["r"], data["rho_poly"], label=r"$\rho_\mathrm{polymer}(r)$")
    if fill:
        ax.fill_between(data["r"], data["rho_poly"], alpha=0.15)

    if "rho_solv" in data:
        ax.plot(data["r"], data["rho_solv"], label=r"$\rho_\mathrm{solvent}(r)$")
        if fill:
            ax.fill_between(data["r"], data["rho_solv"], alpha=0.15)

    ax.set_ylim(bottom=0)
    style.label_axes(ax,
                     xlabel=r"$r$ [$\sigma$]",
                     ylabel=r"$\rho(r)$ [volume fraction]",
                     title="Species Radial Density")
    ax.legend()
    return ax

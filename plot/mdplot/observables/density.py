"""
mdplot.observables.density
==========================
Plotting functions for the spatial density profile rho(x).
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_density(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    show_total: bool = True,
) -> Axes:
    """
    Plot rho(x) for polymer and (optionally) solvent particles.

    Parameters
    ----------
    data : dict
        Output of ``io.load_density``.
    ax : Axes, optional
        Target axes.  Created if not provided.
    show_total : bool
        If solvent data is present, also plot the total density.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["x"], data["rho_poly"], label=r"$\rho_\mathrm{polymer}(x)$")

    if "rho_solv" in data:
        ax.plot(data["x"], data["rho_solv"], ls="--",
                label=r"$\rho_\mathrm{solvent}(x)$")
        if show_total:
            ax.plot(data["x"], data["rho_poly"] + data["rho_solv"],
                    ls=":", lw=1.5, color="gray",
                    label=r"$\rho_\mathrm{total}(x)$")

    ax.set_ylim(bottom=0)
    style.label_axes(ax,
                     xlabel=r"$x$ [$\sigma$]",
                     ylabel=r"$\rho(x)$ [$\sigma^{-3}$]",
                     title="Density Profile")
    ax.legend()
    return ax

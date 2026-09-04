"""
mdplot.observables.gds_diffusion
=================================
Plotting functions for GDS diffusion front analysis.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mdplot import style


def plot_gds_fronts(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    show_both: bool = True,
) -> Axes:
    """
    Plot GDS front positions vs time.

    Parameters
    ----------
    data : dict
        Output of ``io.load_gds``.
        Keys: ``time``, ``solvent_front``, ``polymer_front``,
              ``solvent_uptake``, ``polymer_uptake``.
    ax : Axes, optional
    show_both : bool
        If False, plot only solvent front.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["time"], data["solvent_front"],
            label="Solvent front")
    if show_both:
        ax.plot(data["time"], data["polymer_front"],
                label="Polymer front", ls="--")

    style.label_axes(ax,
                     xlabel=r"$t$ [simulation units]",
                     ylabel=r"$x_\mathrm{GDS}$ [$\sigma$]",
                     title="GDS Diffusion Front")
    ax.legend()
    return ax


def plot_gds_uptake(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot polymer and solvent uptake vs time.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["time"], data["solvent_uptake"], label="Solvent uptake")
    ax.plot(data["time"], data["polymer_uptake"], label="Polymer uptake", ls="--")

    style.label_axes(ax,
                     xlabel=r"$t$ [simulation units]",
                     ylabel="Particle count",
                     title="Uptake vs Time")
    ax.legend()
    return ax


def plot_gds(
    data: Dict[str, np.ndarray],
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Convenience wrapper: front positions and uptake side by side.

    Returns
    -------
    fig, (ax_front, ax_uptake)
    """
    fig, (ax1, ax2) = style.figure(ncols=2, width=12)
    plot_gds_fronts(data, ax=ax1)
    plot_gds_uptake(data, ax=ax2)
    return fig, (ax1, ax2)

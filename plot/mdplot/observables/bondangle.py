"""
mdplot.observables.bondangle
============================
Plotting functions for the bond angle distribution P(theta).
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mdplot import style


def plot_bondangle(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    solid_angle: bool = False,
) -> Axes:
    """
    Plot the bond angle distribution P(θ).

    Parameters
    ----------
    data : dict
        Output of ``io.load_bondangle``.
    ax : Axes, optional
        Target axes.  Created if not provided.
    solid_angle : bool
        If True, plot P(θ)·sin(θ) instead of P(θ).
        The solid-angle weighted form is flat for a freely-jointed chain.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    if solid_angle:
        y     = data["p_sin"]
        ylabel = r"$P(\theta)\cdot\sin(\theta)$"
        label  = r"$P(\theta)\cdot\sin(\theta)$"
        mean   = np.nanmean(y[np.isfinite(y)])
        ax.axhline(mean, color="gray", lw=1.2, ls="--",
                   label=f"mean (FJC: flat) = {mean:.3f}")
    else:
        y      = data["p"]
        ylabel = r"$P(\theta)$"
        label  = r"$P(\theta)$"

    ax.plot(data["theta"], y, label=label)
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)
    style.label_axes(ax,
                     xlabel=r"$\theta$ [degrees]",
                     ylabel=ylabel,
                     title="Bond Angle Distribution")
    ax.legend()
    return ax


def plot_bondangle_both(
    data: Dict[str, np.ndarray],
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Convenience wrapper: P(θ) and P(θ)·sin(θ) side by side.

    Returns
    -------
    fig, (ax_raw, ax_sin)
    """
    fig, (ax1, ax2) = style.figure(ncols=2, width=12)
    plot_bondangle(data, ax=ax1, solid_angle=False)
    plot_bondangle(data, ax=ax2, solid_angle=True)
    return fig, (ax1, ax2)

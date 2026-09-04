"""
mdplot.observables.nematic
==========================
Plotting functions for the nematic order parameter S(x) and director field.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mdplot import style


def plot_nematic(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    show_director: bool = False,
    director_stride: int = 5,
) -> Axes:
    """
    Plot S(x) and optionally overlay the director field as arrows.

    Parameters
    ----------
    data : dict
        Output of ``io.load_nematic``.
        Keys: ``x``, ``S``, ``nx``, ``ny``, ``nz``.
    ax : Axes, optional
    show_director : bool
        Overlay nx arrows on the S(x) curve (scaled by S).
    director_stride : int
        Plot every N-th director arrow to avoid clutter.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["x"], data["S"], label=r"$S(x)$")
    ax.axhline(0,   color="gray", lw=0.8, ls="--")
    ax.axhline(1,   color="gray", lw=0.8, ls=":")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)

    if show_director:
        xs = data["x"][::director_stride]
        Ss = data["S"][::director_stride]
        nx = data["nx"][::director_stride]
        ax.quiver(xs, Ss, nx, np.zeros_like(nx),
                  scale=10, width=0.003, color=style.PALETTE["orange"],
                  label=r"$\hat{n}_x$")

    ax.set_ylim(-0.1, 1.05)
    style.label_axes(ax,
                     xlabel=r"$x$ [$\sigma$]",
                     ylabel=r"$S(x)$",
                     title="Nematic Order Parameter")
    ax.legend()
    return ax


def plot_nematic_full(
    data: Dict[str, np.ndarray],
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Side-by-side: S(x) and all director components nx, ny, nz.

    Returns
    -------
    fig, (ax_S, ax_director)
    """
    fig, (ax1, ax2) = style.figure(ncols=2, width=12)

    plot_nematic(data, ax=ax1)

    ax2.plot(data["x"], data["nx"], label=r"$\hat{n}_x$")
    ax2.plot(data["x"], data["ny"], label=r"$\hat{n}_y$")
    ax2.plot(data["x"], data["nz"], label=r"$\hat{n}_z$")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    style.label_axes(ax2,
                     xlabel=r"$x$ [$\sigma$]",
                     ylabel=r"Director component",
                     title="Nematic Director Field")
    ax2.legend()
    return fig, (ax1, ax2)

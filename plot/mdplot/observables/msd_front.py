"""
mdplot.observables.msd_front
=============================
Plotting functions for front-resolved monomer MSD (swollen vs. dry region).
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_msd_front(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    loglog: bool = False,
    show_slopes: bool = False,
) -> Axes:
    """
    Plot g1_swollen(t) and g1_dry(t) with error bars.

    Parameters
    ----------
    data : dict
        Output of ``io.load_msd_front``.
        Keys: ``t``, ``g1_swollen``, ``err_swollen``, ``g1_dry``, ``err_dry``.
    ax : Axes, optional
        Target axes. Created if not provided.
    loglog : bool
        Use log-log scaling.
    show_slopes : bool
        Overlay t¹ and t^0.5 reference lines, anchored to the swollen curve.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    t = data["t"]
    ax.errorbar(t, data["g1_swollen"], yerr=data["err_swollen"],
                label="swollen (behind front)", capsize=2,
                color=style.PALETTE["blue"])
    ax.errorbar(t, data["g1_dry"], yerr=data["err_dry"],
                label="dry (ahead of front)", capsize=2,
                color=style.PALETTE["red"])

    if show_slopes and loglog:
        t_pos  = t[t > 0]
        g1_pos = data["g1_swollen"][t > 0]
        mid = np.interp(t_pos[len(t_pos) // 2], t_pos, g1_pos)
        style.reference_line(ax, 1.0, t_pos, mid, ls="--", label=r"$\sim t^1$")
        style.reference_line(ax, 0.5, t_pos, mid, ls=":",  label=r"$\sim t^{0.5}$")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    style.label_axes(ax,
                     xlabel=r"$t$ [simulation units]",
                     ylabel=r"$g_1(t)$ [$\sigma^2$]",
                     title="Front-Resolved Monomer MSD")
    ax.legend()
    return ax

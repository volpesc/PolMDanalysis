"""
mdplot.observables.msd
======================
Plotting functions for Mean Squared Displacement data.
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_msd(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    loglog: bool = False,
    show_slopes: bool = False,
    labels: Optional[Dict[str, str]] = None,
) -> Axes:
    """
    Plot g1(t), g2(t), g3(t) on a single axes.

    Parameters
    ----------
    data : dict
        Output of ``io.load_msd``.  Keys: ``t``, ``g1``, ``g2``, ``g3``.
    ax : Axes, optional
        Target axes.  Created if not provided.
    loglog : bool
        Use log-log scaling.
    show_slopes : bool
        Overlay t¹ and t^0.5 reference lines.
    labels : dict, optional
        Override default legend labels, e.g.
        ``{"g1": "my label", "g2": ..., "g3": ...}``.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    _labels = {
        "g1": r"$g_1(t)$ — monomer (lab frame)",
        "g2": r"$g_2(t)$ — monomer (chain CoM)",
        "g3": r"$g_3(t)$ — chain CoM",
    }
    if labels:
        _labels.update(labels)

    t = data["t"]
    ax.plot(t, data["g1"], label=_labels["g1"])
    ax.plot(t, data["g2"], label=_labels["g2"])
    ax.plot(t, data["g3"], label=_labels["g3"])

    if show_slopes and loglog:
        t_pos = t[t > 0]
        g1_pos = data["g1"][t > 0]
        mid = np.interp(t_pos[len(t_pos) // 2], t_pos, g1_pos)
        style.reference_line(ax, 1.0,   t_pos, mid, ls="--", label=r"$\sim t^1$")
        style.reference_line(ax, 0.5,   t_pos, mid, ls=":",  label=r"$\sim t^{0.5}$")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    style.label_axes(ax,
                     xlabel=r"$t$ [simulation units]",
                     ylabel=r"MSD [$\sigma^2$]",
                     title="Mean Squared Displacement")
    ax.legend()
    return ax

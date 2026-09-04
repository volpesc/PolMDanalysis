"""
mdplot.observables.gyr
======================
Plotting functions for the spatial gyration radius profile <Rg²(x)>.
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_gyr(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    show_components: bool = False,
    label: str = r"$R_g^2$ total",
) -> Axes:
    """
    Plot <Rg²(x)> as a function of position x.

    Parameters
    ----------
    data : dict
        Output of ``io.load_gyr``.
    ax : Axes, optional
        Target axes.  Created if not provided.
    show_components : bool
        Also plot the x, y, z components.
    label : str
        Legend label for the total curve.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["x"], data["rgtot"], label=label)
    if show_components:
        ax.plot(data["x"], data["rgx"], ls="--", lw=1.5, label=r"$R_{g,x}^2$")
        ax.plot(data["x"], data["rgy"], ls="--", lw=1.5, label=r"$R_{g,y}^2$")
        ax.plot(data["x"], data["rgz"], ls="--", lw=1.5, label=r"$R_{g,z}^2$")

    style.label_axes(ax,
                     xlabel=r"$x$ [$\sigma$]",
                     ylabel=r"$\langle R_g^2(x) \rangle$ [$\sigma^2$]",
                     title="Gyration Radius Profile")
    ax.legend()
    return ax

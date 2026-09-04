"""
mdplot.observables.msid
=======================
Plotting functions for the Mean Squared Internal Distance C(s).
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_msid(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    show_fjc: bool = True,
    label: str = r"$C(s)$",
) -> Axes:
    """
    Plot C(s) = <R²(s)> / (s·lb²).

    Parameters
    ----------
    data : dict
        Output of ``io.load_msid``.
    ax : Axes, optional
        Target axes.  Created if not provided.
    show_fjc : bool
        Overlay the freely-jointed-chain limit C(s) = 1.
    label : str
        Legend label.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["s"], data["cs"], label=label)
    if show_fjc:
        ax.axhline(1.0, color="gray", lw=1.2, ls="--", label="FJC limit C(s)=1")

    style.label_axes(ax,
                     xlabel=r"$s$ [monomers]",
                     ylabel=r"$C(s) = \langle R^2(s) \rangle / (s \cdot l_b^2)$",
                     title="Mean Squared Internal Distance")
    ax.legend()
    return ax

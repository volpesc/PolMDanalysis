"""
mdplot.observables.rdf
======================
Plotting functions for the radial distribution function g(r).
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from matplotlib.axes import Axes

from mdplot import style


def plot_rdf(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    fill_first_peak: bool = False,
    label: str = r"$g(r)$",
) -> Axes:
    """
    Plot g(r) with an ideal-gas reference line at g=1.

    Parameters
    ----------
    data : dict
        Output of ``io.load_rdf``.  Keys: ``r``, ``gr``.
    ax : Axes, optional
        Target axes.  Created if not provided.
    fill_first_peak : bool
        Shade the area above g=1 under the first peak.
    label : str
        Legend label for the g(r) curve.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    r, gr = data["r"], data["gr"]
    ax.plot(r, gr, label=label)
    ax.axhline(1.0, color="gray", lw=1.0, ls="--", label=r"ideal gas $g(r)=1$")

    if fill_first_peak:
        # Find first local minimum after the peak
        dg = np.diff(gr)
        sign_changes = np.where((dg[:-1] < 0) & (dg[1:] >= 0))[0]
        end = sign_changes[0] + 2 if len(sign_changes) else len(r)
        ax.fill_between(r[:end], 1.0, gr[:end],
                        where=gr[:end] > 1.0,
                        alpha=0.20, label="first peak")

    ax.set_ylim(bottom=0)
    style.label_axes(ax,
                     xlabel=r"$r$ [$\sigma$]",
                     ylabel=r"$g(r)$",
                     title="Radial Distribution Function")
    ax.legend()
    return ax

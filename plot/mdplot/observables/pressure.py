"""
mdplot.observables.pressure
===========================
Plotting functions for the Irving-Kirkwood layer pressure tensor P(x).
"""

from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
from matplotlib.axes import Axes

from mdplot import style

VOIGT_COMPONENTS = ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]


def surface_tension(data: Dict[str, np.ndarray]) -> float:
    """
    Estimate surface tension γ = 0.5 ∫ [PN(x) - PT(x)] dx.

    PN = Pxx (normal to interface)
    PT = (Pyy + Pzz) / 2 (tangential)
    """
    pn = data["Pxx"]
    pt = 0.5 * (data["Pyy"] + data["Pzz"])
    return float(0.5 * np.trapz(pn - pt, data["x"]))


def plot_pressure(
    datasets: Dict[str, np.ndarray] | List[Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    *,
    components: Optional[List[str]] = None,
    show_gamma: bool = False,
) -> Axes:
    """
    Plot pressure tensor components as a function of x.

    Parameters
    ----------
    datasets : dict or list of dicts
        One or more outputs of ``io.load_pressure``.
    labels : list of str, optional
        Source labels (e.g. ["PK_p", "PK_s"]).
    ax : Axes, optional
        Target axes.  Created if not provided.
    components : list of str, optional
        Voigt components to plot.  Defaults to ["Pxx", "Pyy", "Pzz"].
    show_gamma : bool
        Print the surface tension estimate to stdout for each dataset.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure(width=8)

    if isinstance(datasets, dict):
        datasets = [datasets]
    if labels is None:
        labels = [f"dataset {i}" for i in range(len(datasets))]
    if components is None:
        components = ["Pxx", "Pyy", "Pzz"]

    for data, src_label in zip(datasets, labels):
        for comp in components:
            ax.plot(data["x"], data[comp], label=f"{src_label} — {comp}")
        if show_gamma:
            gamma = surface_tension(data)
            print(f"  γ ({src_label}) = {gamma:.4f} [simulation units]")

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    style.label_axes(ax,
                     xlabel=r"$x$ [$\sigma$]",
                     ylabel=r"$P_{\alpha\beta}(x)$",
                     title="Layer Pressure Tensor")
    ax.legend()
    return ax

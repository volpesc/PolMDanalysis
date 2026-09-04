"""
mdplot.observables.endtoend
===========================
Plotting functions for end-to-end distance data.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mdplot import style


def _gaussian_chain_pdf(re2: np.ndarray, re2_mean: float) -> np.ndarray:
    """P(Re²) for an ideal Gaussian chain.

    The textbook result P(R) = 4*pi*R^2 * (3/(2*pi*<Re^2>))^1.5 * exp(-3*R^2/(2*<Re^2>))
    is a density with respect to R. The histogram this overlays is a density
    with respect to Re^2 (matching the C++ tool's normalisation), so this
    applies the R -> Re^2 Jacobian (dR/d(Re^2) = 1/(2R)), giving
    P(Re^2) = 2*pi*sqrt(Re^2) * (3/(2*pi*<Re^2>))^1.5 * exp(-3*Re^2/(2*<Re^2>)).
    """
    re2 = np.where(re2 > 0, re2, 0)
    prefactor = (3.0 / (2.0 * np.pi * re2_mean)) ** 1.5
    return prefactor * 2.0 * np.pi * np.sqrt(re2) * np.exp(-3.0 * re2 / (2.0 * re2_mean))


def plot_endtoend_time(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot <Re²> as a function of simulation frame.

    Parameters
    ----------
    data : dict
        Output of ``io.load_endtoend_time``.
    ax : Axes, optional
        Target axes.  Created if not provided.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    re2_mean = float(np.mean(data["re2"]))
    ax.plot(data["frame"], data["re2"], label=r"$\langle R_e^2 \rangle$")
    ax.axhline(re2_mean, color="gray", ls="--", lw=1.2,
               label=rf"mean = {re2_mean:.2f} $\sigma^2$")

    style.label_axes(ax,
                     xlabel="Frame",
                     ylabel=r"$\langle R_e^2 \rangle$ [$\sigma^2$]",
                     title=r"End-to-End Distance vs Time")
    ax.legend()
    return ax


def plot_endtoend_hist(
    hist_data: Dict[str, np.ndarray],
    time_data: Optional[Dict[str, np.ndarray]] = None,
    ax: Optional[Axes] = None,
    *,
    show_theory: bool = False,
) -> Axes:
    """
    Plot the end-to-end distance distribution P(Re²).

    Parameters
    ----------
    hist_data : dict
        Output of ``io.load_endtoend_hist``.
    time_data : dict, optional
        Output of ``io.load_endtoend_time``.  Required when
        ``show_theory=True`` to compute the mean <Re²>.
    ax : Axes, optional
        Target axes.  Created if not provided.
    show_theory : bool
        Overlay the ideal Gaussian chain prediction.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(hist_data["re2"], hist_data["prob"], label=r"$P(R_e^2)$")

    if show_theory and time_data is not None:
        re2_mean = float(np.mean(time_data["re2"]))
        theory   = _gaussian_chain_pdf(hist_data["re2"], re2_mean)
        ax.plot(hist_data["re2"], theory, ls="--", color="gray",
                lw=1.5, label="Gaussian chain")

    ax.set_ylim(bottom=0)
    style.label_axes(ax,
                     xlabel=r"$R_e^2$ [$\sigma^2$]",
                     ylabel=r"$P(R_e^2)$",
                     title="End-to-End Distribution")
    ax.legend()
    return ax


def plot_endtoend(
    time_data: Dict[str, np.ndarray],
    hist_data: Dict[str, np.ndarray],
    *,
    show_theory: bool = False,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Convenience wrapper: side-by-side time series and histogram.

    Returns
    -------
    fig, (ax_time, ax_hist)
    """
    fig, (ax1, ax2) = style.figure(ncols=2, width=12)
    plot_endtoend_time(time_data, ax=ax1)
    plot_endtoend_hist(hist_data, time_data=time_data, ax=ax2,
                       show_theory=show_theory)
    return fig, (ax1, ax2)

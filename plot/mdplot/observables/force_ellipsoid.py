"""
mdplot.observables.force_ellipsoid
===================================
Plotting functions for per-monomer force ellipsoid analysis.

The force ellipsoid characterises the anisotropy of the net force acting
on each monomer, averaged over frames.  Three principal quantities are
plotted as a function of monomer index (which maps to position along the
backbone or along x when chains are ordered):

  anisotropy  = sqrt(lambda1) / sqrt(lambda3)   >= 1
    1 = perfectly isotropic force, higher = more anisotropic

  prolateness = (lambda1 - lambda2) / (lambda1 - lambda3)  in [0, 1]
    0 = oblate (disc-like), 1 = prolate (rod-like)

  force magnitude  |<f>|  averaged over frames
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mdplot import style


def load_force_ellipsoid(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Load force ellipsoid CSV output.

    Returns
    -------
    dict with keys: ``pid``, ``fx``, ``fy``, ``fz``, ``fmag``,
                    ``l1``, ``l2``, ``l3``, ``anisotropy``, ``prolateness``.
    """
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([
                int(row["particle_id"]),
                float(row["fx_avg"]), float(row["fy_avg"]), float(row["fz_avg"]),
                float(row["fmag_avg"]),
                float(row["lambda1"]), float(row["lambda2"]), float(row["lambda3"]),
                float(row["anisotropy"]), float(row["prolateness"]),
            ])
    d = np.array(rows)
    return {
        "pid":         d[:, 0],
        "fx":          d[:, 1], "fy": d[:, 2], "fz": d[:, 3],
        "fmag":        d[:, 4],
        "l1":          d[:, 5], "l2": d[:, 6], "l3": d[:, 7],
        "anisotropy":  d[:, 8],
        "prolateness": d[:, 9],
    }


def plot_force_magnitude(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    label: str = r"$|\langle \mathbf{f} \rangle|$",
) -> Axes:
    """
    Plot average force magnitude vs monomer index.

    Parameters
    ----------
    data : dict
        Output of ``load_force_ellipsoid``.
    ax : Axes, optional
    label : str

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["pid"], data["fmag"], label=label)
    style.label_axes(ax,
                     xlabel="Monomer index",
                     ylabel=r"$|\langle \mathbf{f} \rangle|$ [$\varepsilon/\sigma$]",
                     title="Average Force Magnitude")
    ax.legend()
    return ax


def plot_force_anisotropy(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    *,
    show_prolateness: bool = False,
) -> Axes:
    """
    Plot force ellipsoid anisotropy (and optionally prolateness) vs monomer index.

    Parameters
    ----------
    data : dict
        Output of ``load_force_ellipsoid``.
    ax : Axes, optional
    show_prolateness : bool
        Overlay prolateness on a twin y-axis.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["pid"], data["anisotropy"],
            color=style.PALETTE["blue"],
            label=r"Anisotropy $\sqrt{\lambda_1}/\sqrt{\lambda_3}$")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")

    if show_prolateness:
        ax2 = ax.twinx()
        ax2.plot(data["pid"], data["prolateness"],
                 color=style.PALETTE["orange"], ls="--",
                 label=r"Prolateness $(\lambda_1-\lambda_2)/(\lambda_1-\lambda_3)$")
        ax2.set_ylabel("Prolateness", color=style.PALETTE["orange"])
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", colors=style.PALETTE["orange"])
        # Combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines+lines2, labels+labels2, fontsize=10)
    else:
        ax.legend()

    style.label_axes(ax,
                     xlabel="Monomer index",
                     ylabel=r"Anisotropy",
                     title="Force Ellipsoid Anisotropy")
    return ax


def plot_force_eigenvalues(
    data: Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot all three eigenvalues λ1, λ2, λ3 of the force covariance tensor.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = style.figure()

    ax.plot(data["pid"], data["l1"], label=r"$\lambda_1$")
    ax.plot(data["pid"], data["l2"], label=r"$\lambda_2$")
    ax.plot(data["pid"], data["l3"], label=r"$\lambda_3$")

    style.label_axes(ax,
                     xlabel="Monomer index",
                     ylabel=r"Eigenvalue [$(\varepsilon/\sigma)^2$]",
                     title="Force Covariance Eigenvalues")
    ax.legend()
    return ax


def plot_force_ellipsoid(
    data: Dict[str, np.ndarray],
) -> Tuple[Figure, Tuple[Axes, Axes, Axes]]:
    """
    Convenience wrapper: three-panel figure with magnitude, anisotropy, eigenvalues.

    Returns
    -------
    fig, (ax_mag, ax_anis, ax_eig)
    """
    fig, axes = style.figure(ncols=3, width=15)
    plot_force_magnitude  (data, ax=axes[0])
    plot_force_anisotropy (data, ax=axes[1], show_prolateness=True)
    plot_force_eigenvalues(data, ax=axes[2])
    return fig, tuple(axes)

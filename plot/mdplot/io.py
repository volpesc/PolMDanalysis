"""
mdplot.io
=========
Data-loading helpers shared across all observable modules.

Each loader returns a plain ``dict`` of labelled ``np.ndarray`` columns so
that observable modules never need to know about file layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np


def _load(path: str | Path, expected_cols: int | None = None) -> np.ndarray:
    """Load a whitespace-separated data file, skipping comment lines."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if expected_cols and data.shape[1] < expected_cols:
        raise ValueError(
            f"{path}: expected ≥{expected_cols} columns, got {data.shape[1]}"
        )
    return data


def load_msd(path: str | Path) -> Dict[str, np.ndarray]:
    """Load MSD output.  Columns: t  g1  g2  g3"""
    d = _load(path, 4)
    return {"t": d[:, 0], "g1": d[:, 1], "g2": d[:, 2], "g3": d[:, 3]}

def load_msd_front(path: str | Path) -> Dict[str, np.ndarray]:
    """Load front-resolved MSD output. Columns: t  g1_swollen  err_swollen  g1_dry  err_dry"""
    d = _load(path, 5)
    return {"t": d[:, 0], "g1_swollen": d[:, 1], "err_swollen": d[:, 2],
            "g1_dry": d[:, 3], "err_dry": d[:, 4]}

def load_gyr(path: str | Path) -> Dict[str, np.ndarray]:
    """Load gyration radius profile.  Columns: x  rgx  rgy  rgz  rgtot"""
    d = _load(path, 5)
    return {"x": d[:, 0], "rgx": d[:, 1], "rgy": d[:, 2],
            "rgz": d[:, 3], "rgtot": d[:, 4]}


def load_msid(path: str | Path) -> Dict[str, np.ndarray]:
    """Load MSID output.  Columns: s  C(s)"""
    d = _load(path, 2)
    return {"s": d[:, 0], "cs": d[:, 1]}


def load_rdf(path: str | Path) -> Dict[str, np.ndarray]:
    """Load RDF output.  Columns: r  g(r)"""
    d = _load(path, 2)
    return {"r": d[:, 0], "gr": d[:, 1]}


def load_density(path: str | Path) -> Dict[str, np.ndarray]:
    """Load density profile.  Columns: x  rho_polymer  [rho_solvent]"""
    d = _load(path, 2)
    result = {"x": d[:, 0], "rho_poly": d[:, 1]}
    if d.shape[1] > 2:
        result["rho_solv"] = d[:, 2]
    return result


def load_endtoend_time(path: str | Path) -> Dict[str, np.ndarray]:
    """Load end-to-end time series.  Columns: frame  Re2  |Re|"""
    d = _load(path, 3)
    return {"frame": d[:, 0], "re2": d[:, 1], "re": d[:, 2]}


def load_endtoend_hist(path: str | Path) -> Dict[str, np.ndarray]:
    """Load end-to-end histogram.  Columns: Re2  P(Re2)"""
    d = _load(path, 2)
    return {"re2": d[:, 0], "prob": d[:, 1]}


def load_bondangle(path: str | Path) -> Dict[str, np.ndarray]:
    """Load bond angle distribution.  Columns: theta[deg]  P(theta)  P*sin"""
    d = _load(path, 3)
    return {"theta": d[:, 0], "p": d[:, 1], "p_sin": d[:, 2]}


def load_sq(path: str | Path) -> Dict[str, np.ndarray]:
    """Load structure factor.  Columns: q  S(q)"""
    d = _load(path, 2)
    return {"q": d[:, 0], "sq": d[:, 1]}


def load_pressure(path: str | Path) -> Dict[str, np.ndarray]:
    """Load pressure tensor.  Columns: x  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"""
    d = _load(path, 7)
    keys = ["x", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    return {k: d[:, i] for i, k in enumerate(keys)}


def load_ppa(path: str | Path) -> Dict[str, float]:
    """Parse PPA key=value output. Returns dict with bpp, app, Ne."""
    from mdplot.observables.ppa import load_ppa as _load
    return _load(path)


def load_entanglement(path: str | Path) -> Dict[str, np.ndarray]:
    """Load entanglement output. Columns: x  kink_count  chains_in_bin  ent_per_chain"""
    d = _load(path, 4)
    return {"x": d[:,0], "kink_count": d[:,1],
            "chains_in_bin": d[:,2], "ent_per_chain": d[:,3]}


def load_spec_density(path: str | Path) -> Dict[str, np.ndarray]:
    """Load species density profile. Columns: r  rho_poly  rho_solv"""
    d = _load(path, 2)
    result = {"r": d[:,0], "rho_poly": d[:,1]}
    if d.shape[1] > 2:
        result["rho_solv"] = d[:,2]
    return result


def load_volume(path: str | Path) -> Dict[str, float]:
    """Load volume output. Columns: Rg  V_eq"""
    d = _load(path, 2)
    return {"Rg": float(d[0,0]), "Veq": float(d[0,1])}


def load_backbone(path: str | Path) -> Dict[str, np.ndarray]:
    """Load backbone force profile. Columns: bond_index  F_proj"""
    d = _load(path, 2)
    return {"bond_index": d[:,0], "F_proj": d[:,1]}


def load_energy(path: str | Path) -> Dict[str, np.ndarray]:
    """Load energy vs lambda file. Columns: lambda  energy"""
    d = _load(path, 2)
    return {"lam": d[:,0], "energy": d[:,1]}


def load_nematic(path: str | Path) -> Dict[str, np.ndarray]:
    """Load nematic output. Columns: x  S  nx  ny  nz"""
    d = _load(path, 5)
    return {"x": d[:,0], "S": d[:,1], "nx": d[:,2], "ny": d[:,3], "nz": d[:,4]}


def load_gds(path: str | Path) -> Dict[str, np.ndarray]:
    """Load GDS CSV. Columns: time, solvent_front, polymer_front, solvent_uptake, polymer_uptake"""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["time"]), float(row["solvent_front"]),
                         float(row["polymer_front"]), float(row["solvent_uptake"]),
                         float(row["polymer_uptake"])])
    d = np.array(rows)
    return {"time": d[:,0], "solvent_front": d[:,1], "polymer_front": d[:,2],
            "solvent_uptake": d[:,3], "polymer_uptake": d[:,4]}


def load_brush_length(path: str | Path) -> Dict[str, np.ndarray]:
    """Load brush length distribution. Columns: L  P(L)"""
    d = _load(path, 2)
    return {"L": d[:,0], "prob": d[:,1]}


def load_force_ellipsoid(path: str | Path) -> Dict[str, np.ndarray]:
    """Load force ellipsoid CSV. See observables/force_ellipsoid.py for keys."""
    from mdplot.observables.force_ellipsoid import load_force_ellipsoid as _load
    return _load(path)

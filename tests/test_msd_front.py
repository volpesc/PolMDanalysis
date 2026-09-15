"""Regression test for front-resolved MSD (msdfront tool).

Builds a synthetic trajectory with two rigid-body populations of chains
moving at different constant velocities, offset on either side of a fixed
solvent cluster that pins the GDS front at a known x. Checks that msdfront
classifies each population correctly at every time origin and reports g1(t)
matching the exact ballistic prediction (v*t)^2 for each population, with
small error bars (constant velocity means every origin gives nearly the
same per-origin mean).
"""
from __future__ import annotations

import pytest

from conftest import run_tool
from _xyz import write_frame


def test_msd_front_separates_populations(tmp_path, analysis_bin):
    Lx = Ly = Lz = 100.0
    Nm, Nc = 4, 4          # chains 0,1 = swollen; chains 2,3 = dry
    v_swollen, v_dry = 0.5, 0.2   # sigma / frame
    x_swollen0, x_dry0 = 10.0, 40.0
    x_front = 20.0
    n_solvent = 20
    nframes = 12
    prefix = "requil_"

    for t in range(nframes):
        positions = []
        # Chains 0-1: swollen population, behind the front.
        for c in range(2):
            for i in range(Nm):
                positions.append((x_swollen0 + v_swollen * t, 5.0 * c + i, 5.0))
        # Chains 2-3: dry population, ahead of the front.
        for c in range(2):
            for i in range(Nm):
                positions.append((x_dry0 + v_dry * t, 5.0 * c + i, 5.0))
        # Static solvent cluster pinning the GDS front at ~x_front.
        for s in range(n_solvent):
            positions.append((x_front, 50.0 + s, 50.0))

        write_frame(tmp_path / f"{prefix}{t}.xyz", positions, Lx, Ly, Lz)

    out = tmp_path / "msd_front.dat"
    result = run_tool(
        analysis_bin, "--tool", "msdfront",
        "--prefix", str(tmp_path / prefix),
        "--Nm", str(Nm), "--Nc", str(Nc), "--Ns", str(n_solvent),
        "--start", "0", "--stop", str(nframes - 1), "--step", "1",
        "--dt", "1", "--xmin", "0", "--xmax", "100", "--binw", "2.0",
        "--frontbuf", "2.0", "--tmax", "300", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    lines = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert len(lines) > 0

    for line in lines:
        t_str, g1s, errs, g1d, errd = line.split()
        t = float(t_str)
        g1s, errs, g1d, errd = float(g1s), float(errs), float(g1d), float(errd)

        expected_s = (v_swollen * t) ** 2
        expected_d = (v_dry * t) ** 2

        assert g1s == pytest.approx(expected_s, rel=1e-3), (
            f"g1_swollen({t}) = {g1s}, expected {expected_s} -- "
            "swollen population misclassified or displaced wrongly"
        )
        assert g1d == pytest.approx(expected_d, rel=1e-3), (
            f"g1_dry({t}) = {g1d}, expected {expected_d} -- "
            "dry population misclassified or displaced wrongly"
        )
        # Constant velocity: every time origin gives nearly the same
        # per-origin mean, so the standard error across origins should
        # stay small (not exactly zero, due to text-precision rounding
        # of positions written to the trajectory).
        assert errs < 1e-3, f"err_swollen({t}) = {errs}, expected small"
        assert errd < 1e-3, f"err_dry({t}) = {errd}, expected small"

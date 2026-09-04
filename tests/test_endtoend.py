"""Regression tests for the end-to-end distance tool and its theory overlay.

Two independent bugs lived here:
  1. mdplot's _gaussian_chain_pdf() plotted P(R) (density w.r.t. R) against
     an Re^2 axis without the Jacobian correction, so the "ideal chain"
     overlay didn't match its own axis.
  2. (Regression guard only, no bug found here) the C++ tool reconstructs
     the end-to-end vector bond-by-bond with minimum-image correction, so a
     chain whose backbone straddles a periodic boundary in a single frame
     should still give the correct end-to-end distance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import run_tool
from _xyz import write_frame

PLOT_DIR = Path(__file__).resolve().parent.parent / "plot"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))


def test_gaussian_chain_pdf_is_normalised():
    numpy = pytest.importorskip("numpy")
    from mdplot.observables.endtoend import _gaussian_chain_pdf

    re2_mean = 25.0
    x = numpy.linspace(0.0, 400.0, 200_000)
    p = _gaussian_chain_pdf(x, re2_mean)

    trapezoid = getattr(numpy, "trapezoid", None) or numpy.trapz
    integral = trapezoid(p, x)
    mean_recovered = trapezoid(p * x, x)

    assert integral == pytest.approx(1.0, abs=1e-4), (
        "P(Re^2) should integrate to 1 over Re^2 -- if this fails the "
        "overlay is back to being a density w.r.t. R plotted on the wrong axis"
    )
    assert mean_recovered == pytest.approx(re2_mean, rel=1e-3)


def test_endtoend_reconstruction_across_periodic_boundary(tmp_path, analysis_bin):
    """A 3-monomer chain whose backbone crosses x=0/L should still give the
    true (short) end-to-end distance, not one inflated by the box length."""
    L = 20.0
    bond = 1.0
    # Chain deliberately placed straddling the x boundary: monomer 0 near
    # x=L, monomer 1 wrapped to just past x=0, monomer 2 further along.
    positions = [
        (L - 0.5, 10.0, 10.0),
        (0.5 % L, 10.0, 10.0),          # bond length 1.0 across the wrap
        (1.5, 10.0, 10.0),
    ]
    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out_stem = tmp_path / "endtoend"
    result = run_tool(
        analysis_bin, "--tool", "endtoend",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", "3", "--Nc", "1",
        "--start", "0", "--stop", "0", "--step", "1",
        "--out", str(out_stem),
    )
    assert result.returncode == 0, result.stderr

    time_lines = [ln for ln in (tmp_path / "endtoend_time.dat").read_text().splitlines()
                  if not ln.startswith("#")]
    assert len(time_lines) == 1
    _frame, re2, re_rms = time_lines[0].split()

    expected_re = 2 * bond  # two collinear bonds of length `bond` each
    assert float(re_rms) == pytest.approx(expected_re, rel=1e-4), (
        f"sqrt(<Re^2>) = {re_rms}, expected {expected_re} -- a much larger "
        "value means the boundary-crossing bond wasn't minimum-imaged"
    )

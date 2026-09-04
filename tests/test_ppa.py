"""Regression test for the PPA (Primitive Path Analysis) tool.

For a single, straight, fully-extended chain of N monomers spaced lb
apart: every bond has length lb (so bpp = lb exactly), the chain is fully
stretched so Re = (N-1)*lb, and the contour length L = (N-1)*lb, giving
app = Re^2/L = (N-1)*lb and Ne = app/bpp = (N-1). Wrapping the chain
across a small periodic box (as in test_msid's multi-wrap case) exercises
minImageVec()'s bond reconstruction alongside this arithmetic.
"""
from __future__ import annotations

import pytest

from conftest import run_tool
from _xyz import write_frame


def test_ppa_straight_chain_wrapped(tmp_path, analysis_bin):
    lb = 1.0
    Nm = 9
    L = 2.5  # small box: this straight chain wraps several times
    raw_x = [(i * lb) % L for i in range(Nm)]
    positions = [(x, 5.0, 5.0) for x in raw_x]
    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out = tmp_path / "ppa.dat"
    result = run_tool(
        analysis_bin, "--tool", "ppa",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", str(Nm), "--Nc", "1",
        "--frame", "0", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    values = {}
    for line in out.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=")
            values[key.strip()] = float(val)

    expected_bpp = lb
    expected_app = (Nm - 1) * lb
    expected_Ne = Nm - 1

    assert values["bpp"] == pytest.approx(expected_bpp, rel=1e-4)
    assert values["app"] == pytest.approx(expected_app, rel=1e-4)
    assert values["Ne"] == pytest.approx(expected_Ne, rel=1e-4)

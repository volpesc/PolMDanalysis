"""Physics regression test for the bond-angle distribution tool.

bond_angle.hpp defines theta via cos(theta) = b1.b2 / (|b1||b2|) with both
bond vectors pointing "forward" along the chain (b1 = i->i+1, b2 =
i+1->i+2) -- the same convention entanglement.hpp uses for kink detection.
Under that convention a straight, fully-extended chain gives theta = 0 deg
(b1 parallel to b2), not 180 deg. This builds such a chain, deliberately
wrapped across a small periodic box (like test_msid's multi-wrap case), so
the check exercises minImageVec()'s bond reconstruction together with the
angle geometry.
"""
from __future__ import annotations

import math

import pytest

from conftest import run_tool
from _xyz import write_frame


def test_bond_angle_straight_chain_gives_zero_degrees(tmp_path, analysis_bin):
    lb = 1.0
    L = 2.5  # wraps more than once over the chain length
    Nm = 8
    nbins = 180
    raw_x = [(i * lb) % L for i in range(Nm)]
    positions = [(x, 5.0, 5.0) for x in raw_x]
    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out = tmp_path / "bond_angle.dat"
    result = run_tool(
        analysis_bin, "--tool", "bondangle",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", str(Nm), "--Nc", "1",
        "--start", "0", "--stop", "0", "--step", "1",
        "--nbins", str(nbins), "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    rows = [ln.split() for ln in out.read_text().splitlines() if not ln.startswith("#")]
    dtheta_rad = math.pi / nbins

    mass_near_0 = sum(float(p) for theta_deg, p, _psin in rows
                      if float(theta_deg) < 10.0) * dtheta_rad
    assert mass_near_0 == pytest.approx(1.0, abs=1e-3), (
        f"only {mass_near_0*100:.2f}% of the P(theta) mass is within 10 deg "
        "of 0 deg for a perfectly straight, fully-extended chain -- either "
        "the bond reconstruction or the angle geometry is off"
    )

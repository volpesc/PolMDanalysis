"""Regression test for UnwrapPolicy::ChainCluster (volume.hpp / spec_density.hpp).

A straight, evenly-spaced chain has a closed-form radius of gyration:
for N points at positions i*lb (i = 0..N-1), Rg^2 = lb^2*(N^2-1)/12 (the
variance of a discrete uniform sequence). Deliberately wrapping that chain
across a small periodic box and checking Rg still matches the closed form
exercises unwrapPolymerCluster()'s handling of periodic wraps.

volume.hpp and spec_density.hpp both go through the same
loadFrame(..., UnwrapPolicy::ChainCluster) -> unwrapPolymerCluster() path,
so this one check covers the shared correctness of both.
"""
from __future__ import annotations

import math

import pytest

from conftest import run_tool
from _xyz import write_frame


def test_volume_rg_survives_periodic_wrap(tmp_path, analysis_bin):
    lb = 1.0
    Nm = 10
    L = 2.5  # small box: this straight chain wraps several times
    raw_x = [(i * lb) % L for i in range(Nm)]
    positions = [(x, 5.0, 5.0) for x in raw_x]
    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out = tmp_path / "volume.dat"
    result = run_tool(
        analysis_bin, "--tool", "volume",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", str(Nm), "--Nc", "1",
        "--frame", "0", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    line = next(ln for ln in out.read_text().splitlines() if not ln.startswith("#"))
    rg_str, _veq_str = line.split()
    rg = float(rg_str)

    expected_rg = lb * math.sqrt((Nm ** 2 - 1) / 12.0)
    assert rg == pytest.approx(expected_rg, rel=1e-4), (
        f"Rg = {rg}, expected {expected_rg} for a straight {Nm}-mer rod -- "
        "check unwrapPolymerCluster()'s handling of periodic wraps"
    )

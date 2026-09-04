"""Regression test for the MSD trajectory-unwrapping bug.

computeMSD() used to compute g1/g2/g3 straight from raw, PBC-wrapped
coordinates. A particle crossing a periodic boundary between two frames
then showed up as a spurious +-L jump in the reported displacement. This
test builds a single chain moving at constant velocity, arranged so it
crosses the box boundary partway through the trajectory, and checks that
g1(t) still matches the exact ballistic prediction (v*t)^2 at every lag --
including the lag that spans the boundary crossing.
"""
from __future__ import annotations

import pytest

from conftest import run_tool
from _xyz import write_frame


def test_msd_survives_periodic_boundary_crossing(tmp_path, analysis_bin):
    L = 10.0
    v = 0.6          # sigma / frame
    bond = 0.9       # sigma
    nframes = 20
    prefix = "requil_"

    for t in range(nframes):
        x0 = (0.5 + v * t) % L
        x1 = (0.5 + bond + v * t) % L
        write_frame(tmp_path / f"{prefix}{t}.xyz",
                    [(x0, 5.0, 5.0), (x1, 5.0, 5.0)], L, L, L)

    # Sanity check the fixture actually crosses the boundary somewhere in
    # [0, nframes): otherwise this test wouldn't be exercising the bug at all.
    xs = [(0.5 + v * t) % L for t in range(nframes)]
    assert any(xs[t + 1] < xs[t] for t in range(nframes - 1)), \
        "fixture must cross the periodic boundary at least once"

    out = tmp_path / "msd.dat"
    result = run_tool(
        analysis_bin, "--tool", "msd",
        "--prefix", str(tmp_path / prefix),
        "--Nm", "2", "--Nc", "1",
        "--start", "0", "--stop", str(nframes - 1), "--step", "1",
        "--dt", "1", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    lines = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert len(lines) == nframes - 1

    for line in lines:
        t_str, g1_str, _g2, _g3 = line.split()
        t, g1 = float(t_str), float(g1_str)
        expected = (v * t) ** 2
        assert g1 == pytest.approx(expected, rel=1e-4), (
            f"g1({t}) = {g1}, expected {expected} (ballistic motion) -- "
            "a mismatch here means MSD is no longer unwrapping the trajectory"
        )

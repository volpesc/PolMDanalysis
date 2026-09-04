"""Physics regression test for the RDF tool.

For an uncorrelated (Poisson) point gas -- no excluded volume, no packing
structure -- the radial distribution function g(r) should be ~1 at every r
except where shot noise dominates (very small r, where the shell volume,
and so the expected count, is tiny). This checks the mid-to-large-r tail
of a genuinely random configuration averages to 1, which is only true if
the normalisation (ideal-gas density, shell volume) is correct.
"""
from __future__ import annotations

import random

import pytest

from conftest import run_tool
from _xyz import write_frame


def test_rdf_of_uniform_random_gas_is_flat(tmp_path, analysis_bin):
    rng = random.Random(7)
    L = 30.0
    N = 3000
    positions = [(rng.uniform(0, L), rng.uniform(0, L), rng.uniform(0, L))
                 for _ in range(N)]
    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out = tmp_path / "rdf.dat"
    result = run_tool(
        analysis_bin, "--tool", "rdf",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", "1", "--Nc", str(N),
        "--start", "0", "--stop", "0", "--step", "1",
        "--rcut", "5.0", "--nbins", "25", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    rows = [ln.split() for ln in out.read_text().splitlines() if not ln.startswith("#")]
    g = [float(g) for _r, g in rows]

    # Skip the small-r half (few pairs -> high shot noise); average g(r)
    # over the rest should sit close to 1 for an uncorrelated gas.
    tail = g[len(g) // 2:]
    mean_g = sum(tail) / len(tail)
    assert mean_g == pytest.approx(1.0, abs=0.15), (
        f"mean g(r) over the mid-to-large-r tail is {mean_g}, expected ~1 "
        "for an uncorrelated point gas -- check the normalisation"
    )

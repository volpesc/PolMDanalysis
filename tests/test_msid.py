"""Physics regression tests for the MSID tool.

Two independent, analytically-known checks:
  1. A freely-jointed chain (independent, fixed-length, randomly oriented
     bonds) must give C(s) = <R^2(s)>/(s*lb^2) ~ 1 for every s -- this is
     the textbook FJC identity msid.hpp's own docstring states.
  2. A straight, fully-extended chain deliberately wrapped across a small
     box (so consecutive monomers cross a periodic boundary more than
     once) must still give C(s) = s exactly, which only holds if
     unwrapChains() correctly recovers multi-box-length jumps (it uses
     round(), not a single +-L step, specifically to handle this case).
"""
from __future__ import annotations

import math
import random

import pytest

from conftest import run_tool
from _xyz import write_frame


def _read_msid(path):
    return {int(s): float(c) for s, c in
            (ln.split() for ln in path.read_text().splitlines()
             if not ln.startswith("#"))}


def test_msid_freely_jointed_chain_limit(tmp_path, analysis_bin):
    rng = random.Random(20240904)
    Nm, Nc = 20, 400
    lb = 1.0
    L = 1000.0  # large box: keep this test about the FJC identity, not PBC
                # handling (that's covered separately below).

    positions = []
    for _c in range(Nc):
        x = y = z = L / 2
        positions.append((x, y, z))
        for _m in range(Nm - 1):
            theta = math.acos(rng.uniform(-1.0, 1.0))
            phi = rng.uniform(0.0, 2 * math.pi)
            x += lb * math.sin(theta) * math.cos(phi)
            y += lb * math.sin(theta) * math.sin(phi)
            z += lb * math.cos(theta)
            positions.append((x, y, z))

    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out = tmp_path / "msid.dat"
    result = run_tool(
        analysis_bin, "--tool", "msid",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", str(Nm), "--Nc", str(Nc),
        "--start", "0", "--stop", "0", "--step", "1",
        "--lb", str(lb), "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    Cs = _read_msid(out)
    mid = [Cs[s] for s in range(5, 16) if s in Cs]
    assert mid, "expected s=5..15 in msid.dat"
    mean_C = sum(mid) / len(mid)
    assert mean_C == pytest.approx(1.0, abs=0.1), (
        f"mean C(s) over s=5..15 is {mean_C}, expected ~1 for a freely "
        "jointed chain"
    )


def test_msid_handles_multi_box_wrap(tmp_path, analysis_bin):
    lb = 1.0
    L = 2.5  # small enough that this straight 6-monomer chain wraps twice
    Nm = 6
    raw_x = [(i * lb) % L for i in range(Nm)]
    positions = [(x, 5.0, 5.0) for x in raw_x]
    write_frame(tmp_path / "requil_0.xyz", positions, L, L, L)

    out = tmp_path / "msid.dat"
    result = run_tool(
        analysis_bin, "--tool", "msid",
        "--prefix", str(tmp_path / "requil_"),
        "--Nm", str(Nm), "--Nc", "1",
        "--start", "0", "--stop", "0", "--step", "1",
        "--lb", str(lb), "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    Cs = _read_msid(out)
    for s in range(1, Nm):
        # Fully extended, collinear chain: R(s) = s*lb exactly, so
        # C(s) = R(s)^2 / (s*lb^2) = s.
        assert Cs[s] == pytest.approx(float(s), rel=1e-4), (
            f"C({s}) = {Cs[s]}, expected {s} -- a wrong value means the "
            "chain wasn't correctly unwrapped across the periodic boundary"
        )

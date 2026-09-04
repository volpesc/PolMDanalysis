"""Broad smoke test: every tool not covered by a dedicated physics check
(see the other test_*.py files) should run cleanly and produce non-empty
output on a generic synthetic trajectory. This isn't a correctness check
on its own -- it's a cheap net against exceptions, crashes, and
argument-wiring mistakes across the whole CLI surface.
"""
from __future__ import annotations

import random

import pytest

from conftest import run_tool
from _xyz import write_frame


@pytest.fixture(scope="module")
def generic_trajectory(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke_traj")
    rng = random.Random(99)
    L = 20.0
    Nm, Nc, Ns = 10, 12, 150
    nframes = 4
    for t in range(nframes):
        positions = []
        for _c in range(Nc):
            x, y, z = rng.uniform(0, L), rng.uniform(0, L), rng.uniform(0, L)
            for _m in range(Nm):
                x = (x + rng.uniform(-0.3, 0.3)) % L
                y = (y + rng.uniform(-0.3, 0.3)) % L
                z = (z + rng.uniform(-0.3, 0.3)) % L
                positions.append((x, y, z))
        for _s in range(Ns):
            positions.append((rng.uniform(0, L), rng.uniform(0, L), rng.uniform(0, L)))
        write_frame(d / f"requil_{t}.xyz", positions, L, L, L)
    return d, dict(Nm=Nm, Nc=Nc, Ns=Ns, L=L, nframes=nframes)


def _assert_ok_and_nonempty(result, *paths):
    assert result.returncode == 0, result.stderr
    for p in paths:
        assert p.exists() and p.stat().st_size > 0, f"{p} missing or empty"


def test_density(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "density.dat"
    r = run_tool(analysis_bin, "--tool", "density", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--start", "0", "--stop", str(p["nframes"] - 1), "--step", "1",
                 "--nbins", "20", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


def test_nematic(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "nematic.dat"
    r = run_tool(analysis_bin, "--tool", "nematic", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--frame", "1", "--nbins", "10", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


@pytest.mark.parametrize("mode", ["chain_com", "monomer"])
def test_entanglement(generic_trajectory, analysis_bin, tmp_path, mode):
    d, p = generic_trajectory
    out = tmp_path / f"ent_{mode}.dat"
    r = run_tool(analysis_bin, "--tool", "entanglement", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--frame", "1", "--nbins", "10", "--kinkdeg", "150",
                 "--entmode", mode, "--out", str(out))
    _assert_ok_and_nonempty(r, out)


def test_gds(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "gds.csv"
    r = run_tool(analysis_bin, "--tool", "gds", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--start", "0", "--stop", str(p["nframes"] - 1),
                 "--xmin", "0", "--xmax", str(p["L"]), "--binw", "2.0",
                 "--dt", "10", "--xthresh", "10", "--vthresh", "0.001", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


def test_brushlength(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "brush.dat"
    r = run_tool(analysis_bin, "--tool", "brushlength", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--frame", "1", "--percentile", "90", "--cbuffer", "8",
                 "--binw", "1", "--minbin", "0", "--maxbin", "15", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


def test_force_ellipsoid(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out_stem = tmp_path / "interval"
    r = run_tool(analysis_bin, "--tool", "fellipsoid", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--start", "0", "--stop", str(p["nframes"] - 1), "--step", "1",
                 "--threads", "1", "--out", str(out_stem))
    _assert_ok_and_nonempty(r, tmp_path / "interval_ellipsoid.csv")


def test_gyr(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "gyr.dat"
    r = run_tool(analysis_bin, "--tool", "gyr", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]),
                 "--frame", "1", "--binw", "2", "--nbins", "10", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


@pytest.mark.parametrize("mode,sampling", [
    ("sc", "spherical"), ("sc", "cartesian"),
    ("tot", "spherical"), ("tot", "cartesian"),
])
def test_sq(generic_trajectory, analysis_bin, tmp_path, mode, sampling):
    d, p = generic_trajectory
    out = tmp_path / f"sq_{mode}_{sampling}.dat"
    args = ["--tool", "sq", "--prefix", str(d / "requil_"),
            "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]),
            "--mode", mode, "--sampling", sampling, "--nkbound", "10", "--out", str(out)]
    args += (["--frame", "1"] if mode == "sc"
             else ["--start", "0", "--stop", str(p["nframes"] - 1)])
    r = run_tool(analysis_bin, *args)
    _assert_ok_and_nonempty(r, out)


def test_backbone_full(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "backbone_full.dat"
    r = run_tool(analysis_bin, "--tool", "backbone", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]),
                 "--frame", "1", "--bbmode", "full", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


def test_energy(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "energy.dat"
    r = run_tool(analysis_bin, "--tool", "energy", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--frame", "1", "--lambda", "0.5", "--out", str(out))
    _assert_ok_and_nonempty(r, out)


def test_pressure_under_mpi(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out_stem = tmp_path / "pressure"
    r = run_tool(analysis_bin, "--tool", "pressure", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--frame", "1", "--nlayer", "10", "--out", str(out_stem), ranks=2)
    _assert_ok_and_nonempty(r, tmp_path / "pressure_total.txt")


def test_spec_density(generic_trajectory, analysis_bin, tmp_path):
    d, p = generic_trajectory
    out = tmp_path / "spec_density.dat"
    r = run_tool(analysis_bin, "--tool", "specdensity", "--prefix", str(d / "requil_"),
                 "--Nm", str(p["Nm"]), "--Nc", str(p["Nc"]), "--Ns", str(p["Ns"]),
                 "--frame", "1", "--nshells", "10", "--shellw", "1.0", "--out", str(out))
    _assert_ok_and_nonempty(r, out)

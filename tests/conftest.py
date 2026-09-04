"""Shared fixtures for the PolMDanalysis test suite.

Builds the C++ `analysis` binary once per test session (out-of-tree, in
build/) and provides a helper to run it as a subprocess. Every tool goes
through mpirun -np <ranks>, since analysis.cpp calls MPI_Init/MPI_Finalize
unconditionally regardless of the tool -- running it as a bare executable
depends on the MPI implementation's optional (and not always enabled)
"singleton init" support, so we don't rely on it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"


@pytest.fixture(scope="session")
def analysis_bin() -> Path:
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=BUILD_DIR, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["make", f"-j{os.cpu_count() or 2}"],
        cwd=BUILD_DIR, check=True, capture_output=True, text=True,
    )
    binary = BUILD_DIR / "analysis"
    assert binary.exists(), "build did not produce build/analysis"
    return binary


def run_tool(analysis_bin: Path, *args: str, ranks: int = 1,
             timeout: float = 120) -> subprocess.CompletedProcess:
    """Run one analysis invocation under mpirun and return the completed process.

    --oversubscribe keeps this working on CI runners with fewer detected
    slots than `ranks` (OpenMPI otherwise refuses to start).
    """
    cmd = ["mpirun", "--allow-run-as-root", "--oversubscribe",
           "-np", str(ranks), str(analysis_bin), *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

# Test suite

This drives the compiled `analysis` binary against small, hand-built synthetic trajectories that have a known correct answer, plus a broad smoke test across the rest of the CLI. You'll need MPI (`mpirun`) and CMake/OpenMPI installed, same as for building the project normally, see the root `README.md`.

## Running

```bash
pip install -r tests/requirements.txt
pip install -e plot/    # mdplot, needed by test_endtoend.py
pytest                  # from the repo root
```

The `analysis_bin` fixture in `conftest.py` builds `build/analysis` once per test session (Release, out-of-tree) the first time something needs it, so a plain `pytest` from a clean checkout just works.

## What's actually covered

Each `test_<tool>.py` is built around one specific, closed-form physics prediction rather than a comparison against stored "golden" output. The trajectories are simple enough (straight rods, freely-jointed chains, a uniform random gas, ballistic motion) that I could work out the correct answer by hand, so a failure here means the physics is actually wrong, not just that some number changed:

- `test_msd.py`: a chain drifting at constant velocity across a periodic boundary has to give exact ballistic MSD. This is the regression test for the trajectory-unwrapping bug in `msd_mpi.hpp`.
- `test_msid.py`: a freely-jointed chain gives C(s) ≈ 1, the textbook FJC identity. A straight chain wrapped several times across a small box gives C(s) = s exactly, which tests `unwrapChains()`'s handling of multi-box jumps.
- `test_endtoend.py`: the Gaussian-chain PDF overlay integrates to 1 over Re² (regression test for the Jacobian bug in `mdplot`), and a chain wrapped across a boundary still gives the correct, short end-to-end distance.
- `test_bond_angle.py`: a straight, wrapped chain gives θ ≈ 0° for every triplet.
- `test_pbc_unwrap.py`: a straight rod wrapped across a small box gives the closed-form Rg. This exercises `unwrapPolymerCluster()`, which `volume.hpp` and `spec_density.hpp` both rely on.
- `test_ppa.py`: a straight, wrapped chain gives the closed-form bpp/app/Ne.
- `test_rdf.py`: an uncorrelated point gas gives g(r) ≈ 1.
- `test_smoke_all_tools.py`: everything else (density, nematic, entanglement, gds, brushlength, fellipsoid, gyr, sq, backbone, energy, pressure under MPI, specdensity) just has to run without error and produce non-empty output on a generic trajectory. This is only a safety net against crashes and CLI wiring mistakes, not a physics check, since these tools don't have a simple closed-form answer to check against yet.

What's still missing: there's no true multi-rank (`-np > 1`) correctness check for `msd` itself (only `pressure` is exercised under MPI here), and no dedicated check for `backbone --bbmode single`, for `entanglement`'s spatial binning, or for the force-ellipsoid eigenvalue decomposition beyond confirming it runs.

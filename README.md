# MD Analysis Suite

A C++17 toolkit for analysing molecular dynamics trajectories of polymer systems, paired with an installable Python plotting package (`mdplot`).

19 analysis tools covering structural, mechanical, thermodynamic, and topological observables, all reachable through one `--help` CLI, with MPI and OpenMP used where it actually helps.

## Provenance

I wrote and used these tools throughout my PhD (2022-2026) at MPIP Mainz, adding to them as the physics I was studying demanded. This repo is that work cleaned up, documented, and refactored into one codebase, published in 2026.

AI assistance (Claude) was used for cleanup and documentation work; the underlying analysis code and physics are my own.

---
## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Building](#building)
- [Usage](#usage)
  - [Global options](#global-options)
- [Analysis Tools](#analysis-tools)
  - [Structural / Conformational](#structural--conformational)
  - [Scattering / Density](#scattering--density)
  - [Mechanics / Energy](#mechanics--energy)
  - [Entanglement](#entanglement)
  - [Order / Dynamics](#order--dynamics)
  - [Brush / Interface](#brush--interface)
- [Plotting with mdplot](#plotting-with-mdplot)
- [Input format](#input-format)
- [Output format](#output-format)
- [Project structure](#project-structure)
- [Design notes](#design-notes)

---

## Features

### Structural / Conformational

| Observable | Tool flag | Parallelism |
|---|---|---|
| Mean Squared Displacement g₁(t), g₂(t), g₃(t) | `--tool msd` | MPI |
| Front-resolved monomer MSD g1_swollen(t), g1_dry(t) | `--tool msdfront` | MPI |
| Gyration radius spatial profile ⟨Rg²(x)⟩ | `--tool gyr` | single |
| Mean Squared Internal Distance C(s) | `--tool msid` | single |
| End-to-end distance ⟨Re²⟩ and P(Re²) | `--tool endtoend` | single |
| Bond angle distribution P(θ) | `--tool bondangle` | single |
| Equivalent spherical volume Rg, V_eq | `--tool volume` | single |

### Scattering / Density

| Observable | Tool flag | Parallelism |
|---|---|---|
| Radial distribution function g(r) | `--tool rdf` | OpenMP |
| Spatial density profile ρ(x) | `--tool density` | single |
| Species radial density ρ(r) from CoM | `--tool specdensity` | single |
| Static structure factor S(q) | `--tool sq` | OpenMP |

### Mechanics / Energy

| Observable | Tool flag | Parallelism |
|---|---|---|
| FENE backbone bond-force profile | `--tool backbone` | single |
| Polymer-solvent interaction energy E(λ) | `--tool energy` | OpenMP |
| Irving-Kirkwood layer pressure tensor P(x) | `--tool pressure` | MPI |
| Per-monomer force ellipsoid (anisotropy, prolateness) | `--tool fellipsoid` | OpenMP |

### Entanglement

| Observable | Tool flag | Parallelism |
|---|---|---|
| Primitive Path Analysis: bpp, app, Ne | `--tool ppa` | single |
| Spatial kink / entanglement density | `--tool entanglement` | single |

### Order / Dynamics

| Observable | Tool flag | Parallelism |
|---|---|---|
| Nematic order parameter S(x) and director n̂(x) | `--tool nematic` | single |
| GDS diffusion front, uptake, induction time | `--tool gds` | single |

### Brush / Interface

| Observable | Tool flag | Parallelism |
|---|---|---|
| Brush penetration length distribution P(L) | `--tool brushlength` | single |
---

## Requirements

### C++ analysis suite

| Dependency | Minimum version | Notes |
|---|---|---|
| C++ compiler | GCC 9 / Clang 10 | C++17 required |
| CMake | 3.16 | |
| MPI | OpenMPI 4 / MPICH 3 | Required |
| OpenMP | 4.5 | Optional, used in `fellipsoid`, `energy`, `rdf`, `sq` |

### Python plotting package (mdplot)

| Dependency | Minimum version |
|---|---|
| Python | 3.9 |
| numpy | 1.21 |
| matplotlib | 3.5 |

---

## Building

```bash
git clone https://github.com/volpesc/PolMDanalysis.git
cd PolMDanalysis

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Debug build with sanitizers:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Controlling threads

The OpenMP tools (`energy`, `rdf`, `sq`) take their thread count from the
environment; `fellipsoid` also accepts `--threads`:

```bash
OMP_NUM_THREADS=8 ./analysis --tool rdf --Nm 50 --Nc 200 --start 0 --stop 100
./analysis --tool fellipsoid --threads 8 ...
```

Install the Python plotting package (once):

```bash
cd plot/
pip install -e .
```

### Running the tests

```bash
pip install -r tests/requirements.txt
pip install -e plot/
pytest
```

See `tests/README.md` for what's actually being checked.

---

## Usage

```
mpirun -np <N_ranks> ./analysis --tool <name> [options]
./analysis --help
./analysis --help --tool <name>
```

### Global options

| Option | Type | Default | Description |
|---|---|---|---|
| `--tool` | string | — | Tool to run (**required**) |
| `--help` | flag | — | Global help or `--help --tool <n>` for tool help |
| `--prefix` | string | `requil_` | Trajectory filename prefix |
| `--Nm` | int | `10` | Monomers per chain |
| `--Nc` | int | `10000` | Number of chains |
| `--Ns` | int | `0` | Solvent particles |
| `--start` | int | `0` | First frame index |
| `--stop` | int | `400` | Last frame index |
| `--step` | int | `1` | Frame stride |
| `--frame` | int | `0` | Single frame index |
| `--out` | string | tool-specific | Output file path |

---

## Analysis Tools

### `msd`: Mean Squared Displacement

Computes g₁(t), g₂(t), g₃(t), the three standard polymer MSD observables. g₁ only uses the inner 50% of each chain to avoid end-monomer artifacts. The time-origin loop is MPI-parallelised.

```bash
mpirun -np 4 ./analysis --tool msd \
  --prefix requil_ --Nm 50 --Nc 200 \
  --start 0 --stop 1000 --step 2 --dt 0.01 --out msd.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--dt` | `50.0` | Physical time between frames [τ] |

**Output:** `t  g1(t)  g2(t)  g3(t)`

---

### `msdfront`: Front-Resolved Monomer MSD

Splits g1(t) into two populations by each monomer's position relative to the
instantaneous solvent GDS front, decided fresh at every time origin: swollen
(behind the front, solvent-penetrated) and dry (ahead of the front,
unpenetrated/glassy). A buffer zone straddling the front excludes
ambiguously-placed monomers. Lags are log-spaced and capped by --tmax,
keeping this cheap and focused on early-time diffusion, before the
tube-constraint crossover. Like msd, uses the inner 50% of each chain and is
MPI-parallelised over time origins. Error bars are the standard error of the
per-origin population-mean g1, across time origins.

```bash
mpirun -np 8 ./analysis --tool msdfront \
  --prefix requil_ --Nm 500 --Nc 1000 --Ns 2000000 \
  --start 0 --stop 50 --step 1 --dt 1.0 \
  --xmin 0 --xmax 100 --binw 3.0 \
  --frontbuf 2.0 --tmax 300 --nlog 40 \
  --out msd_front.dat
```

| Extra option | Default | Description |
|---|---|---|
| '--frontbuf' | '2.0' | Exclusion half-width straddling the front [sigma] |
| '--tmax' | '300.0' | Max lag time reported [τ] |
| '--nlog' | '40' | Number of log-spaced output lag points |

**Output:** 't [τ]  g1_swollen[σ^2]  err  g1_dry[σ^2]  err'



---

### `gyr`: Gyration Radius Profile

⟨Rg²(x)⟩ binned by PBC-corrected chain CoM x-position. Longitudinal and transverse components reported separately.

```bash
./analysis --tool gyr --frame 100 --Nm 50 --Nc 200 \
  --binw 2.0 --nbins 80 --out rg_profile.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--binw` | `3.0` | Bin width [σ] |
| `--nbins` | `62` | Number of bins |

**Output:** `x  Rg2_x  Rg2_y  Rg2_z  Rg2_total`

---

### `msid`: Mean Squared Internal Distance

C(s) = ⟨R²(s)⟩ / (s·lb²). FJC limit: C(s) = 1. Deviations encode stiffness and excluded-volume effects.

```bash
./analysis --tool msid --Nm 100 --Nc 300 \
  --start 500 --stop 1000 --step 10 --lb 0.97 --out msid.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--lb` | `0.964` | Bond length lb [σ] |

**Output:** `s  C(s)`

---

### `endtoend`: End-to-End Distance

Bond-by-bond PBC-safe reconstruction of Re. Produces a per-frame time series ⟨Re²⟩ and a histogram P(Re²) for comparison with Gaussian chain theory.

```bash
./analysis --tool endtoend --Nm 50 --Nc 200 \
  --start 0 --stop 500 --nbins 100 --out endtoend
```

**Output files:** `<stem>_time.dat`, `<stem>_hist.dat`

---

### `bondangle`: Bond Angle Distribution

P(θ) for backbone triplets i, i+1, i+2. Reports both raw P(θ) and the solid-angle-corrected P(θ)/sin(θ), which is flat for a freely-jointed chain (since P(θ) ∝ sin(θ) for isotropic orientation).

```bash
./analysis --tool bondangle --Nm 50 --Nc 200 \
  --start 0 --stop 500 --nbins 180 --out bond_angle.dat
```

**Output:** `theta[deg]  P(theta)  P(theta)/sin(theta)`

---

### `volume`: Gyration Volume

Rg of the whole polymer system and the equivalent spherical volume V = (4/3)πRg³. Coordinates are unwrapped bond-by-bond before computing the CoM.

```bash
./analysis --tool volume --frame 100 --Nm 50 --Nc 200 --out volume.dat
```

**Output:** `Rg  V_eq`

---

### `rdf`: Radial Distribution Function

All-pairs g(r) with minimum image convention, averaged over frames. g(r) → 1 at large r for a homogeneous liquid.

```bash
./analysis --tool rdf --Nm 50 --Nc 200 \
  --start 0 --stop 100 --rcut 6.0 --nbins 300 --out rdf.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--rcut` | `5.0` | Max distance [σ] |
| `--nbins` | `200` | Histogram bins |

**Output:** `r  g(r)`

---

### `density`: Density Profile

ρ(x) = ⟨N(x)⟩ / (A·Δx) along x, averaged over frames. Separate profiles for polymer and solvent when `--Ns > 0`.

```bash
./analysis --tool density --Nm 50 --Nc 200 --Ns 500 \
  --start 100 --stop 500 --nbins 150 --out density.dat
```

**Output:** `x  rho_polymer  [rho_solvent]`

---

### `specdensity`: Species Radial Density

Volume-fraction in concentric shells around the polymer centre of mass. Uses hard-sphere diameters σ_monomer = 1.0 σ and σ_solvent = 0.75 σ.

```bash
./analysis --tool specdensity --frame 50 --Nm 50 --Nc 200 \
  --Ns 5000 --nshells 18 --shellw 1.0 --out spec_density.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--nshells` | `18` | Number of radial shells |
| `--shellw` | `1.0` | Shell width [σ] |

**Output:** `r  rho_polymer  rho_solvent`

---

### `sq`: Static Structure Factor

Radially averaged S(q). Two modes (`--mode sc` or `tot`) and two wavevector sampling strategies (`--sampling spherical` or `cartesian`).

```bash
./analysis --tool sq --mode sc --sampling spherical \
  --frame 50 --Nm 100 --Nc 200 --nkbound 80 --out Sc_q.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--mode` | `sc` | `sc` (single-chain) or `tot` (total) |
| `--sampling` | `spherical` | `spherical` or `cartesian` |
| `--nkbound` | `100` | Shells / grid bound |
| `--dkmult` | `1.0` | dk multiplier (cartesian) |
| `--bintol` | `0.1` | Radial binning tolerance |

**Output:** `q  S(q)`

---

### `backbone`: FENE Bond-Force Profile

Projected FENE bond force per bond index along the backbone. Two modes: `full` (all chains averaged per bond index) and `single` (ID-ordered, stops at first broken bond).

FENE parameters: K = 30 ε/σ², R₀ = 1.5 σ (Kremer-Grest).

```bash
./analysis --tool backbone --frame 0 --Nm 50 --Nc 200 \
  --bbmode full --stretchx 1.0 --out backbone.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--bbmode` | `full` | `full` or `single` |
| `--stretchx/y/z` | `1 0 0` | Stretch direction |

**Output:** `bond_index  F_proj`

---

### `energy`: Polymer-Solvent Interaction Energy

WCA + cosine-attractive tail between all polymer-solvent pairs. Attractive amplitude α = 0.5145 λ ε.

```bash
./analysis --tool energy --frame 100 --Nm 50 --Nc 200 \
  --Ns 500 --lambda 0.5 --out energy.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--lambda` | `1.0` | Coupling λ ∈ [0,1] |

**Output:** `lambda  E_total` (appended to file per call)

---

### `pressure`: Irving-Kirkwood Layer Pressure Tensor

Full symmetric stress tensor Pαβ(x) in Voigt notation: kinetic (polymer and solvent), pairwise virial (WCA + cosine-attractive + FENE bonds), and three-body bending. The wall term isn't included, since it needs wall positions/forces that aren't in the trajectory. Pair, bond, and bending work is partitioned across MPI ranks (cell-list neighbour search) and reduced to rank 0. Surface tension: γ = ½ ∫ [PN(x) − PT(x)] dx, using the total tensor.

```bash
mpirun -np 8 ./analysis --tool pressure \
  --frame 500 --Nm 50 --Nc 200 --Ns 1000 --nlayer 200 --out pressure_500
```

| Extra option | Default | Description |
|---|---|---|
| `--nlayer` | `120` | Layers along x |

**Output files:** `<stem>_PK_p.txt` (kinetic, polymer), `<stem>_PK_s.txt` (kinetic, solvent), `<stem>_Upair.txt` (pairwise virial), `<stem>_Ubend.txt` (bending virial), `<stem>_total.txt` (sum of all contributions).
**Output columns:** `x  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz`

---

### `fellipsoid`: Force Ellipsoid

Recomputes all non-bonded forces (WCA + cosine-attractive), bond forces (harmonic), and bending forces for every monomer using a cell-list neighbour search (O(N)). Builds the force covariance tensor T_αβ = ⟨f_α f_β⟩ averaged over frames and diagonalises it analytically (Cardano), so there's no Eigen dependency.

Reports two numbers per monomer: anisotropy (√λ₁/√λ₃, the aspect ratio of the force ellipsoid, 1 means a sphere) and prolateness ((λ₁−λ₂)/(λ₁−λ₃), which runs from 0 for an oblate disc to 1 for a prolate rod).

```bash
./analysis --tool fellipsoid \
  --prefix requil_ --Nm 500 --Nc 1000 --Ns 2000000 \
  --start 0 --stop 10 --step 1 --threads 8 --out interval
```

| Extra option | Default | Description |
|---|---|---|
| `--threads` | `1` | OpenMP threads |
| `--alpha_pp` | `0.5145` | pp attractive amplitude |
| `--alpha_ps` | `0.5145` | ps attractive amplitude |

**Output:** `<stem>_ellipsoid.csv`, columns: `particle_id, fx_avg, fy_avg, fz_avg, fmag_avg, lambda1, lambda2, lambda3, anisotropy, prolateness`

---

### `ppa`: Primitive Path Analysis

Computes bpp (average bond length), app (Kuhn length = ⟨Re²⟩/⟨L⟩), and Ne = app/bpp (entanglement length). Related to plateau modulus: Gₑ ≈ ρ kT / Ne.

```bash
./analysis --tool ppa --frame 0 --Nm 500 --Nc 200 --out ppa.dat
```

**Output (key=value):** `bpp  app  Ne`

---

### `entanglement`: Spatial Entanglement Density

Identifies backbone kinks (bond angles below threshold) and bins them along x. Two modes:

| Mode | Description |
|---|---|
| `chain_com` (default) | One bin per chain based on CoM x. No double-counting. Gives avg kinks per chain. |
| `monomer` | Each kink placed at the middle monomer's position. Tracks chains per bin. |

Recommended threshold: 147° (Kremer-Grest kinks). Default threshold is 147°.

```bash
./analysis --tool entanglement --frame 0 --Nm 500 --Nc 200 \
  --nbins 100 --kinkdeg 147.0 --entmode chain_com --out entanglement.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--entmode` | `chain_com` | `chain_com` or `monomer` (a valid `--bbmode` is also accepted for back-compat) |
| `--kinkdeg` | `147.0` | Bond-angle threshold [°] |
| `--nbins` | `100` | Spatial bins along x |

**Output (chain_com):** `x_center  chains_in_bin  total_kinks  avg_kinks_per_chain`
**Output (monomer):** `x_center  kink_count  chains_in_bin  ent_per_chain`

---

### `nematic`: Nematic Order Parameter

S(x) and director n̂(x) from backbone bond vectors binned along x. Builds the Q-tensor per bin, finds its largest eigenvalue (Cardano) as S and the director by power iteration. Chain end-bonds excluded.

S = 0 means isotropic, S = 1 means perfectly aligned, S ≈ 0.5 is weakly nematic.

```bash
./analysis --tool nematic --frame 0 --Nm 500 --Nc 1000 --nbins 85 --out nematic.dat
```

**Output:** `x  S  nx  ny  nz`

---

### `gds`: GDS Diffusion Front

Tracks the Gibbs Dividing Surface front positions of polymer and solvent species across a range of frames. Computes uptake and detects the induction time (first frame where GDS velocity exceeds threshold).

```bash
./analysis --tool gds --prefix requil_ --Nm 500 --Nc 1000 \
  --Ns 2000000 --start 70 --stop 120 --dt 1000 \
  --xmin 0 --xmax 100 --binw 3.0 --xthresh 45.0 --out gds.csv
```

| Extra option | Default | Description |
|---|---|---|
| `--xmin/xmax` | `0 / 100` | Profile range [σ] |
| `--binw` | `3.0` | Bin width [σ] |
| `--dt` | `1000.0` | Time between frames [τ] |
| `--xthresh` | `45.0` | Polymer uptake threshold [σ] |
| `--vthresh` | `0.01` | Induction velocity threshold |

**Output CSV:** `time, solvent_front, polymer_front, solvent_uptake, polymer_uptake`

---

### `brushlength`: Brush Penetration Length

For chains anchored near the box centre, measures how far each chain extends beyond the solvent front (estimated from the 95th percentile of solvent x-positions). Histograms penetration lengths L.

```bash
./analysis --tool brushlength --frame 100 --Nm 500 --Nc 1000 \
  --Ns 2000000 --percentile 95 --cbuffer 10.0 --out brush_length.dat
```

| Extra option | Default | Description |
|---|---|---|
| `--percentile` | `95.0` | Front detection percentile |
| `--cbuffer` | `10.0` | Centre anchor half-width [σ] |
| `--minbin/maxbin` | `2 / 90` | Output bin range |

**Output:** `L  P(L)`

---

## Plotting with mdplot

All 19 tools have corresponding Python plot functions in the `mdplot` package.

### Install

```bash
cd plot/
pip install -e .
```

### Command line

```bash
mdplot --tool msd        msd.dat --loglog --slopes
mdplot --tool msdfront   msd_front.dat --loglog --slopes
mdplot --tool gyr        rg_profile.dat --components
mdplot --tool msid       msid.dat
mdplot --tool endtoend   endtoend --theory
mdplot --tool bondangle  bond_angle.dat --both
mdplot --tool rdf        rdf.dat --fill
mdplot --tool density    density.dat
mdplot --tool sq         Sc_q.dat --loglog
mdplot --tool pressure   pressure_total.txt --gamma
mdplot --tool nematic    nematic.dat --both
mdplot --tool gds        gds.csv
mdplot --tool brushlength brush_length.dat
```

### Python API: single plot

```python
from mdplot import io, style
from mdplot.observables import plot_msd

data = io.load_msd("msd.dat")
fig, ax = style.figure()
plot_msd(data, ax=ax, loglog=True, show_slopes=True)
style.save(fig, "msd.pdf")
```

### Python API: multi-panel figure

```python
from mdplot import io, style
from mdplot.observables import plot_nematic_full, plot_entanglement, plot_force_ellipsoid

# Nematic: S(x) + director components side by side
fig, _ = plot_nematic_full(io.load_nematic("nematic.dat"))
style.save(fig, "nematic.pdf")

# Force ellipsoid: 3-panel (magnitude, anisotropy, eigenvalues)
from mdplot.observables import load_force_ellipsoid
data = load_force_ellipsoid("interval_ellipsoid.csv")
fig, _ = plot_force_ellipsoid(data)
style.save(fig, "ellipsoid.pdf")
```

### Python API: compose any layout

```python
from mdplot import io, style
from mdplot.observables import plot_rdf, plot_density, plot_sq

fig, axes = style.figure(ncols=3, width=14)
plot_rdf    (io.load_rdf    ("rdf.dat"),     ax=axes[0], fill_first_peak=True)
plot_density(io.load_density("density.dat"), ax=axes[1])
plot_sq     (io.load_sq     ("Sc_q.dat"),    ax=axes[2], loglog=True)
style.save(fig, "structure_overview.pdf")
```

See [`plot/README.md`](plot/README.md) for full `mdplot` documentation.

---

## Input format

Trajectories are numbered XYZ files: `<prefix><index>.xyz`

```
<N_total>  <Lx>  <Ly>  <Lz>
<id>  <type>  <x>  <y>  <z>  <vx>  <vy>  <vz>
...
```

Particles must be ordered chain-major: all Nm monomers of chain 0, then chain 1, etc. Solvent particles follow polymer particles.

---

## Output format

All output files are plain text with a commented header (`#`). Columns are tab-separated. CSV outputs (GDS, force ellipsoid) use comma separation with a header row.

Plotting directly:
```gnuplot
plot "msd.dat" u 1:2 w l title "g1(t)", \
     "msd.dat" u 1:3 w l title "g2(t)", \
     "msd.dat" u 1:4 w l title "g3(t)"
```

---

## Project structure

```
PolMDanalysis/
├── analysis.cpp              # Entry point: CLI, --help, registry dispatch
├── CMakeLists.txt            # Build: MPI required, OpenMP optional, sanitizers in Debug
├── pytest.ini
├── .gitignore
├── README.md
├── .github/workflows/ci.yml  # Builds the project and runs the test suite
│
├── tests/                    # pytest suite, drives the compiled binary against
│   │                         # synthetic trajectories with known analytic answers
│   ├── conftest.py           # Builds analysis once per session, run_tool() helper
│   ├── _xyz.py                # .xyz trajectory-frame writer
│   ├── test_msd.py, test_msid.py, test_endtoend.py, ...
│   └── test_smoke_all_tools.py # everything else: runs, doesn't crash, non-empty output
│
├── func/                     # Header-only C++ analysis modules
│   ├── utility.hpp           # Data structures, I/O, force models, CoM helpers
│   ├── args.hpp              # CLI arguments struct + parser
│   ├── analysis_base.hpp     # Analysis interface + self-registering Registry
│   ├── tools.hpp             # Concrete tool classes (one per --tool), self-registered
│   │
│   │   ── Structural / Conformational ──────────────────────────────────────
│   ├── msd_mpi.hpp           # MSD g1, g2, g3  (MPI)
│   ├── msd_front.hpp         # Front resolved MSD g1_swollen, g2_dry  (MPI)
│   ├── gyr_endz.hpp          # Gyration radius profile ⟨Rg²(x)⟩
│   ├── msid.hpp              # Mean Squared Internal Distance C(s)
│   ├── endtoend.hpp          # End-to-end distance ⟨Re²⟩, P(Re²)
│   ├── bond_angle.hpp        # Bond angle distribution P(θ)
│   ├── volume.hpp            # Gyration volume Rg, V_eq
│   │
│   │   ── Scattering / Density ────────────────────────────────────────────
│   ├── rdf.hpp               # Radial distribution function g(r)
│   ├── density.hpp           # Density profile ρ(x)
│   ├── spec_density.hpp      # Species radial density ρ(r)
│   ├── structure_factor.hpp  # S(q): Sc and Stot, Cartesian+Spherical
│   │
│   │   ── Mechanics / Energy ──────────────────────────────────────────────
│   ├── backbone.hpp          # FENE bond-force profile
│   ├── energy.hpp            # Polymer-solvent interaction energy
│   ├── pressure_z_mpi.hpp    # IK layer pressure tensor  (MPI)
│   ├── force_ellipsoid.hpp   # Force covariance tensor, ellipsoid (OpenMP)
│   │
│   │   ── Entanglement ─────────────────────────────────────────────────────
│   ├── ppa.hpp               # Primitive Path Analysis: bpp, app, Ne
│   ├── entanglement.hpp      # Kink/entanglement density (chain_com + monomer)
│   │
│   │   ── Order / Dynamics ─────────────────────────────────────────────────
│   ├── nematic.hpp           # Nematic order S(x), director n̂(x)
│   ├── gds_diffusion.hpp     # GDS diffusion front, uptake, induction time
│   │
│   │   ── Brush / Interface ─────────────────────────────────────────────────
│   └── brush_length.hpp      # Brush penetration length P(L)
│
└── plot/                     # Python plotting package (mdplot)
    ├── pyproject.toml        # pip-installable; registers `mdplot` command
    ├── README.md             # Full mdplot documentation
    └── mdplot/
        ├── __init__.py       # Version, top-level imports
        ├── style.py          # Shared style: colour palette, rcParams, figure factory
        ├── io.py             # Data loaders → dicts of named arrays
        ├── cli.py            # `mdplot` CLI entry point
        └── observables/      # One plot function (or module) per observable
            ├── msd.py        # plot_msd()
            ├── msd_front.py  # plot_msd_front()
            ├── gyr.py        # plot_gyr()
            ├── msid.py       # plot_msid()
            ├── endtoend.py   # plot_endtoend(), plot_endtoend_time/hist()
            ├── bondangle.py  # plot_bondangle(), plot_bondangle_both()
            ├── rdf.py        # plot_rdf()
            ├── density.py    # plot_density()
            ├── spec_density.py # plot_spec_density()
            ├── sq.py         # plot_sq()
            ├── pressure.py   # plot_pressure(), surface_tension()
            ├── backbone.py   # plot_backbone()
            ├── energy.py     # plot_energy()
            ├── force_ellipsoid.py # plot_force_ellipsoid(), anisotropy, eigenvalues
            ├── ppa.py        # plot_ppa(), load_ppa()
            ├── entanglement.py # plot_entanglement()
            ├── nematic.py    # plot_nematic(), plot_nematic_full()
            ├── gds_diffusion.py # plot_gds(), plot_gds_fronts/uptake()
            └── brush_length.py  # plot_brush_length()
```

---

## Design notes

### Architecture

The `Analysis` interface (`func/analysis_base.hpp`) is what every tool implements: a class with one `run(const Args&)` method. The driver only ever depends on this interface.

`Registry` maps a name to a tool factory. Tools register themselves at static initialisation through an inline `Register<T>` object, so the driver just finds them, nothing has to list them by hand.

Each tool in `func/tools.hpp` wraps its own `compute*()` function unchanged: `run()` just translates `Args` into that tool's `Config` and calls the numerics. Computation and I/O stay separate; the current test suite (see `tests/`) checks them the other way, through the CLI, against synthetic trajectories with known answers.

`sq` uses a Strategy pattern for its k-space sampling: `SphericalSampling` and `CartesianSampling` are picked at runtime and threaded through one generic dispatch lambda.

So `main()` just parses arguments, looks the tool up in the registry, and runs it. Tools that have to execute on every MPI rank (`msd`, `pressure`) override `runsOnAllRanks()`; everything else runs on rank 0 only.

### Adding a tool

You don't need to touch `analysis.cpp` for this:

```cpp
// 1. a Config + compute*() (your numerics)               -> func/mytool.hpp
// 2. a class implementing the interface                  -> func/tools.hpp
class MyAnalysis : public md::Analysis {
public:
    void run(const md::Args& a) const override {
        MyConfig c; c.Nm = a.Nm; c.Nc = a.Nc; /* ... */
        computeMyThing(c, a.out.empty() ? "mytool.dat" : a.out);
    }
};
// 3. one self-registration line
inline const md::Register<MyAnalysis> reg_mytool{"mytool"};
```




# mdplot: MD Analysis Suite Plotting Package

Python package for visualising the output of the MD Analysis Suite. Works both as a command-line tool and as an importable library for scripts and Jupyter notebooks.

---

## Installation

```bash
cd plot/
pip install -e .
```

This installs the `mdplot` command and makes `import mdplot` available.

Dependencies: `numpy`, `matplotlib` (both installed automatically).

---

## Command-line usage

```
mdplot --tool <name> <input> [options]
mdplot --help
mdplot --help --tool <name>
```

### Examples

```bash
# MSD on log-log axes with reference slopes
mdplot --tool msd msd.dat --loglog --slopes --out msd.pdf

# RDF with first-peak shading
mdplot --tool rdf rdf.dat --fill --out rdf.pdf

# Density profile (auto-detects solvent column)
mdplot --tool density density.dat --out density.pdf

# End-to-end: side-by-side time series + histogram with Gaussian theory
mdplot --tool endtoend endtoend --theory --out re.pdf

# Bond angle: both raw and solid-angle-weighted panels
mdplot --tool bondangle bond_angle.dat --both --out angle.pdf

# Compare Sc(q) and S(q) on log-log axes
mdplot --tool sq Sc_q.dat Stot_q.dat --labels "Sc(q)" "S(q)" --loglog --out sq.pdf

# Pressure tensor with surface tension estimate
mdplot --tool pressure pressure_PK_p.txt pressure_PK_s.txt \
       --labels PK_p PK_s --gamma --out pressure.pdf
```

---

## Python API

Every plot function accepts an optional `ax` argument so it can be embedded
into any figure layout, including multi-panel publications figures.

### Basic use

```python
from mdplot import io, style
from mdplot.observables import plot_msd

data = io.load_msd("msd.dat")

fig, ax = style.figure()
plot_msd(data, ax=ax, loglog=True, show_slopes=True)
style.save(fig, "msd.pdf")
```

### Multi-panel figure

```python
from mdplot import io, style
from mdplot.observables import plot_rdf, plot_density, plot_sq

fig, axes = style.figure(ncols=3, width=14)

plot_rdf    (io.load_rdf    ("rdf.dat"),     ax=axes[0])
plot_density(io.load_density("density.dat"), ax=axes[1])
plot_sq     (io.load_sq     ("Sc_q.dat"),    ax=axes[2])

style.save(fig, "overview.pdf")
```

### Jupyter notebook

```python
%matplotlib inline
import mdplot          # applies shared style automatically on import
from mdplot import io
from mdplot.observables import plot_endtoend

time_data = io.load_endtoend_time("endtoend_time.dat")
hist_data = io.load_endtoend_hist("endtoend_hist.dat")

fig, (ax1, ax2) = plot_endtoend(time_data, hist_data, show_theory=True)
fig.show()
```

---

## Package structure

```
plot/
├── pyproject.toml          # pip-installable; registers `mdplot` command
└── mdplot/
    ├── __init__.py         # top-level imports + version
    ├── style.py            # shared rcParams, colour palette, figure factory
    ├── io.py               # data loaders (return plain dicts of arrays)
    ├── cli.py              # `mdplot` CLI entry point
    └── observables/        # one module per --tool, same names as the CLI
        ├── __init__.py     # re-exports all plot functions
        ├── msd.py, gyr.py, msid.py, rdf.py, density.py, spec_density.py
        ├── endtoend.py, bondangle.py, sq.py, pressure.py
        ├── backbone.py, energy.py, force_ellipsoid.py
        ├── ppa.py, entanglement.py
        └── nematic.py, gds_diffusion.py, brush_length.py
```

## Design principles

I kept the layers separate on purpose: `io.py` only deals with reading files, `style.py` only with colours and fonts, `observables/` only with the physics of each plot, and `cli.py` just wires the three together. That way changing the colour scheme in `style.py` doesn't touch any plotting logic, and none of it depends on the CLI at all.

Every `plot_*` function takes an `ax=None` argument. If you pass one, it draws into it and returns it, so building a multi-panel figure doesn't need any special API beyond matplotlib's own. Since the observable modules are just plotting functions and the CLI is a thin layer on top of them, the same code works the same way from a terminal, a script, or a notebook.

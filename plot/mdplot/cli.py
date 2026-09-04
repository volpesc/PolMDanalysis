"""
mdplot.cli
==========
Single command-line entry point for the mdplot package.

Installed as the ``mdplot`` command via pyproject.toml.

Usage
-----
::

    mdplot --tool msd msd.dat [options]
    mdplot --tool rdf rdf.dat --fill --out rdf.pdf
    mdplot --tool endtoend endtoend --theory --out endtoend.pdf
    mdplot --help
    mdplot --help --tool msd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mdplot import io, style
from mdplot import observables as obs


# ── Per-tool help strings ─────────────────────────────────────────────────────

TOOL_HELP = {
    "msd": """
Plot Mean Squared Displacement g1(t), g2(t), g3(t).

  Input:  msd.dat  (columns: t  g1  g2  g3)
  Flags:  --loglog       Log-log axes
          --slopes       Overlay t¹ and t^0.5 reference lines (requires --loglog)
          --out FILE     Output path  [msd.pdf]

  Example:
    mdplot --tool msd msd.dat --loglog --slopes --out msd_loglog.pdf
""",
    "gyr": """
Plot gyration radius profile <Rg²(x)>.

  Input:  rg_profile.dat  (columns: x  Rg2_x  Rg2_y  Rg2_z  Rg2_total)
  Flags:  --components   Also plot x, y, z components
          --out FILE     Output path  [gyr.pdf]

  Example:
    mdplot --tool gyr rg_profile.dat --components --out rg.pdf
""",
    "msid": """
Plot Mean Squared Internal Distance C(s).

  Input:  msid.dat  (columns: s  C(s))
  Flags:  --no-fjc   Hide the FJC reference line C(s)=1
          --out FILE Output path  [msid.pdf]

  Example:
    mdplot --tool msid msid.dat --out msid.pdf
""",
    "rdf": """
Plot radial distribution function g(r).

  Input:  rdf.dat  (columns: r  g(r))
  Flags:  --fill    Shade area under the first peak
          --out FILE Output path  [rdf.pdf]

  Example:
    mdplot --tool rdf rdf.dat --fill --out rdf.pdf
""",
    "density": """
Plot density profile rho(x).

  Input:  density.dat  (columns: x  rho_polymer  [rho_solvent])
  Flags:  --no-total   Hide total density line when solvent is present
          --out FILE   Output path  [density.pdf]

  Example:
    mdplot --tool density density.dat --out density.pdf
""",
    "endtoend": """
Plot end-to-end distance time series and distribution.

  Input:  FILE stem (produces <stem>_time.dat and <stem>_hist.dat)
          OR pass two files: <time_file> <hist_file>
  Flags:  --theory   Overlay Gaussian chain prediction on histogram
          --time     Plot time series only
          --hist     Plot histogram only
          --out FILE Output path  [endtoend.pdf]

  Example:
    mdplot --tool endtoend endtoend --theory --out re.pdf
""",
    "bondangle": """
Plot bond angle distribution P(theta).

  Input:  bond_angle.dat  (columns: theta[deg]  P(theta)  P(theta)*sin(theta))
  Flags:  --both     Side-by-side raw and solid-angle-weighted panels
          --sin      Plot P(theta)*sin(theta) instead of P(theta)
          --out FILE Output path  [bond_angle.pdf]

  Example:
    mdplot --tool bondangle bond_angle.dat --both --out angle.pdf
""",
    "sq": """
Plot static structure factor S(q).

  Input:  One or more S(q) files (columns: q  S(q))
  Flags:  --labels L1 L2 ...  Legend labels
          --loglog             Log-log axes
          --out FILE           Output path  [sq.pdf]

  Example:
    mdplot --tool sq Sc_q.dat Stot_q.dat --labels "Sc(q)" "S(q)" --out sq.pdf
""",
    "pressure": """
Plot Irving-Kirkwood layer pressure tensor P(x).

  Input:  One or more pressure files (columns: x Pxx Pyy Pzz Pxy Pxz Pyz)
  Flags:  --labels L1 L2 ...       Source labels
          --components C1 C2 ...   Voigt components to plot  [Pxx Pyy Pzz]
          --gamma                  Print surface tension estimate γ
          --out FILE               Output path  [pressure.pdf]

  Example:
    mdplot --tool pressure pressure_PK_p.txt pressure_PK_s.txt \\
           --labels PK_p PK_s --gamma --out pressure.pdf
""",
}

GLOBAL_HELP = """
mdplot — MD Analysis Suite plotting tool
════════════════════════════════════════

USAGE
  mdplot --tool <name> <input> [options]
  mdplot --help
  mdplot --help --tool <name>

AVAILABLE TOOLS
  msd         Mean Squared Displacement g1, g2, g3
  gyr         Gyration radius profile <Rg²(x)>
  msid        Mean Squared Internal Distance C(s)
  rdf         Radial distribution function g(r)
  density     Density profile rho(x)
  endtoend    End-to-end distance time series and distribution
  bondangle   Bond angle distribution P(theta)
  sq          Static structure factor S(q)
  pressure    Layer pressure tensor P(x)

COMMON OPTIONS
  --out FILE  Output figure path  (default: <tool>.pdf)

PYTHON API
  All plot functions are importable directly:

    from mdplot import io, style
    from mdplot.observables import plot_msd

    data = io.load_msd("msd.dat")
    fig, ax = style.figure()
    plot_msd(data, ax=ax, loglog=True)
    style.save(fig, "msd.pdf")
"""


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdplot",
        description="MD Analysis Suite — plotting tool",
        add_help=False,
    )
    p.add_argument("--tool",    metavar="NAME",  help="Analysis tool to plot")
    p.add_argument("--help",    action="store_true")
    p.add_argument("--out",     metavar="FILE",  help="Output figure path")
    p.add_argument("--loglog",  action="store_true")
    p.add_argument("--slopes",  action="store_true")
    p.add_argument("--components",  nargs="*")
    p.add_argument("--labels",      nargs="*")
    p.add_argument("--fill",    action="store_true")
    p.add_argument("--theory",  action="store_true")
    p.add_argument("--both",    action="store_true")
    p.add_argument("--sin",     action="store_true")
    p.add_argument("--time",    action="store_true")
    p.add_argument("--hist",    action="store_true")
    p.add_argument("--no-fjc",  action="store_true", dest="no_fjc")
    p.add_argument("--no-total",action="store_true", dest="no_total")
    p.add_argument("--gamma",   action="store_true")
    p.add_argument("inputs",    nargs="*", metavar="FILE")
    return p


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _require(inputs: list, n: int, tool: str) -> None:
    if len(inputs) < n:
        print(f"Error: --tool {tool} requires {n} input file(s).")
        sys.exit(1)


def handle_msd(args) -> None:
    _require(args.inputs, 1, "msd")
    data = io.load_msd(args.inputs[0])
    fig, ax = style.figure()
    obs.plot_msd(data, ax=ax, loglog=args.loglog, show_slopes=args.slopes)
    style.save(fig, args.out or "msd.pdf")


def handle_gyr(args) -> None:
    _require(args.inputs, 1, "gyr")
    data = io.load_gyr(args.inputs[0])
    fig, ax = style.figure()
    obs.plot_gyr(data, ax=ax, show_components=args.components is not None)
    style.save(fig, args.out or "gyr.pdf")


def handle_msid(args) -> None:
    _require(args.inputs, 1, "msid")
    data = io.load_msid(args.inputs[0])
    fig, ax = style.figure()
    obs.plot_msid(data, ax=ax, show_fjc=not args.no_fjc)
    style.save(fig, args.out or "msid.pdf")


def handle_rdf(args) -> None:
    _require(args.inputs, 1, "rdf")
    data = io.load_rdf(args.inputs[0])
    fig, ax = style.figure()
    obs.plot_rdf(data, ax=ax, fill_first_peak=args.fill)
    style.save(fig, args.out or "rdf.pdf")


def handle_density(args) -> None:
    _require(args.inputs, 1, "density")
    data = io.load_density(args.inputs[0])
    fig, ax = style.figure()
    obs.plot_density(data, ax=ax, show_total=not args.no_total)
    style.save(fig, args.out or "density.pdf")


def handle_endtoend(args) -> None:
    # Accept either a stem or explicit time/hist files
    if len(args.inputs) == 2:
        time_path, hist_path = args.inputs
    elif len(args.inputs) == 1:
        stem = args.inputs[0]
        time_path = stem + "_time.dat"
        hist_path = stem + "_hist.dat"
    else:
        _require(args.inputs, 1, "endtoend")

    time_data = io.load_endtoend_time(time_path)
    hist_data = io.load_endtoend_hist(hist_path)
    out = args.out or "endtoend.pdf"

    if args.time:
        fig, ax = style.figure()
        obs.plot_endtoend_time(time_data, ax=ax)
    elif args.hist:
        fig, ax = style.figure()
        obs.plot_endtoend_hist(hist_data, time_data=time_data, ax=ax,
                               show_theory=args.theory)
    else:
        fig, _ = obs.plot_endtoend(time_data, hist_data, show_theory=args.theory)

    style.save(fig, out)


def handle_bondangle(args) -> None:
    _require(args.inputs, 1, "bondangle")
    data = io.load_bondangle(args.inputs[0])
    out  = args.out or "bond_angle.pdf"

    if args.both:
        fig, _ = obs.plot_bondangle_both(data)
    else:
        fig, ax = style.figure()
        obs.plot_bondangle(data, ax=ax, solid_angle=args.sin)

    style.save(fig, out)


def handle_sq(args) -> None:
    _require(args.inputs, 1, "sq")
    datasets = [io.load_sq(f) for f in args.inputs]
    fig, ax  = style.figure()
    obs.plot_sq(datasets, labels=args.labels, ax=ax, loglog=args.loglog)
    style.save(fig, args.out or "sq.pdf")


def handle_pressure(args) -> None:
    _require(args.inputs, 1, "pressure")
    datasets = [io.load_pressure(f) for f in args.inputs]
    fig, ax  = style.figure(width=8)
    obs.plot_pressure(datasets, labels=args.labels, ax=ax,
                      components=args.components,
                      show_gamma=args.gamma)
    style.save(fig, args.out or "pressure.pdf")


# ── Dispatch table ────────────────────────────────────────────────────────────

HANDLERS = {
    "msd":       handle_msd,
    "gyr":       handle_gyr,
    "msid":      handle_msid,
    "rdf":       handle_rdf,
    "density":   handle_density,
    "endtoend":  handle_endtoend,
    "bondangle": handle_bondangle,
    "sq":        handle_sq,
    "pressure":  handle_pressure,
}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.help:
        if args.tool and args.tool in TOOL_HELP:
            print(TOOL_HELP[args.tool])
        else:
            print(GLOBAL_HELP)
        sys.exit(0)

    if not args.tool:
        print("Error: --tool is required.  Run 'mdplot --help' for usage.")
        sys.exit(1)

    if args.tool not in HANDLERS:
        print(f"Error: unknown tool '{args.tool}'.  Run 'mdplot --help'.")
        sys.exit(1)

    HANDLERS[args.tool](args)


if __name__ == "__main__":
    main()

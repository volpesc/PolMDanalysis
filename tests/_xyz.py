"""Low-level .xyz trajectory-frame writer for the test suite.

File format matches func/utility.hpp::readFrame(): a header line
"N Lx Ly Lz" followed by N atom lines "id type x y z vx vy vz", with
polymer atoms first (chain-major: all Nm monomers of chain 0, then
chain 1, ...) and any solvent atoms after them -- see README.md,
"Input format".
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def write_frame(path: Path, positions: Sequence[Vec3], Lx: float, Ly: float,
                 Lz: float, velocities: Iterable[Vec3] | None = None,
                 type_of: "callable[[int], int]" = lambda i: 1) -> None:
    """Write one trajectory frame.

    type_of(i) selects the atom "type" column for atom index i (0-based);
    the analysis tools never read it, but it lets a caller mark solvent
    atoms distinctly for readability.
    """
    n = len(positions)
    vel = list(velocities) if velocities is not None else [(0.0, 0.0, 0.0)] * n
    lines = [f"{n} {Lx} {Ly} {Lz}"]
    for i, ((x, y, z), (vx, vy, vz)) in enumerate(zip(positions, vel)):
        lines.append(f"{i+1} {type_of(i)} {x:.6f} {y:.6f} {z:.6f} "
                     f"{vx:.6f} {vy:.6f} {vz:.6f}")
    path.write_text("\n".join(lines) + "\n")

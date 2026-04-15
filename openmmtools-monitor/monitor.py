#! /usr/bin/env python3
import curses
import os
import logging
import math
import textwrap
import time
import types
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Generator, Sequence
from typing import Any, Protocol
from datetime import datetime
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # must be set before HDF5 loads

import netCDF4
import numpy as np
import yaml
from cyclopts import App  # type: ignore[import-untyped]

_LOG = logging.getLogger(__name__)

yaml.add_multi_constructor(
    "!",
    lambda loader, tag, node: loader.construct_mapping(node),  # type: ignore[arg-type]
    Loader=yaml.SafeLoader,
)

app = App()

KB_KJMOL_PER_K = 0.008314462618  # kJ/mol/K (NIST 2018 CODATA)

_SPARK_HELP = [
    # 0 — reduced U
    ("Energy of whichever replica currently occupies state 0 (the fully-unscaled "
     "REST2 state), converted from dimensionless reduced potential (kT) to kJ/mol. "
     "Should fluctuate around a stable mean once the system is equilibrated; "
     "a sustained drift indicates incomplete equilibration. "
     "Yellow reference line: energy at the first trajectory frame."),
    # 1 — volume
    ("Box volume (nm\u00b3) of the replica currently in state 0. "
     "Should fluctuate around a stable mean under the NPT barostat; "
     "a drifting mean indicates the density has not yet equilibrated. "
     "Yellow reference line: running mean of all observed volumes — "
     "the best current estimate of the barostat equilibrium volume."),
    # 2 — online \u0394F (0\u2192N-1)
    ("Total free energy difference from state 0 (unscaled) to state N-1 (most "
     "scaled), estimated by offline MBAR and updated at checkpoint intervals. "
     "Values should converge quickly and remain stable; "
     "large changes late in the run suggest insufficient sampling. "
     "No reference line."),
    # 3 — RMSD to first frame
    ("Kabsch RMSD (\u00c5) of the state-0 replica vs the very first trajectory frame, "
     "using optimal rigid-body superposition (translation + rotation). "
     "Shows structural drift from the starting conformation. "
     "Not a good convergence metric for flexible systems: "
     "a flexible peptide that re-visits its starting conformation looks converged "
     "even if large regions of conformational space are unexplored. "
     "No reference line."),
    # 4 — min-RMSD
    ("For each frame: minimum Kabsch RMSD (\u00c5) to any previously seen frame. "
     "High early on when every structure is novel; "
     "converges to a thermal noise floor once all accessible conformations "
     "have been visited at least once. "
     "Yellow reference line: all-pairs minimum RMSD — the closest any two frames "
     "have ever been, i.e. the thermal noise floor. "
     "When the curve plateaus at the reference line, the simulation has exhausted "
     "conformational space and is only revisiting structures within thermal fluctuations."),
    # 5 — max-RMSD
    ("For each frame: maximum Kabsch RMSD (\u00c5) to any previously seen frame — "
     "the structural eccentricity of that frame within the explored ensemble. "
     "High when a frame is far from all known structures; "
     "stabilises once the ensemble diameter is fully covered. "
     "Unlike min-RMSD (which measures novelty) this measures reach: "
     "a central frame has small max-RMSD even if it is novel. "
     "Yellow reference line: all-pairs maximum RMSD observed so far — "
     "the structural diameter of the trajectory, which the curve converges toward."),
    # 6 — RMSD ACF
    ("Mean Kabsch RMSD (\u00c5) between all pairs of state-0 frames separated by a "
     "given lag time. X-axis is logarithmic lag time (not simulation time), "
     "so each decade of lag gets equal visual space — the rise from zero is "
     "clearly resolved even when the plateau spans orders of magnitude longer. "
     "Rises from 0 at lag=0 and plateaus at the structural variance of the ensemble. "
     "The lag at which it plateaus is the conformational decorrelation time. "
     "Faster decorrelation in REST2 vs plain MD indicates the enhanced sampling is working. "
     "Yellow reference line: estimated plateau value (mean of the highest-lag half) — "
     "converges toward the true ensemble structural variance as more frames accumulate."),
]

_BAR_CHARS = " ▁▂▃▄▅▆▇█"  # index by round(fraction * 6); max displayed is ▆ (index 6)

_CP_RED    = 1
_CP_YELLOW = 2
_CP_GREEN  = 3
_CP_BLUE   = 4
_CP_GREY   = 5
_CP_DIM    = 6  # dimmed colour for reference line

# 256-colour gradient: red → yellow → green (sweet spot ~25%) → blue (over-mixed)
_N_GRADIENT = 24
_CP_GRADIENT_START = 7

_GRADIENT_KEY_POINTS = [
    # (gradient_fraction, r, g, b)  — r/g/b in 0-5 (256-colour cube)
    # gradient_fraction = sqrt(rate / 100), so:
    #   frac 0.00 → rate  0%
    #   frac 0.32 → rate 10%
    #   frac 0.50 → rate 25%  ← sweet spot
    #   frac 0.71 → rate 50%
    #   frac 1.00 → rate 100%
    (0.00, 5, 0, 0),  # rate  0%  red
    (0.32, 5, 4, 0),  # rate 10%  orange
    (0.50, 0, 5, 0),  # rate 25%  green  ← sweet spot
    (0.75, 0, 2, 4),  # rate 56%  teal
    (1.00, 0, 0, 5),  # rate 100% blue
]


def _init_gradient_pairs() -> list[int]:
    """Initialise _N_GRADIENT curses color pairs for the acceptance rate gradient."""
    pairs = []
    for i in range(_N_GRADIENT):
        t = i / (_N_GRADIENT - 1)
        # Fallback to last key point (handles t==1.0 exactly)
        _, r, g, b = _GRADIENT_KEY_POINTS[-1]
        for j in range(len(_GRADIENT_KEY_POINTS) - 1):
            t0, r0, g0, b0 = _GRADIENT_KEY_POINTS[j]
            t1, r1, g1, b1 = _GRADIENT_KEY_POINTS[j + 1]
            if t <= t1 + 1e-9:
                s = (t - t0) / (t1 - t0)
                r = round(r0 + s * (r1 - r0))
                g = round(g0 + s * (g1 - g0))
                b = round(b0 + s * (b1 - b0))
                break
        color_idx = 16 + 36 * r + 6 * g + b
        pair_id = _CP_GRADIENT_START + i
        curses.init_pair(pair_id, color_idx, -1)
        pairs.append(pair_id)
    return pairs


def _read_thermo_state(nc: netCDF4.Dataset) -> dict[str, Any]:
    """Parse the first thermodynamic state from the already-open analysis nc handle."""
    raw = b"".join(nc.groups["thermodynamic_states"].variables["state0"][:]).decode()
    return yaml.safe_load(raw)


def _read_mcmc_move(nc: netCDF4.Dataset) -> dict[str, Any]:
    """Parse the MCMC moves from the already-open analysis nc handle.

    One move is stored per replica. Asserts there is one move per replica and
    that all moves are identical, then returns the first move's doc.
    """
    move_vars = nc.groups["mcmc_moves"].variables
    n_replicas = nc.variables["energies"].shape[1]
    assert len(move_vars) == n_replicas and all(
        move_vars[f"move{i}"][0] == move_vars["move0"][0] for i in range(1, n_replicas)
    ), f"Expected {n_replicas} identical MCMC moves, got: {list(move_vars)}"
    return yaml.safe_load(str(move_vars["move0"][0]))


def _load_masses_from_nc(nc: netCDF4.Dataset) -> tuple[np.ndarray[Any, np.dtype[Any]], list[tuple[int, int]], bool]:
    """Read particle masses (amu), constraint particle pairs, and CMMotionRemover
    presence from the standard_system in the nc file."""
    doc = _read_thermo_state(nc)
    system_bytes = doc["standard_system"]  # bytes, decoded by yaml !!binary
    system_xml = zlib.decompress(system_bytes).decode()
    root = ET.fromstring(system_xml)
    particles_elem = root.find("Particles")
    assert particles_elem is not None, "System XML has no <Particles> element"
    masses = np.array([float(p.attrib["mass"]) for p in particles_elem])
    constraints_elem = root.find("Constraints")
    constraints = (
        [(int(c.attrib["p1"]), int(c.attrib["p2"])) for c in constraints_elem]
        if constraints_elem is not None else []
    )
    forces_elem = root.find("Forces")
    has_cm_remover = forces_elem is not None and any(
        f.attrib.get("type") == "CMMotionRemover" for f in forces_elem
    )
    return masses, constraints, has_cm_remover


def _ke_from_velocities(velocities: np.ndarray[Any, np.dtype[Any]], masses_amu: np.ndarray[Any, np.dtype[Any]]) -> float:
    """Compute KE in kJ/mol from velocities (nm/ps) and masses (amu).
    1 amu*(nm/ps)^2 = 1 kJ/mol.  Overflow (unwritten NC frames) produces inf,
    which the caller detects via math.isfinite().
    """
    with np.errstate(over="ignore"):
        return 0.5 * float(np.sum(masses_amu[:, None] * velocities**2))


def _rate_colour_pair(rate: float, gradient: list[int] | None) -> int:
    if math.isnan(rate):
        return _CP_GREY
    if gradient is not None:
        idx = min(int(math.sqrt(rate / 100) * len(gradient)), len(gradient) - 1)
        return gradient[idx]
    # fallback for 8-colour terminals
    if rate < 1:
        return _CP_RED
    elif rate < 15:
        return _CP_YELLOW
    elif rate < 40:
        return _CP_GREEN
    else:
        return _CP_BLUE


def _addstr(stdscr: curses.window, text: str, attr: int = 0) -> None:
    """Write text to stdscr, ignoring writes that exceed terminal bounds."""
    try:
        stdscr.addstr(text, attr) if attr else stdscr.addstr(text)
    except curses.error:
        pass


# Braille dot bit values indexed by [sub_col][sub_row].
# sub_col: 0 = left half of cell, 1 = right half.
# sub_row: 0 = top, 3 = bottom (within the cell's 4-row band).
# Unicode braille block starts at U+2800; add bit pattern to get character.
_BRAILLE_BITS: list[list[int]] = [
    [0x01, 0x02, 0x04, 0x40],  # left column
    [0x08, 0x10, 0x20, 0x80],  # right column
]


def _build_sparkline_grid(
    ground_u_iters: Sequence[int],
    ground_u_history: list[float],
    total_iters: int,
    height: int,
    width: int,
    cp_mean: int,
    cp_range: int,
    cp_ref: int = 0,
    ref_val: float | None = None,
    x_transform: Callable[[int], float] | None = None,
) -> tuple[list[list[tuple[str, int]]], float, float]:
    """Build a 2-D sparkline grid.

    Uses braille (2×4 sub-cell resolution) when data is dense enough to fill
    at least one point per terminal column.  Falls back to a plain character
    grid (● mean, │ range, ─ reference) when data is sparse, so isolated
    data points are shown as clearly positioned dots rather than lonely braille
    specks scattered across empty cells.

    *x_transform*, if given, maps each integer iter/lag to a float x-coordinate
    (e.g. ``math.log10`` for the ACF mode's log x-axis).  The transformed upper
    bound is ``x_transform(total_iters) * 1.05`` to leave a small right margin.

    Returns (grid, y_min, y_max).  grid[row][col] = (char, curses_attr).
    """
    empty: tuple[str, int] = (" ", 0)
    grid: list[list[tuple[str, int]]] = [[empty] * width for _ in range(height)]

    if not ground_u_history:
        return grid, 0.0, 1.0

    # Pre-compute x-coordinates in display space (float, same scale as x_total)
    if x_transform is not None:
        x_positions = [x_transform(it) for it in ground_u_iters]
        x_total     = x_transform(total_iters) * 1.05
    else:
        x_positions = [float(it) for it in ground_u_iters]
        x_total     = float(total_iters)

    y_min = min(ground_u_history)
    y_max = max(ground_u_history)
    if ref_val is not None:
        y_min = min(y_min, ref_val)
        y_max = max(y_max, ref_val)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.05
    y_min -= pad
    y_max += pad

    def _cell_row(v: float) -> int:
        """Map value to terminal row (0 = top)."""
        frac = (v - y_min) / (y_max - y_min)
        return max(0, min(height - 1, round((1.0 - frac) * (height - 1))))

    # ── Sparse fallback: plain ●/│/─ characters ───────────────────────────
    if len(ground_u_history) < width:
        # Reference line first (lowest priority — overwritten by data)
        if ref_val is not None:
            ref_r = _cell_row(ref_val)
            for c in range(width):
                grid[ref_r][c] = ("─", cp_ref)

        for c in range(width):
            lo_it = c / width * x_total
            hi_it = (c + 1) / width * x_total
            vals = [e for x, e in zip(x_positions, ground_u_history)
                    if lo_it <= x < hi_it]
            if not vals:
                continue
            mean_r = _cell_row(float(np.mean(vals)))
            top_r  = _cell_row(max(vals))
            bot_r  = _cell_row(min(vals))
            for r in range(top_r, bot_r + 1):
                grid[r][c] = ("●" if r == mean_r else "│", cp_mean if r == mean_r else cp_range)

        return grid, y_min, y_max

    # ── Dense path: braille 2×4 sub-cell resolution ───────────────────────
    data_h = 4 * height  # braille sub-rows per terminal row
    data_w = 2 * width   # braille sub-columns per terminal column

    def _data_row(e: float) -> int:
        frac = (e - y_min) / (y_max - y_min)
        return max(0, min(data_h - 1, round((1.0 - frac) * (data_h - 1))))

    # One independent bit grid per layer.  Each cell accumulates only that
    # layer's dots; the highest-priority layer with any bits wins the cell
    # outright — its dots are rendered alone, with no mixing from lower layers.
    mean_bits:  list[list[int]] = [[0] * width for _ in range(height)]
    range_bits: list[list[int]] = [[0] * width for _ in range(height)]

    def _dot(grid2d: list[list[int]], dc: int, dr: int) -> None:
        grid2d[dr // 4][dc // 2] |= _BRAILLE_BITS[dc % 2][dr % 4]

    # Reference line — rendered as a full-width horizontal stroke rather than
    # braille dots, so it looks like a line and is visually distinct from data.
    # Combining overline (U+0305) positions the stroke at the top of the cell;
    # combining low line (U+0332) at the bottom; plain ─ in the middle two
    # sub-rows.  We still track which terminal row the reference occupies so
    # the composition step knows where to draw it.
    ref_row: int | None = None   # terminal row
    ref_ch:  str        = "─"   # character to draw
    if ref_val is not None:
        ref_dr  = _data_row(ref_val)
        ref_row = ref_dr // 4
        ref_ch  = ("─\u0305" if ref_dr % 4 == 0 else
                   "─\u0332" if ref_dr % 4 == 3 else "─")

    # Data bars and mean dots
    for dc in range(data_w):
        lo_it = dc / data_w * x_total
        hi_it = (dc + 1) / data_w * x_total
        vals = [e for x, e in zip(x_positions, ground_u_history) if lo_it <= x < hi_it]
        if not vals:
            continue
        mean_dr = _data_row(float(np.mean(vals)))
        top_dr  = _data_row(max(vals))
        bot_dr  = _data_row(min(vals))
        # Only the topmost and bottommost range dots; mean dot in its own layer.
        _dot(mean_bits, dc, mean_dr)
        if top_dr != mean_dr:
            _dot(range_bits, dc, top_dr)
        if bot_dr != mean_dr:
            _dot(range_bits, dc, bot_dr)

    # Compose: mean wins, then range; reference fills empty cells in its row.
    for r in range(height):
        for c in range(width):
            if mean_bits[r][c]:
                grid[r][c] = (chr(0x2800 + mean_bits[r][c]), cp_mean)
            elif range_bits[r][c]:
                grid[r][c] = (chr(0x2800 + range_bits[r][c]), cp_range)
            elif r == ref_row:
                grid[r][c] = (ref_ch, cp_ref)

    return grid, y_min, y_max


def _fmt_2sf(n: float) -> str:
    """Format n with 2 significant figures, no scientific notation."""
    if n <= 0:
        return "0"
    mag = math.floor(math.log10(n))
    decimals = max(0, 1 - mag)
    return f"{n:.{decimals}f}"


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '1h 23m' or '45m'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _parse_timestamp(s: str) -> float:
    """Parse an OpenMM-format timestamp string to Unix epoch seconds."""
    from datetime import datetime
    return datetime.strptime(s.strip(), "%a %b %d %H:%M:%S %Y").timestamp()



def _parse_atom_selection(s: str) -> list[int]:
    """Parse a selection string like '0-64,67,69,200-300' into sorted unique indices."""
    indices: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            indices.add(int(part))
    if not indices:
        raise ValueError("empty selection")
    return sorted(indices)


def _kabsch_rmsd(P: np.ndarray[Any, np.dtype[Any]], Q: np.ndarray[Any, np.dtype[Any]]) -> float:
    """RMSD of P relative to Q after optimal superposition. Both (N, 3) in nm."""
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum((P_c @ R.T - Q_c) ** 2, axis=-1))))


# ── Data-source abstraction ────────────────────────────────────────────────────


class SimulationReader(Protocol):
    """Abstract interface for reading HREX/REMD simulation data.

    Implement this class for each trajectory format (openmmtools netCDF4,
    GROMACS, PLUMED, …).  The monitor's poll/render pipeline calls only these
    methods, with no format-specific logic.
    """

    # ── Static properties (available immediately after construction) ───────

    @property
    def n_replicas(self) -> int:
        """Number of replicas."""
        ...

    @property
    def n_states(self) -> int:
        """Number of thermodynamic states."""
        ...

    @property
    def n_iterations(self) -> int | None:
        """Total planned iterations, or None if unknown."""
        ...

    @property
    def steps_per_iter(self) -> int:
        """MD steps between exchange attempts."""
        ...

    @property
    def timestep_ps(self) -> float:
        """MD timestep in picoseconds."""
        ...

    @property
    def ref_temp_k(self) -> float:
        """Reference (lowest) temperature in Kelvin."""
        ...

    @property
    def pos_interval(self) -> int:
        """Exchange iterations between position/volume writes; 0 if unavailable."""
        ...

    @property
    def vel_interval(self) -> int:
        """Exchange iterations between velocity writes; 0 if unavailable."""
        ...

    @property
    def has_t_kin(self) -> bool:
        """Whether kinetic temperatures can be computed or read."""
        ...

    @property
    def has_volume(self) -> bool:
        """Whether box volumes are recorded."""
        ...

    @property
    def has_positions(self) -> bool:
        """Whether atomic positions are recorded (needed for RMSD modes)."""
        ...

    # ── File management ────────────────────────────────────────────────────


    def refresh(self) -> None:
        """Reopen / re-read underlying files to expose the latest written data.

        Must be called before any data-access method to guarantee freshness.
        Raises OSError if the file cannot be opened.
        """
        ...

    def close(self) -> None:
        """Release all file handles."""
        ...

    # ── Iteration bookkeeping ──────────────────────────────────────────────


    def last_iteration(self) -> int:
        """Index of the last fully-written iteration, or -1 if not yet started."""
        ...

    # ── Per-iteration data (single frame) ─────────────────────────────────


    def replica_states(self, iteration: int) -> np.ndarray[Any, np.dtype[Any]]:
        """(n_replicas,) int: thermodynamic state of each replica at *iteration*."""
        ...

    def energies(self, iteration: int) -> np.ndarray[Any, np.dtype[Any]]:
        """(n_replicas, n_states) float: reduced potential u_rs = U(x_r) / kT_s."""
        ...

    def t_kin(self, iteration: int, replica: int) -> float | None:
        """Kinetic temperature in K for *replica* at *iteration*, or None."""
        ...

    def volume(self, iteration: int, replica: int) -> float | None:
        """Box volume in nm³ for *replica* at *iteration*, or None."""
        ...

    def positions(
        self,
        iteration: int,
        replica: int,
        atom_indices: list[int],
    ) -> np.ndarray[Any, np.dtype[Any]] | None:
        """(len(atom_indices), 3) positions in nm, or None if unavailable."""
        ...

    def timestamp(self, iteration: int) -> float | None:
        """Unix wall-clock time for *iteration*, or None if unavailable."""
        ...

    # ── Range reads (for efficient bulk accumulation) ─────────────────────


    def exchange_counts(
        self, iter_slice: slice
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """(accepted, proposed) arrays of shape (n_states, n_states) summed
        over *iter_slice*."""
        ...

    def states_range(self, iter_slice: slice) -> np.ndarray[Any, np.dtype[Any]]:
        """(n_iters, n_replicas) int: state assignments over *iter_slice*."""
        ...

    def energies_range(self, iter_slice: slice) -> np.ndarray[Any, np.dtype[Any]]:
        """(n_iters, n_replicas, n_states) float: reduced potentials over *iter_slice*."""
        ...

    def volumes_range(self, iter_slice: slice) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
        """(n_iters, n_replicas) masked float: volumes in nm³ over *iter_slice*."""
        ...

    # ── Optional / derived quantities ─────────────────────────────────────


    def free_energy_history(
        self, max_iter: int
    ) -> list[tuple[int, float]] | None:
        """[(iteration, ΔF_kT)] total free-energy estimates up to *max_iter*.

        Returns the cumulative ΔF from state 0 to state n_states-1, sampled
        at checkpoint intervals.  Returns None if unavailable.
        """
        ...


class OpenmmToolsReader:
    """SimulationReader for openmmtools HREX netCDF4 files."""

    def __init__(self, storage: Path) -> None:
        self._storage = Path(storage)
        self._nc = netCDF4.Dataset(str(self._storage), "r")
        self._read_static()

    def _read_static(self) -> None:
        """Extract static metadata from the open nc handle."""
        energies_var = self._nc.variables["energies"]
        assert energies_var.ndim == 3, (
            f"Expected energies to have 3 dimensions, got {energies_var.ndim}"
        )
        self._n_replicas = energies_var.shape[1]
        self._n_states   = energies_var.shape[2]

        analysis_particle_indices = tuple(
            int(i) for i in self._nc.variables["analysis_particle_indices"][:]
        )
        has_ap = len(analysis_particle_indices) > 0

        all_masses, all_constraints, has_cm_remover = _load_masses_from_nc(self._nc)
        kinetic_set = (
            set(analysis_particle_indices) if has_ap else set(range(len(all_masses)))
        )
        self._kinetic_masses = all_masses[sorted(kinetic_set)]
        n_kp = len(self._kinetic_masses)
        n_con = sum(
            1 for p1, p2 in all_constraints
            if p1 in kinetic_set and p2 in kinetic_set
        )
        self._n_dof = 3 * n_kp - n_con - (3 if has_cm_remover else 0)

        move_doc = _read_mcmc_move(self._nc)
        assert "n_steps" in move_doc and "integrator" in move_doc
        integrator_root = ET.fromstring(move_doc["integrator"])
        assert "stepSize" in integrator_root.attrib
        self._timestep_ps   = float(integrator_root.attrib["stepSize"])
        self._steps_per_iter = int(move_doc["n_steps"])

        thermo_doc  = _read_thermo_state(self._nc)
        temp_entry  = thermo_doc["temperature"]
        assert isinstance(temp_entry, dict) and "value" in temp_entry
        self._ref_temp_k = float(temp_entry["value"])

        self._n_iterations: int | None = None
        try:
            opts = yaml.safe_load(str(self._nc.variables["options"][0]))
            self._n_iterations = int(opts["number_of_iterations"])
        except Exception:
            pass

        self._pos_interval = 500
        try:
            self._pos_interval = int(self._nc.PositionInterval)
        except AttributeError:
            pass

        self._vel_interval = 500
        try:
            self._vel_interval = int(self._nc.VelocityInterval)
        except AttributeError:
            pass

        self._has_t_kin     = has_ap and "velocities" in self._nc.variables
        self._has_volume    = "volumes"    in self._nc.variables
        self._has_positions = "positions"  in self._nc.variables

    # ── Static properties ──────────────────────────────────────────────────

    @property
    def n_replicas(self) -> int:       return self._n_replicas
    @property
    def n_states(self) -> int:         return self._n_states
    @property
    def n_iterations(self) -> int | None: return self._n_iterations
    @property
    def steps_per_iter(self) -> int:   return self._steps_per_iter
    @property
    def timestep_ps(self) -> float:    return self._timestep_ps
    @property
    def ref_temp_k(self) -> float:     return self._ref_temp_k
    @property
    def pos_interval(self) -> int:     return self._pos_interval
    @property
    def vel_interval(self) -> int:     return self._vel_interval
    @property
    def has_t_kin(self) -> bool:       return self._has_t_kin
    @property
    def has_volume(self) -> bool:      return self._has_volume
    @property
    def has_positions(self) -> bool:   return self._has_positions

    # ── File management ────────────────────────────────────────────────────

    def refresh(self) -> None:
        try:
            self._nc.close()
        except RuntimeError:
            pass
        self._nc = netCDF4.Dataset(str(self._storage), "r")

    def close(self) -> None:
        try:
            self._nc.close()
        except RuntimeError:
            pass

    # ── Iteration bookkeeping ──────────────────────────────────────────────

    def last_iteration(self) -> int:
        try:
            last = int(self._nc.variables["last_iteration"][0])
            if last == -1 or self._nc.variables["energies"].shape[0] <= 1:
                return -1
            return last
        except (OSError, IndexError, RuntimeError):
            return -1

    # ── Per-iteration data ─────────────────────────────────────────────────

    def replica_states(self, iteration: int) -> np.ndarray[Any, np.dtype[Any]]:
        return self._nc.variables["states"][iteration, :].astype(int)  # type: ignore[union-attr]

    def energies(self, iteration: int) -> np.ndarray[Any, np.dtype[Any]]:
        return np.array(self._nc.variables["energies"][iteration, :, :])  # type: ignore[union-attr]

    def t_kin(self, iteration: int, replica: int) -> float | None:
        if not self._has_t_kin:
            return None
        try:
            v  = np.array(self._nc.variables["velocities"][iteration, replica, :, :])  # type: ignore[union-attr]
            ke = _ke_from_velocities(v, self._kinetic_masses)
            t  = 2.0 * ke / (self._n_dof * KB_KJMOL_PER_K)
            return t if math.isfinite(t) and t > 0 else None
        except (OSError, IndexError, RuntimeError):
            return None

    def volume(self, iteration: int, replica: int) -> float | None:
        if not self._has_volume:
            return None
        try:
            v = self._nc.variables["volumes"][iteration, replica]  # type: ignore[union-attr]
            if np.ma.is_masked(v):
                return None
            val = float(v)
            return val if math.isfinite(val) and val > 0 else None
        except (OSError, IndexError, RuntimeError):
            return None

    def positions(
        self,
        iteration: int,
        replica: int,
        atom_indices: list[int],
    ) -> np.ndarray[Any, np.dtype[Any]] | None:
        if not self._has_positions:
            return None
        try:
            pos = np.array(
                self._nc.variables["positions"][iteration, replica, :, :]  # type: ignore[union-attr]
            )[atom_indices, :]
            return pos if np.isfinite(pos).all() else None
        except (OSError, IndexError, RuntimeError):
            return None

    def timestamp(self, iteration: int) -> float | None:
        try:
            return _parse_timestamp(str(self._nc.variables["timestamp"][iteration]))  # type: ignore[union-attr]
        except Exception:
            return None

    # ── Range reads ────────────────────────────────────────────────────────

    def exchange_counts(
        self, iter_slice: slice
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        accepted = np.sum(self._nc.variables["accepted"][iter_slice], axis=0)  # type: ignore[union-attr]
        proposed = np.sum(self._nc.variables["proposed"][iter_slice], axis=0)  # type: ignore[union-attr]
        return accepted, proposed

    def states_range(self, iter_slice: slice) -> np.ndarray[Any, np.dtype[Any]]:
        return np.array(self._nc.variables["states"][iter_slice, :]).astype(int)  # type: ignore[union-attr]

    def energies_range(self, iter_slice: slice) -> np.ndarray[Any, np.dtype[Any]]:
        return np.array(self._nc.variables["energies"][iter_slice, :, :])  # type: ignore[union-attr]

    def volumes_range(self, iter_slice: slice) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
        return self._nc.variables["volumes"][iter_slice, :]  # type: ignore[union-attr]

    # ── Free energy ────────────────────────────────────────────────────────

    def free_energy_history(
        self, max_iter: int
    ) -> list[tuple[int, float]] | None:
        # Use f_k_offline_history[:, n_states-1] — the cumulative ΔF from
        # state 0 to state n_states-1, written at checkpoint intervals.
        #
        # DO NOT use free_energy_history[:, 0].  That variable interleaves
        # two different quantities in the same array:
        #   - Every iteration:            f_k[1] only (~4→83 kT for 8 states)
        #   - Every checkpoint interval:  total MBAR ΔF (~199 kT for 8 states)
        # The checkpoint values (~199 kT) appear as the max in every braille
        # sparkline bin (one checkpoint per ~455 iters), producing a spurious
        # flat range line across the top of the entire sparkline.  The per-
        # iteration f_k[1] values are NOT the total free energy and should not
        # be plotted as such.
        #
        # f_k_offline_history is written only at checkpoint intervals (~250
        # points for a 50k-iteration run) and stores the converging cumulative
        # ΔF for every state, making it the correct source for this plot.
        try:
            fe_ma = self._nc.groups["online_analysis"].variables[
                "f_k_offline_history"
            ][:max_iter + 1, self._n_states - 1]
            return [
                (i, float(fe_ma[i]))
                for i, v in enumerate(fe_ma)
                if not np.ma.is_masked(v)
            ]
        except (KeyError, OSError, IndexError, RuntimeError):
            return None


# ── State initialisation and event loop ───────────────────────────────────────


def _init_state(
    reader: SimulationReader,
    n_iterations_arg: int | None,
    init_atom_sel: list[int] | None,
    interval: float,
) -> types.SimpleNamespace:
    """Build initial mutable state namespace from a SimulationReader."""
    n_replicas   = reader.n_replicas
    n_states     = reader.n_states
    n_iterations = n_iterations_arg if n_iterations_arg is not None else reader.n_iterations

    return types.SimpleNamespace(
        # Derived from reader static properties
        interval=interval,
        n_replicas=n_replicas,
        n_states=n_states,
        n_steps=reader.steps_per_iter,
        timestep_ps=reader.timestep_ps,
        kt_kjmol=reader.ref_temp_k * KB_KJMOL_PER_K,
        n_iterations=n_iterations,
        has_t_kin=reader.has_t_kin,
        has_volume=reader.has_volume,
        pos_interval=reader.pos_interval,
        vel_interval=reader.vel_interval,
        # Accumulated exchange/energy state
        prev_iter=-2,
        acc_sum=np.zeros((n_states, n_states)),
        prop_sum=np.zeros((n_states, n_states)),
        state_counts=np.zeros((n_replicas, n_states), dtype=int),
        last_extreme=[None] * n_replicas,
        half_trips=np.zeros(n_replicas, dtype=int),
        last_mixed_iter=-1,
        ground_u_history=[],
        ground_u_iters=[],
        ground_vol_history=[],
        ground_vol_iters=[],
        # T_kin / Volume cache
        cached_t_kin=["—"] * n_replicas,
        cached_vol=["—"] * n_replicas,
        cached_t_kin_at=[-1] * n_replicas,
        cached_vol_at=[-1] * n_replicas,
        # Position / RMSD state
        solute_atom_sel=init_atom_sel,
        solute_sel_str=(f"0-{init_atom_sel[-1]}" if init_atom_sel is not None else ""),
        pos_frame_iters=[],
        pos_all_frames=[],
        pairwise_rmsd_mat=[],
        n_pos_frames=0,
        pos_scan_iter=0,      # next iteration index to attempt for RMSD accumulation
        history_computing=False,  # True while _history_scan_gen task is in the runner
        rmsd_computing=False,     # True while _rmsd_gen task is in the runner
        # UI state
        sparkline_mode=0,
        show_spark_help=False,
        rmsd_input_active=False,
        rmsd_input_buf="",
        # Scrub state
        scrub_iter=None,          # None = live; int = pinned display iteration
        scrub_dirty=False,        # True when scrub_iter changed and data not yet read
        scrub_states=None,        # replica_states at scrub_iter
        scrub_energies=None,      # energies at scrub_iter
        scrub_state_counts=None,  # (n_replicas, n_states) counts up to scrub_iter
        scrub_half_trips=None,    # (n_replicas,) half-trips up to scrub_iter
        scrub_t_kin=None,         # per-replica T_kin strings at scrub_iter
        scrub_t_kin_at=None,      # per-replica 1-based iter T_kin was read from
        scrub_vol=None,           # per-replica volume strings at scrub_iter
        scrub_vol_at=None,        # per-replica 1-based iter volume was read from
        scrub_checkpoints=[],     # [(last_iter, state_counts, half_trips, last_extreme), …]
        scrub_input_active=False, # True while typing a jump iteration
        scrub_input_buf="",
        # Current render state (populated by _poll)
        ever_polled=False,  # True after the first _poll attempt, regardless of result
        waiting=True,
        display_iter=0,
        replica_states=None,
        energies=None,
        sim_str="",
        iter_str="",
        timing_str="",
        error_text="",
        fe_iters=[],
        fe_vals=[],
    )


def _handle_key(key: int, S: types.SimpleNamespace) -> tuple[bool, bool]:
    """Handle a keypress. Returns (needs_redraw, should_quit)."""
    if key == -1:
        return False, False
    if key == ord("q"):
        return False, True  # always quit regardless of mode

    # ── Jump-to-iteration input mode ───────────────────────────────────────
    if S.scrub_input_active:
        if key in (ord("\n"), ord("\r"), curses.KEY_ENTER):
            try:
                n = int(S.scrub_input_buf)
                # Positive: 1-based iteration number.
                # Negative: offset from end (-1 = last, -2 = second-to-last, …)
                if n >= 0:
                    target = n - 1
                else:
                    target = S.display_iter + 1 + n
                target = max(0, min(S.display_iter, target))
                S.scrub_iter  = target
                S.scrub_dirty = True
            except ValueError:
                pass
            S.scrub_input_active = False
            S.scrub_input_buf    = ""
            return True, False
        elif key == 27:  # Esc cancels
            S.scrub_input_active = False
            S.scrub_input_buf    = ""
            return True, False
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            S.scrub_input_buf = S.scrub_input_buf[:-1]
            return True, False
        elif 0 <= key <= 0x10FFFF and (
            chr(key).isdigit()
            or (chr(key) == "-" and not S.scrub_input_buf)
        ):
            S.scrub_input_buf += chr(key)
            return True, False
        return False, False

    if S.rmsd_input_active:
        if key in (ord("\n"), ord("\r"), curses.KEY_ENTER):
            try:
                S.solute_atom_sel = _parse_atom_selection(S.rmsd_input_buf)
                S.solute_sel_str = S.rmsd_input_buf
                S.pos_frame_iters  = []
                S.pos_all_frames   = []
                S.pairwise_rmsd_mat = []
                S.n_pos_frames     = 0
                S.pos_scan_iter    = 0
                S.rmsd_computing   = False  # old task invalidated; runner drains naturally
            except ValueError:
                pass
            S.rmsd_input_active = False
            S.rmsd_input_buf = ""
            return True, False
        elif key == 27:  # Esc cancels
            S.rmsd_input_active = False
            S.rmsd_input_buf = ""
            return True, False
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            S.rmsd_input_buf = S.rmsd_input_buf[:-1]
            return True, False
        elif 0 <= key <= 0x10FFFF and chr(key) in "0123456789-,":
            S.rmsd_input_buf += chr(key)
            return True, False
        return False, False
    else:
        if key == ord("q"):
            return False, True
        elif key == ord("s"):
            S.sparkline_mode = (S.sparkline_mode + 1) % 7
            return True, False
        elif key == ord("S"):
            S.sparkline_mode = (S.sparkline_mode - 1) % 7
            return True, False
        elif key == ord("?"):
            S.show_spark_help = not S.show_spark_help
            return True, False
        elif key == ord("a") and S.sparkline_mode in (3, 4, 5, 6):
            S.rmsd_input_active = True
            S.rmsd_input_buf = S.solute_sel_str
            return True, False
        elif key == ord("j") and not S.waiting and S.replica_states is not None:
            S.scrub_input_active = True
            S.scrub_input_buf    = ""
            return True, False
        elif key == ord("z") and not S.waiting and S.replica_states is not None:
            if S.scrub_iter is not None:
                # Go live — unpin
                S.scrub_iter     = None
                S.scrub_dirty    = False
                S.scrub_states   = None
                S.scrub_energies = None
            else:
                # Freeze — pin to the current live iteration
                S.scrub_iter  = S.display_iter
                S.scrub_dirty = True
            return True, False
        elif key in (ord("w"), ord("e"), ord("W"), ord("E")) and not S.waiting and S.replica_states is not None:
            total     = S.display_iter + 1
            fast_step = max(1, total // 20)
            current   = S.scrub_iter if S.scrub_iter is not None else S.display_iter
            if key == ord("w"):
                new_iter = max(0, current - 1)
            elif key == ord("e"):
                new_iter = min(S.display_iter, current + 1)
            elif key == ord("W"):
                new_iter = max(0, current - fast_step)
            else:  # ord("E")
                new_iter = min(S.display_iter, current + fast_step)
            if new_iter >= S.display_iter:
                # Reached or passed live — unpin
                S.scrub_iter     = None
                S.scrub_dirty    = False
                S.scrub_states   = None
                S.scrub_energies = None
            else:
                S.scrub_iter  = new_iter
                S.scrub_dirty = True
            return True, False
        return False, False


class TaskRunner:
    """Cooperative scheduler for generator-based incremental tasks.

    Each task is a generator that yields ``None`` after each atomic unit of
    work.  ``run_until`` drives them round-robin until the deadline, giving
    every task a fair share of any remaining budget.
    """

    def __init__(self) -> None:
        self._tasks: list[Generator[None, None, None]] = []

    def submit(self, task: Generator[None, None, None]) -> None:
        """Enqueue a new task."""
        self._tasks.append(task)

    def run_until(self, deadline: float) -> bool:
        """Step tasks round-robin until *deadline* (monotonic seconds).

        Returns True if at least one unit of work was performed (caller should
        schedule a redraw).  Exhausted generators are removed automatically.
        """
        if not self._tasks:
            return False
        worked = False
        while self._tasks and time.monotonic() < deadline:
            task = self._tasks.pop(0)
            try:
                next(task)
                worked = True
                self._tasks.append(task)   # re-queue for round-robin
            except StopIteration:
                pass                        # task finished — not re-queued
        return worked

    def __bool__(self) -> bool:
        return bool(self._tasks)


_HISTORY_CHUNK = 2000  # iterations per yield in _history_scan_gen


def _history_scan_gen(
    reader: SimulationReader,
    S: types.SimpleNamespace,
) -> Generator[None, None, None]:
    """Incrementally accumulate exchange / energy / volume history.

    Processes iterations in chunks of ``_HISTORY_CHUNK``, yielding after each
    so the main loop can render progress and check the deadline.  Advances
    ``S.last_mixed_iter`` as it goes so ``_poll`` sees consistent state.
    Clears ``S.history_computing`` on exit.
    """
    t_scan_start = time.perf_counter()
    n_chunks = 0
    n_tkin_reads = 0
    _LOG.debug(
        "history scan start: display_iter=%d last_mixed=%d chunk_size=%d",
        S.display_iter, S.last_mixed_iter, _HISTORY_CHUNK,
    )
    try:
        while S.last_mixed_iter < S.display_iter:
            chunk_start = S.last_mixed_iter + 1
            chunk_end   = min(chunk_start + _HISTORY_CHUNK, S.display_iter + 1)
            chunk       = slice(chunk_start, chunk_end)
            t0 = time.perf_counter()
            try:
                t1 = time.perf_counter()
                acc_delta, prop_delta = reader.exchange_counts(chunk)
                S.acc_sum  += acc_delta
                S.prop_sum += prop_delta
                t2 = time.perf_counter()
                new_states_arr   = reader.states_range(chunk)
                t3 = time.perf_counter()
                new_energies_arr = reader.energies_range(chunk)
                t4 = time.perf_counter()
                new_volumes_ma   = reader.volumes_range(chunk) if S.has_volume else None
                t5 = time.perf_counter()
                _LOG.debug(
                    "chunk %d–%d (n=%d): exchange=%.1fms states=%.1fms energies=%.1fms volumes=%.1fms",
                    chunk_start, chunk_end - 1, chunk_end - chunk_start,
                    (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3,
                )
                for r in range(S.n_replicas):
                    S.state_counts[r] += np.bincount(
                        new_states_arr[:, r], minlength=S.n_states
                    )
                abs_iters_arr = np.arange(chunk_start, chunk_end)
                n_states_m1   = S.n_states - 1

                # Volume cache: last valid value per replica in this chunk.
                if new_volumes_ma is not None:
                    for r in range(S.n_replicas):
                        col   = new_volumes_ma[:, r]
                        valid = ~np.ma.getmaskarray(col) & np.isfinite(col.data) & (col.data > 0)
                        if valid.any():
                            last_idx           = int(np.where(valid)[0][-1])
                            S.cached_vol[r]    = f"{float(col.data[last_idx]):.2f}"
                            S.cached_vol_at[r] = chunk_start + last_idx + 1

                # T_kin: read velocity arrays at vel_interval boundaries in chunk.
                if S.has_t_kin and S.vel_interval > 0:
                    first_bdry = (
                        (chunk_start + S.vel_interval - 1) // S.vel_interval
                    ) * S.vel_interval
                    for tkin_iter in range(first_bdry, chunk_end, S.vel_interval):
                        t_tkin = time.perf_counter()
                        for r in range(S.n_replicas):
                            t = reader.t_kin(tkin_iter, r)
                            if t is not None:
                                S.cached_t_kin[r]    = f"{t:.1f}"
                                S.cached_t_kin_at[r] = tkin_iter + 1
                        n_tkin_reads += 1
                        _LOG.debug(
                            "  t_kin iter %d: %.1fms (%d replicas)",
                            tkin_iter, (time.perf_counter() - t_tkin) * 1e3, S.n_replicas,
                        )

                # Half-trips and ground-state sparkline data: vectorised per replica.
                for r in range(S.n_replicas):
                    states_r = new_states_arr[:, r]

                    # Extreme-state transitions → half-trips.
                    extreme_idxs = np.where((states_r == 0) | (states_r == n_states_m1))[0]
                    if extreme_idxs.size > 0:
                        ext_states = states_r[extreme_idxs]
                        if S.last_extreme[r] is not None:
                            chain    = np.empty(ext_states.size + 1, dtype=ext_states.dtype)
                            chain[0] = S.last_extreme[r]
                            chain[1:] = ext_states
                        else:
                            chain = ext_states
                        S.half_trips[r]   += int(np.count_nonzero(np.diff(chain)))
                        S.last_extreme[r]  = int(ext_states[-1])

                    # Ground-state rows: batch-append energies and volumes.
                    gs_idxs = np.where(states_r == 0)[0]
                    if gs_idxs.size > 0:
                        gs_abs = abs_iters_arr[gs_idxs]
                        gs_u   = new_energies_arr[gs_idxs, r, 0] * S.kt_kjmol
                        S.ground_u_history.extend(gs_u.tolist())
                        S.ground_u_iters.extend(gs_abs.tolist())
                        if new_volumes_ma is not None:
                            gs_vols = np.ma.filled(new_volumes_ma[gs_idxs, r], np.nan)
                            valid_v = np.isfinite(gs_vols) & (gs_vols > 0)
                            if valid_v.any():
                                S.ground_vol_history.extend(gs_vols[valid_v].tolist())
                                S.ground_vol_iters.extend(gs_abs[valid_v].tolist())
                S.last_mixed_iter = chunk_end - 1
                S.scrub_checkpoints.append((
                    S.last_mixed_iter,
                    S.state_counts.copy(),
                    S.half_trips.copy(),
                    list(S.last_extreme),
                ))
                n_chunks += 1
                _LOG.debug(
                    "chunk total: %.1fms  (last_mixed=%d)",
                    (time.perf_counter() - t0) * 1e3, S.last_mixed_iter,
                )
            except (OSError, IndexError, RuntimeError):
                _LOG.exception("history scan error at chunk %d–%d", chunk_start, chunk_end - 1)
                break
            yield
    finally:
        elapsed = time.perf_counter() - t_scan_start
        _LOG.info(
            "history scan done: %d iters in %d chunks, %d t_kin reads, total %.3fs",
            S.last_mixed_iter + 1, n_chunks, n_tkin_reads, elapsed,
        )
        S.history_computing = False


def _rmsd_gen(
    reader: SimulationReader,
    S: types.SimpleNamespace,
) -> Generator[None, None, None]:
    """Incrementally load position frames and compute pairwise RMSD.

    Yields after each frame so ``TaskRunner`` can interleave with other tasks
    and respect the loop deadline.  Clears ``S.rmsd_computing`` on exit.
    """
    try:
        while (
            not S.waiting
            and S.solute_atom_sel is not None
            and S.pos_interval > 0
            and S.pos_scan_iter <= S.display_iter
        ):
            abs_i = S.pos_scan_iter
            S.pos_scan_iter += S.pos_interval   # advance before any early-out
            try:
                step  = reader.replica_states(abs_i)
                r_pos = int(np.where(step == 0)[0][0])
                pos   = reader.positions(abs_i, r_pos, S.solute_atom_sel)
                if pos is not None:
                    new_row = [
                        _kabsch_rmsd(pos, S.pos_all_frames[j]) * 10.0
                        for j in range(len(S.pos_all_frames))
                    ]
                    S.pairwise_rmsd_mat.append(new_row)
                    S.pos_all_frames.append(pos)
                    S.pos_frame_iters.append(abs_i)
                    S.n_pos_frames = len(S.pos_all_frames)
            except (OSError, IndexError, RuntimeError, np.linalg.LinAlgError):
                pass
            yield
    finally:
        S.rmsd_computing = False



def _fetch_scrub_data(reader: SimulationReader, S: types.SimpleNamespace) -> None:
    """Read/compute all per-iteration data for the pinned scrub position."""
    import bisect
    si         = S.scrub_iter
    n_replicas = S.n_replicas
    n_states   = S.n_states

    S.scrub_states   = reader.replica_states(si)
    S.scrub_energies = reader.energies(si)

    # State counts and half-trips: start from nearest chunk checkpoint then
    # apply the residual (at most one chunk worth) of raw states from the NC.
    checkpoints = S.scrub_checkpoints
    if checkpoints:
        idx = bisect.bisect_right([c[0] for c in checkpoints], si) - 1
    else:
        idx = -1

    if idx >= 0:
        _, state_counts, half_trips_arr, last_extreme = checkpoints[idx]
        state_counts = state_counts.copy()
        half_trips   = list(half_trips_arr)
        last_extreme = list(last_extreme)
        read_from    = checkpoints[idx][0] + 1
    else:
        state_counts = np.zeros((n_replicas, n_states), dtype=int)
        half_trips   = [0] * n_replicas
        last_extreme = [None] * n_replicas
        read_from    = 0

    if read_from <= si:
        residual    = reader.states_range(slice(read_from, si + 1))
        n_states_m1 = n_states - 1
        for r in range(n_replicas):
            col = residual[:, r]
            state_counts[r] += np.bincount(col, minlength=n_states)
            ext_idxs = np.where((col == 0) | (col == n_states_m1))[0]
            if ext_idxs.size > 0:
                ext_s = col[ext_idxs]
                chain = (np.concatenate([[last_extreme[r]], ext_s])
                         if last_extreme[r] is not None else ext_s)
                half_trips[r] += int(np.count_nonzero(np.diff(chain)))

    S.scrub_state_counts = state_counts
    S.scrub_half_trips   = np.array(half_trips, dtype=int)

    # T_kin at scrub position (nearest vel_interval boundary)
    scrub_t_kin    = ["—"] * n_replicas
    scrub_t_kin_at = [si + 1] * n_replicas  # default: current frame (no asterisk)
    if S.has_t_kin:
        for r in range(n_replicas):
            t_at = si
            t    = reader.t_kin(si, r)
            if t is None and S.vel_interval > 0:
                t_at = (si // S.vel_interval) * S.vel_interval
                t    = reader.t_kin(t_at, r)
            if t is not None:
                scrub_t_kin[r]    = f"{t:.1f}"
                scrub_t_kin_at[r] = t_at + 1
    S.scrub_t_kin    = scrub_t_kin
    S.scrub_t_kin_at = scrub_t_kin_at

    # Volume at nearest pos_interval boundary (volume not written every iteration)
    scrub_vol    = ["—"] * n_replicas
    scrub_vol_at = [si + 1] * n_replicas  # default: current frame (no asterisk)
    if S.has_volume and S.pos_interval > 0:
        v_at = (si // S.pos_interval) * S.pos_interval
        for r in range(n_replicas):
            v = reader.volume(v_at, r)
            if v is not None:
                scrub_vol[r]    = f"{v:.2f}"
                scrub_vol_at[r] = v_at + 1
    S.scrub_vol    = scrub_vol
    S.scrub_vol_at = scrub_vol_at


def _poll(reader: SimulationReader, S: types.SimpleNamespace, stdscr: curses.window) -> bool:
    """Read fresh data via *reader* and update S. Returns True if a redraw is needed."""
    last_iter     = reader.last_iteration()
    S.ever_polled = True

    if last_iter == -1:
        S.waiting = True
        return True  # always redraw so the "no data" spinner animates

    S.waiting = False

    if last_iter == S.prev_iter:
        return False  # no new data

    S.prev_iter = last_iter
    if last_iter == 0:
        return False

    display_iter = last_iter - 1
    S.display_iter = display_iter

    # History accumulation (exchange counts, ground-state energies/volumes,
    # half-trips) is handled incrementally by _history_scan_gen so that a large
    # finished simulation doesn't block the UI on the first poll.
    try:
        S.replica_states = reader.replica_states(display_iter)
        S.energies       = reader.energies(display_iter)
    except (OSError, IndexError, RuntimeError):
        S.prev_iter = last_iter - 1  # retry this iteration
        return False
    except Exception:
        import traceback
        S.error_text = traceback.format_exc()
        return True

    S.error_text = ""

    # Format timing strings
    sim_ps = (display_iter + 1) * S.n_steps * S.timestep_ps
    S.sim_str = f"{sim_ps / 1000:.3f} ns" if sim_ps >= 1000 else f"{sim_ps:.1f} ps"
    if S.n_iterations:
        total_sim_ps  = S.n_iterations * S.n_steps * S.timestep_ps
        total_sim_str = (
            f"{total_sim_ps / 1000:.1f} ns" if total_sim_ps >= 1000
            else f"{total_sim_ps:.1f} ps"
        )
        S.sim_str = f"{S.sim_str} / {total_sim_str}"
    S.iter_str = f"{display_iter + 1}" + (f" / {S.n_iterations}" if S.n_iterations else "")

    S.timing_str = ""
    try:
        ts0   = reader.timestamp(0)
        tsnow = reader.timestamp(display_iter)
        if ts0 is not None and tsnow is not None:
            elapsed_s = tsnow - ts0
            S.timing_str = f"    Elapsed: {_fmt_duration(elapsed_s)}"
            if S.n_iterations and display_iter > 0:
                window       = max(50, display_iter // 20)
                window_start = max(0, display_iter - window)
                ts_window    = reader.timestamp(window_start)
                if ts_window is not None:
                    window_elapsed = tsnow - ts_window
                    window_iters   = display_iter - window_start
                    rate = (
                        window_iters / window_elapsed if window_elapsed > 0
                        else display_iter / elapsed_s
                    )
                    remaining_s = (S.n_iterations - display_iter - 1) / rate
                    eta = datetime.fromtimestamp(tsnow + remaining_s)
                    S.timing_str += (
                        f"    ETA: {eta.strftime('%Y-%m-%d %H:%M')}"
                        f" (~{_fmt_duration(remaining_s)} remaining)"
                    )
    except Exception:
        pass

    # Per-replica T_kin and Volume at display_iter — only when history is fully
    # loaded so we don't jump to the final value while the scan is in progress.
    # Only read T_kin / Volume once history has fully caught up — using the data
    # state (last_mixed_iter) rather than the task flag so we don't race with
    # the task submission that happens after _poll in the main loop.
    if S.last_mixed_iter >= display_iter:
        for r in range(S.n_replicas):
            if S.has_t_kin:
                t_iter = display_iter
                t = reader.t_kin(t_iter, r)
                if t is None and S.vel_interval > 0:
                    # display_iter may not be a velocity-write boundary; try nearest one
                    t_iter = (display_iter // S.vel_interval) * S.vel_interval
                    t = reader.t_kin(t_iter, r)
                if t is not None:
                    S.cached_t_kin[r]    = f"{t:.1f}"
                    S.cached_t_kin_at[r] = t_iter + 1
            if S.has_volume:
                v = reader.volume(display_iter, r)
                if v is not None:
                    S.cached_vol[r]    = f"{v:.2f}"
                    S.cached_vol_at[r] = display_iter + 1

    # Online ΔF
    fe_result = reader.free_energy_history(display_iter)
    if fe_result is not None:
        S.fe_iters = [i for i, _ in fe_result]
        S.fe_vals  = [fv for _, fv in fe_result]

    return True


def _render(stdscr: curses.window, S: types.SimpleNamespace, gradient: list[int] | None) -> None:
    """Redraw the entire screen from S. Pure display, no nc access."""
    stdscr.erase()

    if S.error_text:
        _addstr(stdscr, f"Unexpected error:\n{S.error_text}")
        stdscr.refresh()
        return

    if S.waiting:
        wall     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spin_ch  = _SPINNER[int(time.monotonic() * 4) % len(_SPINNER)]
        _addstr(stdscr, f"Wall: {wall}\n\n")
        if not S.ever_polled:
            _addstr(stdscr, f"  {spin_ch}  Reading...", curses.color_pair(_CP_YELLOW))
        else:
            _addstr(stdscr,
                f"  {spin_ch}  No data yet — production may not have started",
                curses.color_pair(_CP_YELLOW))
        stdscr.refresh()
        return

    if S.replica_states is None:
        stdscr.refresh()
        return

    n_replicas   = S.n_replicas
    n_states     = S.n_states
    display_iter = S.display_iter

    # Scrub mode: override which per-iteration data is shown
    _scrubbing        = S.scrub_iter is not None and S.scrub_states is not None
    _render_states    = S.scrub_states       if _scrubbing else S.replica_states
    _render_energies  = S.scrub_energies     if _scrubbing else S.energies
    _render_scounts   = S.scrub_state_counts if _scrubbing else S.state_counts
    _render_htrips    = S.scrub_half_trips   if _scrubbing else S.half_trips
    _render_t_kin     = S.scrub_t_kin        if _scrubbing else S.cached_t_kin
    _render_vol       = S.scrub_vol          if _scrubbing else S.cached_vol

    # While history is loading, show the scan cursor rather than the final iter.
    if S.history_computing and S.last_mixed_iter >= 0:
        _cur = S.last_mixed_iter + 1
        _tot = S.n_iterations or (S.display_iter + 1)
        _iter_str = f"{_cur} / {_tot}"
        _sim_ps   = _cur * S.n_steps * S.timestep_ps
        _sim_str  = f"{_sim_ps / 1000:.3f} ns" if _sim_ps >= 1000 else f"{_sim_ps:.1f} ps"
        if S.n_iterations:
            _total_ps = S.n_iterations * S.n_steps * S.timestep_ps
            _total_str = f"{_total_ps / 1000:.1f} ns" if _total_ps >= 1000 else f"{_total_ps:.1f} ps"
            _sim_str = f"{_sim_str} / {_total_str}"
    elif _scrubbing:
        _si   = S.scrub_iter
        _tot  = S.n_iterations or (display_iter + 1)
        _iter_str = f"[{_si + 1}] / {_tot}"
        _sim_ps   = (_si + 1) * S.n_steps * S.timestep_ps
        _sim_str  = f"[{_sim_ps / 1000:.3f} ns]" if _sim_ps >= 1000 else f"[{_sim_ps:.1f} ps]"
        if S.n_iterations:
            _total_ps  = S.n_iterations * S.n_steps * S.timestep_ps
            _total_str = f"{_total_ps / 1000:.1f} ns" if _total_ps >= 1000 else f"{_total_ps:.1f} ps"
            _sim_str   = f"{_sim_str} / {_total_str}"
    else:
        _iter_str = S.iter_str
        _sim_str  = S.sim_str
    _addstr(stdscr, f"Iteration: {_iter_str}    Time: {_sim_str}{S.timing_str}\n\n")

    col_w    = 7
    matrix_w = 2 + 5 + n_states * col_w
    sep      = "   "

    total_iters = (S.n_iterations - 1) if S.n_iterations else (display_iter + 1)
    _term_rows, _term_cols = stdscr.getmaxyx()
    spark_w = max(10, _term_cols - matrix_w - len(sep) - 1)

    _SPARK_NAMES = ["reduced U", "volume", "online ΔF", "RMSD", "min-RMSD", "max-RMSD", "RMSD ACF"]
    _SPARK_UNITS = ["kJ/mol",   "nm³",    "kT",        "Å",    "Å",        "Å",         "Å"]

    # Derive sparkline data from cached state based on current mode
    mode = S.sparkline_mode
    _acf_max_lag: int = 0  # only used when mode == 6; set below
    if mode == 0:
        sp_iters, sp_vals = S.ground_u_iters, S.ground_u_history
    elif mode == 1:
        sp_iters, sp_vals = S.ground_vol_iters, S.ground_vol_history
    elif mode == 2:
        # online ΔF is stored in S.fe_iters / S.fe_vals by _poll each cycle.
        sp_iters = getattr(S, "fe_iters", [])
        sp_vals  = getattr(S, "fe_vals",  [])
    elif mode == 3:
        if S.n_pos_frames:
            sp_iters = S.pos_frame_iters
            sp_vals  = [0.0] + [S.pairwise_rmsd_mat[i][0] for i in range(1, S.n_pos_frames)]
        else:
            sp_iters, sp_vals = [], []
    elif mode == 4:
        if S.n_pos_frames > 1:
            sp_iters = S.pos_frame_iters[1:]
            sp_vals  = [min(S.pairwise_rmsd_mat[i]) for i in range(1, S.n_pos_frames)]
        else:
            sp_iters, sp_vals = [], []
    elif mode == 5:  # max-RMSD: running maximum pairwise RMSD
        if S.n_pos_frames > 1:
            sp_iters = S.pos_frame_iters[1:]
            sp_vals = [max(S.pairwise_rmsd_mat[i]) for i in range(1, S.n_pos_frames)]
        else:
            sp_iters, sp_vals = [], []
    else:  # mode == 6 — ACF (log x-axis for adaptive resolution)
        if S.n_pos_frames >= 2:
            max_lag = S.n_pos_frames // 2
            # Geometrically-spaced lags: dense where ACF rises, sparse at plateau.
            # 2*spark_w points matches the braille sub-column resolution.
            n_pts = min(max_lag, max(4, 2 * spark_w))
            raw_lags = np.unique(np.round(np.geomspace(1, max_lag, n_pts)).astype(int))
            raw_lags = raw_lags[(raw_lags >= 1) & (raw_lags <= max_lag)]
            sp_vals = []
            for lag in raw_lags:
                pairs = [S.pairwise_rmsd_mat[i + int(lag)][i] for i in range(S.n_pos_frames - int(lag))]
                sp_vals.append(float(np.mean(pairs)))
            # Store raw integer lags; log10 transform is applied inside _build_sparkline_grid
            sp_iters     = [int(lag) for lag in raw_lags]
            _acf_max_lag = int(raw_lags[-1])
        else:
            sp_iters, sp_vals, _acf_max_lag = [], [], 0

    # Reference line value (mode-specific)
    if mode == 0 and sp_vals:
        sp_ref_val: float | None = sp_vals[0]
    elif mode == 1 and sp_vals:
        sp_ref_val = float(np.mean(sp_vals))
    elif mode == 4 and S.n_pos_frames >= 2:
        sp_ref_val = min(v for row in S.pairwise_rmsd_mat for v in row)
    elif mode == 5 and S.n_pos_frames >= 2:
        sp_ref_val = max(v for row in S.pairwise_rmsd_mat for v in row)
    elif mode == 6 and len(sp_vals) >= 4:
        half = len(sp_vals) // 2
        sp_ref_val = float(np.mean(sp_vals[half:]))
    else:
        sp_ref_val = None

    # x-axis labels and grid_total (must precede n_per_unit which uses grid_total)
    if mode == 6 and sp_iters:
        grid_total = _acf_max_lag  # integer max lag; log10 applied inside sparkline
        _frame_ps = S.pos_interval * S.n_steps * S.timestep_ps
        xaxis_l = f"lag {_fmt_2sf(_frame_ps / 1000)}ns (log)"
        xaxis_r = f"lag {_acf_max_lag * _frame_ps / 1000:.1f}ns"
    else:
        grid_total = total_iters
        total_ns = total_iters * S.n_steps * S.timestep_ps / 1000
        xaxis_l = "0 ns"
        xaxis_r = f"{total_ns:.1f} ns"

    # Average data points per rendered dot.  Divide by the number of non-empty
    # bins (dots that actually get drawn), not total possible positions — dots
    # with no data are never shown so they shouldn't count in the denominator.
    n_per_unit: float | None = None
    if mode != 6 and spark_w > 0 and sp_vals and grid_total > 0:
        _use_braille = len(sp_vals) >= spark_w
        _n_bins = 2 * spark_w if _use_braille else spark_w
        _filled = len({min(int(it / grid_total * _n_bins), _n_bins - 1) for it in sp_iters})
        if _filled > 0:
            n_per_unit = len(sp_vals) / _filled

    if mode == 6:
        xaxis_m = f"{S.n_pos_frames} frames" if S.n_pos_frames else ""
    else:
        xaxis_m = f"~{_fmt_2sf(n_per_unit)}/dot" if n_per_unit is not None else ""

    spark_grid, spark_ymin, spark_ymax = _build_sparkline_grid(
        sp_iters, sp_vals,
        grid_total, n_states, spark_w,
        cp_mean=curses.color_pair(_CP_GREEN),
        cp_range=curses.color_pair(_CP_GREY),
        cp_ref=curses.color_pair(_CP_DIM),
        ref_val=sp_ref_val,
        x_transform=math.log10 if mode == 6 else None,
    )

    # Scrub cursor: blue vertical bar only on empty cells (never overwrites data)
    if _scrubbing and mode != 6 and grid_total > 0:
        cur_col  = max(0, min(spark_w - 1, int(S.scrub_iter / grid_total * spark_w)))
        cur_attr = curses.color_pair(_CP_BLUE)
        for r in range(n_states):
            if spark_grid[r][cur_col][0] == " ":
                spark_grid[r][cur_col] = ("│", cur_attr)

    sp_name   = _SPARK_NAMES[mode]
    sp_unit   = _SPARK_UNITS[mode]
    sp_prefix = f"State 0→{n_states-1} {sp_name}" if mode == 2 else f"State 0 {sp_name}"

    if mode in (3, 4, 5, 6) and S.solute_atom_sel is None:
        spark_title = f"{sp_prefix}  (press 'a' to set atom selection)"
    elif mode in (3, 4, 5, 6):
        n_frames_total = display_iter // S.pos_interval + 1
        if S.n_pos_frames < n_frames_total:
            pct = S.n_pos_frames / n_frames_total * 100
            spark_title = (
                f"{sp_prefix}  (computing... {pct:.0f}%"
                f"  {S.n_pos_frames}/{n_frames_total} frames)"
            )
        elif sp_vals:
            spark_title = f"{sp_prefix}  [{spark_ymin:.4g}, {spark_ymax:.4g}] {sp_unit}"
        else:
            spark_title = f"{sp_prefix}  (accumulating...)"
    elif S.history_computing and mode in (0, 1, 2):
        pct = (S.last_mixed_iter + 1) / (display_iter + 1) * 100
        spark_title = f"{sp_prefix}  (loading history... {pct:.0f}%)"
    elif sp_vals:
        spark_title = f"{sp_prefix}  [{spark_ymin:.4g}, {spark_ymax:.4g}] {sp_unit}"
    else:
        spark_title = f"{sp_prefix}  (accumulating...)"

    # Title row
    matrix_hdr = "Exchange acceptance (%):"
    _addstr(stdscr, matrix_hdr, curses.A_BOLD)
    _addstr(stdscr, " " * (matrix_w - len(matrix_hdr)) + sep)
    _addstr(stdscr, spark_title + "\n", curses.A_BOLD)

    # Column header + x-axis (left label | dim centre annotation | right label)
    _addstr(stdscr, f"  {'':>5}" + "".join(f"{i:>{col_w}}" for i in range(n_states)))
    _addstr(stdscr, sep)
    _addstr(stdscr, xaxis_l)
    remaining = spark_w - len(xaxis_l) - len(xaxis_r)
    if xaxis_m and remaining >= len(xaxis_m) + 2:
        lpad = (remaining - len(xaxis_m)) // 2
        rpad = remaining - len(xaxis_m) - lpad
        _addstr(stdscr, " " * lpad)
        _addstr(stdscr, xaxis_m, curses.color_pair(_CP_GREY) | curses.A_DIM)
        _addstr(stdscr, " " * rpad)
    else:
        _addstr(stdscr, " " * max(1, remaining))
    _addstr(stdscr, xaxis_r + "\n")

    # Data rows + sparkline rows (interleaved by state index)
    for i in range(n_states):
        _addstr(stdscr, f"  {i:>5}")
        for j in range(n_states):
            if i == j:
                _addstr(stdscr, " " * col_w)
            else:
                prop = S.prop_sum[i, j]
                rate = S.acc_sum[i, j] / prop * 100 if prop > 0 else float("nan")
                text = f"{'nan':>7}" if math.isnan(rate) else f"{rate:>7.1f}"
                _addstr(stdscr, text, curses.color_pair(_rate_colour_pair(rate, gradient)))
        _addstr(stdscr, sep)
        for col in range(spark_w):
            char, attr = spark_grid[i][col]
            _addstr(stdscr, char, attr)
        _addstr(stdscr, "\n")

    if S.show_spark_help:
        help_text = _SPARK_HELP[mode]
        indent = " " * (matrix_w + len(sep))
        wrap_width = len(indent) + max(20, spark_w)
        wrapped = textwrap.fill(
            help_text, width=wrap_width,
            initial_indent=indent, subsequent_indent=indent,
        )
        _addstr(stdscr, "\n" + wrapped + "\n")

    visits_w = max(n_states, 6)

    # Determine whether to show * in column headers / footnote.
    # * means "value is from a different iteration than the one being displayed".
    _cur_iter_1 = (S.scrub_iter + 1) if _scrubbing else (display_iter + 1)
    if _scrubbing:
        _tkin_at_list = S.scrub_t_kin_at if S.scrub_t_kin_at is not None else []
        _vol_at_list  = S.scrub_vol_at   if S.scrub_vol_at   is not None else []
    else:
        _tkin_at_list = S.cached_t_kin_at
        _vol_at_list  = S.cached_vol_at
    # iters where the value is stale (from a frame other than current)
    _tkin_stale_iters = sorted(set(at for at in _tkin_at_list if at >= 0 and at != _cur_iter_1))
    _vol_stale_iters  = sorted(set(at for at in _vol_at_list  if at >= 0 and at != _cur_iter_1))
    _show_tkin_star   = bool(_tkin_stale_iters)
    _show_vol_star    = bool(_vol_stale_iters)

    _tkin_hdr = "T_kin (K)*" if _show_tkin_star else "T_kin (K) "
    _vol_hdr  = "Volume (nm³)*" if _show_vol_star else "Volume (nm³) "

    _addstr(stdscr, "\n")
    _addstr(stdscr,
        f"  {'Replica':>7}  {'State':>5}  {'Reduced U (kJ/mol)':>18}  "
        f"{_tkin_hdr:>10}  {_vol_hdr:>13}  "
        f"{'Visits':>{visits_w}}  {'Trips':>5}\n"
    )
    _addstr(stdscr,
        f"  {'-------':>7}  {'-----':>5}  {'------------------':>18}  "
        f"{'---------':>10}  {'-------------':>13}  "
        f"{'─' * visits_w}  {'-----':>5}\n"
    )

    _spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _spin_ch  = _spinner[int(time.monotonic() * 4) % len(_spinner)]

    for r in range(n_replicas):
        s = int(_render_states[r])
        reduced_u_kjmol = float(_render_energies[r, s]) * S.kt_kjmol
        _loading = not _scrubbing and S.last_mixed_iter < display_iter
        if _loading and S.cached_t_kin_at[r] == -1:
            t_kin_str = f"{_spin_ch:>10}"
        else:
            t_kin_str = f"{_render_t_kin[r]:>10}"
        if _loading and S.cached_vol_at[r] == -1:
            vol_str = f"{_spin_ch:>13}"
        else:
            vol_str = f"{_render_vol[r]:>13}"

        max_visits = _render_scounts[r].max()
        if max_visits > 0:
            bar = "".join(
                " " if _render_scounts[r, s2] == 0
                else _BAR_CHARS[max(1, round(_render_scounts[r, s2] / max_visits * 6))]
                for s2 in range(n_states)
            )
        else:
            bar = " " * n_states
        trips = _render_htrips[r] // 2

        _addstr(stdscr,
            f"  {r:>7}  {s:>5}  {reduced_u_kjmol:>18.1f}  "
            f"{t_kin_str}  {vol_str}  "
        )
        _addstr(stdscr, f"{bar:>{visits_w}}", curses.color_pair(_CP_GREEN))
        _addstr(stdscr, f"  {trips:>5}\n")

    _addstr(stdscr, "\n")
    _rate_iter = S.scrub_iter if _scrubbing else display_iter
    rate_ps    = (_rate_iter + 1) * S.n_steps * S.timestep_ps
    if rate_ps > 0:
        rate_ns    = rate_ps / 1000
        trip_rates = [_render_htrips[r] // 2 / rate_ns for r in range(n_replicas)]
        avg_rate   = float(np.mean(trip_rates))
        _addstr(stdscr,
            f"  Round trip rate: {avg_rate:.2f} trips/ns avg  "
            f"(min {min(trip_rates):.2f}  max {max(trip_rates):.2f})\n"
        )

    _addstr(stdscr, "\n")
    # _tkin_stale_iters / _vol_stale_iters already computed above (before headers)
    if _tkin_stale_iters or _vol_stale_iters:
        def _fmt_iters(iters: list[int]) -> str:
            return ", ".join(str(i) for i in iters)
        if _tkin_stale_iters == _vol_stale_iters:
            _addstr(stdscr,
                f"  * T_kin & Volume from iter {_fmt_iters(_tkin_stale_iters)}\n",
                curses.color_pair(_CP_GREY))
        else:
            if _tkin_stale_iters:
                _addstr(stdscr,
                    f"  * T_kin from iter {_fmt_iters(_tkin_stale_iters)}\n",
                    curses.color_pair(_CP_GREY))
            if _vol_stale_iters:
                _addstr(stdscr,
                    f"  * Volume from iter {_fmt_iters(_vol_stale_iters)}\n",
                    curses.color_pair(_CP_GREY))

    # Bottom key-binding bar (nano-style, always at last row)
    _term_rows, _term_cols = stdscr.getmaxyx()
    if S.scrub_input_active:
        bar = f"  Jump to iteration: {S.scrub_input_buf}\u2588   (negative = from end)   Enter confirm   Esc cancel"
    elif S.rmsd_input_active:
        bar = f"  Atom selection: {S.rmsd_input_buf}\u2588   e.g. 0-64,67,200-300   Enter confirm   Esc cancel"
    else:
        _z_label = "z Live" if _scrubbing else "z Freeze"
        if mode in (3, 4, 5, 6):
            sel_hint = f" ({S.solute_sel_str})" if S.solute_sel_str else ""
            bar = f"  q Quit   s/S Sparkline   ? Explain   a Atoms{sel_hint}   w/e ±1   W/E ±5%   j Jump   {_z_label}"
        else:
            bar = f"  q Quit   s/S Sparkline   ? Explain   w/e ±1   W/E ±5%   j Jump   {_z_label}"
    bar = bar.ljust(_term_cols - 1)
    try:
        stdscr.move(_term_rows - 1, 0)
        stdscr.addstr(bar, curses.A_REVERSE)
    except curses.error:
        pass

    stdscr.refresh()


_LOOP_PERIOD = 0.05  # 50 ms — target wall time per main loop iteration


@app.default
def main(
    storage: Path = Path("cyclic_peptide.nc"),
    interval: float = 0.1,
    n_iterations: int | None = None,
    solute_n_atoms: int | None = None,
    log_file: Path | None = None,
) -> None:
    """Monitor an HREX/REST2 simulation.

    Pass --log-file monitor.log to enable debug timing output.
    """
    if log_file is not None:
        logging.disable(logging.NOTSET)
        _h = logging.FileHandler(log_file, mode="w")
        _h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                                          datefmt="%H:%M:%S"))
        _LOG.addHandler(_h)
        _LOG.setLevel(logging.DEBUG)
    while True:
        try:
            print(f"Opening {storage}...", end="", flush=True)
            reader = OpenmmToolsReader(storage)
            print(" done.")
            break
        except OSError:
            print(f"\nCannot open {storage}, retrying in {interval}s...")
            time.sleep(interval)
    init_sel = list(range(solute_n_atoms)) if solute_n_atoms is not None else None
    os.environ.setdefault("ESCDELAY", "25")  # reduce ESC recognition delay (default 1000ms)
    curses.wrapper(lambda stdscr: _main(stdscr, reader, interval, n_iterations, init_sel))


def _main(
    stdscr: curses.window,
    reader: SimulationReader,
    interval: float,
    n_iterations_arg: int | None,
    init_atom_sel: list[int] | None,
) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.init_pair(_CP_RED,    curses.COLOR_RED,    -1)
    curses.init_pair(_CP_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(_CP_GREEN,  curses.COLOR_GREEN,  -1)
    curses.init_pair(_CP_BLUE,   curses.COLOR_BLUE,   -1)
    curses.init_pair(_CP_GREY,   curses.COLOR_WHITE,  -1)
    curses.init_pair(_CP_DIM,          curses.COLOR_YELLOW, -1)
    gradient = _init_gradient_pairs() if curses.COLORS >= 256 else None

    S      = _init_state(reader, n_iterations_arg, init_atom_sel, interval)
    runner = TaskRunner()

    last_poll    = -1e9   # force immediate first poll
    needs_redraw = True

    while True:
        loop_start = time.monotonic()
        deadline   = loop_start + _LOOP_PERIOD

        # ── Input — drain all queued events so held keys scrub smoothly ──────
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key == curses.KEY_RESIZE:
                needs_redraw = True
            else:
                changed, should_quit = _handle_key(key, S)
                if should_quit:
                    return
                if changed:
                    needs_redraw = True

        # ── Scrub data fetch (when pinned to a non-live iteration) ───────────
        if S.scrub_dirty and S.scrub_iter is not None:
            try:
                _fetch_scrub_data(reader, S)
            except (OSError, IndexError, RuntimeError):
                S.scrub_iter = None
            S.scrub_dirty = False
            needs_redraw  = True

        # ── Terminal resize (KEY_RESIZE unreliable under nodelay on Linux) ─
        try:
            os_sz = os.get_terminal_size()
            cur_h, cur_w = stdscr.getmaxyx()
            if os_sz.lines != cur_h or os_sz.columns != cur_w:
                curses.resizeterm(os_sz.lines, os_sz.columns)
                stdscr.clear()
                needs_redraw = True
        except OSError:
            pass

        # ── Render first — paints "Waiting…" immediately on startup so the
        #    screen is never black while a slow poll or bulk read blocks below ─
        if needs_redraw:
            _render(stdscr, S, gradient)
            needs_redraw = False

        # ── Data poll (at the user-configured interval) ────────────────────
        if loop_start - last_poll >= interval:
            try:
                reader.refresh()
                if _poll(reader, S, stdscr):
                    needs_redraw = True
            except OSError:
                pass
            last_poll = loop_start

        # ── Submit incremental tasks (idempotent: check flags before adding) ─
        if (
            not S.history_computing
            and not S.waiting
            and S.last_mixed_iter < S.display_iter
        ):
            S.history_computing = True
            runner.submit(_history_scan_gen(reader, S))

        if (
            not S.rmsd_computing
            and not S.waiting
            and S.solute_atom_sel is not None
            and S.pos_interval > 0
            and S.pos_scan_iter <= S.display_iter
        ):
            S.rmsd_computing = True
            runner.submit(_rmsd_gen(reader, S))

        # ── Run tasks for the rest of the 50 ms budget ─────────────────────
        # round-robin across tasks; each next() = one atomic work unit.
        if runner.run_until(deadline):
            needs_redraw = True

        # ── Sleep the remainder — skip if computation consumed the full budget ─
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)



if __name__ == "__main__":
    app()

#! /usr/bin/env python3
import bisect
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
from dataclasses import dataclass, field
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

_BAR_CHARS = " ▁▂▃▄▅▆▇█"  # index by round(fraction * 6); max displayed is ▆ (index 6)

_CP_RED    = 1
_CP_YELLOW = 2
_CP_GREEN  = 3
_CP_BLUE   = 4
_CP_GREY   = 5
_CP_DIM    = 6  # dimmed colour for reference line

# 256-colour gradient key points — r/g/b in 0-5 (256-colour cube)
# gradient_fraction = sqrt(rate / 100), so:
#   frac 0.00 → rate  0%
#   frac 0.32 → rate 10%
#   frac 0.50 → rate 25%  ← sweet spot
#   frac 0.71 → rate 50%
#   frac 1.00 → rate 100%
_N_GRADIENT = 24
_CP_GRADIENT_START = 7

# Raster display palette: up to 8 visually distinct hues for replica / cluster colouring.
# Pairs _CP_RASTER_START … _CP_RASTER_START+7 are allocated by _init_raster_pairs().
_N_RASTER_PALETTE = 8
_CP_RASTER_START  = _CP_GRADIENT_START + _N_GRADIENT  # = 31

# (r, g, b) in 0-5 cube — chosen to be maximally distinct and colour-blind tolerant.
_RASTER_PALETTE: list[tuple[int, int, int]] = [
    (0, 3, 5),  # 0 blue
    (5, 3, 0),  # 1 orange
    (2, 5, 2),  # 2 green
    (5, 0, 4),  # 3 magenta
    (0, 5, 5),  # 4 cyan
    (5, 4, 0),  # 5 yellow
    (4, 1, 0),  # 6 brown
    (3, 0, 5),  # 7 purple
]

_RASTER_FALLBACK_PAIRS = [
    _CP_BLUE, _CP_YELLOW, _CP_GREEN, _CP_RED, _CP_GREY, _CP_DIM,
    _CP_BLUE, _CP_YELLOW,
]

# Default: near-black → bright blue (1%) → light cyan (10%) → near-white (25%) → gold → brown
# Explicit key point at frac=0.10 (rate=1%) creates a visible inflection so the
# 1–10% range (bright blue→cyan) is clearly distinct from the 0–1% range (near-black→blue).
# No red/green contrast — safe for deuteranopia, protanopia, and tritanopia.
_GRADIENT_KEY_POINTS: list[tuple[float, int, int, int]] = [
    (0.00, 0, 0, 1),  # rate  0%  near-black navy
    (0.10, 0, 3, 5),  # rate  1%  bright blue      ← inflection: 1–10% starts here
    (0.32, 2, 5, 5),  # rate 10%  light cyan
    (0.50, 4, 5, 5),  # rate 25%  near-white       ← sweet spot
    (0.75, 5, 4, 1),  # rate 56%  pale gold
    (1.00, 3, 2, 0),  # rate 100% muted brown
]

# Classic: red → orange → green (sweet spot) → teal → blue
# Vivid and intuitive on standard displays; not suitable for red/green
# colour blindness.  Enabled with --no-colorblind-mode.
_GRADIENT_KEY_POINTS_CLASSIC: list[tuple[float, int, int, int]] = [
    (0.00, 5, 0, 0),  # rate  0%  red
    (0.32, 5, 4, 0),  # rate 10%  orange
    (0.50, 0, 5, 0),  # rate 25%  green            ← sweet spot
    (0.75, 0, 2, 4),  # rate 56%  teal
    (1.00, 0, 0, 5),  # rate 100% blue
]


def _init_raster_pairs() -> None:
    """Allocate _N_RASTER_PALETTE curses color pairs for the raster cluster display."""
    for i, (r, g, b) in enumerate(_RASTER_PALETTE):
        color_idx = 16 + 36 * r + 6 * g + b
        curses.init_pair(_CP_RASTER_START + i, color_idx, -1)


def _raster_color_attr(idx: int, has_256: bool) -> int:
    """Return the curses attr for raster palette index *idx*."""
    if has_256:
        return curses.color_pair(_CP_RASTER_START + (idx % _N_RASTER_PALETTE))
    return curses.color_pair(_RASTER_FALLBACK_PAIRS[idx % len(_RASTER_FALLBACK_PAIRS)])


def _init_gradient_pairs(
    key_points: list[tuple[float, int, int, int]],
) -> list[int]:
    """Initialise _N_GRADIENT curses color pairs for the acceptance rate gradient."""
    pairs = []
    for i in range(_N_GRADIENT):
        t = i / (_N_GRADIENT - 1)
        # Fallback to last key point (handles t==1.0 exactly)
        _, r, g, b = key_points[-1]
        for j in range(len(key_points) - 1):
            t0, r0, g0, b0 = key_points[j]
            t1, r1, g1, b1 = key_points[j + 1]
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


def _parse_masses_from_system_xml(
    system_xml: str,
) -> tuple[np.ndarray[Any, np.dtype[Any]], list[tuple[int, int]], bool]:
    """Parse particle masses (amu), constraint pairs, and CMMotionRemover presence
    from an OpenMM System XML string."""
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


# Atomic masses (amu) for element lookup.  Values are standard atomic weights;
# HMR-adjusted heavy-atom masses are corrected before this table is consulted.
_ELEMENT_MASSES: dict[str, float] = {
    "H":  1.008,  "He":  4.003, "Li":  6.941, "Be":  9.012, "B":  10.811,
    "C": 12.011,  "N":  14.007, "O":  15.999, "F":  18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P":  30.974,
    "S":  32.065, "Cl": 35.453, "Ar": 39.948, "K":  39.098, "Ca": 40.078,
    "Fe": 55.845, "Cu": 63.546, "Zn": 65.380, "Br": 79.904, "I": 126.904,
}
_ELEMENT_MASS_ITEMS = sorted(_ELEMENT_MASSES.items(), key=lambda kv: kv[1])


def _format_index_ranges(indices: list[int]) -> str:
    """Compress a sorted list of 0-based indices into a compact range string.

    E.g. [0,1,2,5,6,7] → "0-2,5-7".  Single indices are shown without a dash.
    """
    if not indices:
        return ""
    parts: list[str] = []
    start = indices[0]
    end   = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            parts.append(str(start) if start == end else f"{start}-{end}")
            start = end = idx
    parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


def _nearest_element(mass: float) -> str:
    """Return the element symbol whose standard atomic mass is closest to *mass*."""
    return min(_ELEMENT_MASS_ITEMS, key=lambda kv: abs(kv[1] - mass))[0]


def _hill_formula(element_counts: dict[str, int]) -> str:
    """Return Hill-order formula string (C first, H second, then alphabetical)."""
    counts = dict(element_counts)
    parts: list[str] = []
    for el in ("C", "H"):
        if el in counts:
            parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
            del counts[el]
    for el in sorted(counts):
        parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
    return "".join(parts)


def _parse_system_topology(
    system_xml: str,
) -> list[tuple[str, list[int]]]:
    """Parse an OpenMM System XML string and return one entry per molecule:
    ``(hill_formula, sorted_atom_indices)``, ordered by each molecule's lowest
    atom index.

    Bond sources: ``<Constraints>`` + every ``<Force>`` whose ``type``
    attribute contains ``"Bond"``.

    HMR handling: degree-1 atoms lighter than 6 amu are identified as
    hydrogen regardless of their exact mass.  Their median mass is used to
    estimate how much mass was transferred so that heavy-atom masses can be
    corrected before element lookup.
    """
    root = ET.fromstring(system_xml)

    particles_elem = root.find("Particles")
    assert particles_elem is not None
    masses = [float(p.attrib["mass"]) for p in particles_elem]
    n = len(masses)

    # ── Build adjacency from all bond-like sources ────────────────────────
    adj: list[set[int]] = [set() for _ in range(n)]

    def _add_bond(p1: int, p2: int) -> None:
        adj[p1].add(p2)
        adj[p2].add(p1)

    constraints_elem = root.find("Constraints")
    if constraints_elem is not None:
        for c in constraints_elem:
            _add_bond(int(c.attrib["p1"]), int(c.attrib["p2"]))

    forces_elem = root.find("Forces")
    if forces_elem is not None:
        for force in forces_elem:
            if "Bond" not in force.attrib.get("type", ""):
                continue
            bonds_block = force.find("Bonds")
            if bonds_block is None:
                continue
            for bond in bonds_block:
                a = bond.attrib
                # OpenMM uses p1/p2 for HarmonicBondForce; CustomBondForce uses
                # the same convention.  Fall back to particle1/particle2 if present.
                p1 = int(a.get("p1", a.get("particle1", -1)))
                p2 = int(a.get("p2", a.get("particle2", -1)))
                if p1 >= 0 and p2 >= 0:
                    _add_bond(p1, p2)

    # ── HMR-aware element assignment ──────────────────────────────────────
    # Pass 1: identify hydrogen candidates by connectivity (degree 1) and a
    # generous mass ceiling (6 amu) that covers standard H and HMR H up to 4×.
    # Carbonyl / terminal oxygens also have degree 1 but mass ≥ 16 — safely above.
    H_MASS_CEIL = 6.0
    h_candidates: set[int] = {
        i for i in range(n) if len(adj[i]) == 1 and masses[i] < H_MASS_CEIL
    }
    # Isolated light atoms (no bonds, e.g. H in H₂ after dissociation) also H.
    h_candidates |= {i for i in range(n) if len(adj[i]) == 0 and masses[i] < H_MASS_CEIL}

    # Pass 2: estimate HMR factor from median H-candidate mass.
    if h_candidates:
        h_mass_median = float(np.median([masses[i] for i in h_candidates]))
        transferred_per_h = max(0.0, h_mass_median - 1.008)  # mass moved onto each H
    else:
        h_mass_median = 1.008
        transferred_per_h = 0.0

    # Pass 3: assign elements.
    elements: list[str] = []
    for i, mass in enumerate(masses):
        if i in h_candidates:
            elements.append("H")
        else:
            n_h_bonded = sum(1 for j in adj[i] if j in h_candidates)
            corrected = mass + n_h_bonded * transferred_per_h
            elements.append(_nearest_element(corrected))

    # ── Connected components → molecules ──────────────────────────────────
    visited = [False] * n
    molecules: list[tuple[str, list[int]]] = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adj[node]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        component.sort()
        counts: dict[str, int] = {}
        for idx in component:
            el = elements[idx]
            counts[el] = counts.get(el, 0) + 1
        molecules.append((_hill_formula(counts), component))

    molecules.sort(key=lambda m: m[1][0])
    return molecules


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
        pass  # raised when text reaches the bottom-right corner; harmless


# Braille dot bit values indexed by [sub_col][sub_row].
# sub_col: 0 = left half of cell, 1 = right half.
# sub_row: 0 = top, 3 = bottom (within the cell's 4-row band).
# Sparkline rendering: if the mean number of data points per non-empty bin
# exceeds this threshold the whole sparkline switches to mean+range marks;
# below it every data point is plotted as an individual green dot.  Applied
# globally so all columns use the same representation.
_SPARK_DOTS_THRESHOLD = 3

# Unicode braille block starts at U+2800; add bit pattern to get character.
_BRAILLE_BITS: list[list[int]] = [
    [0x01, 0x02, 0x04, 0x40],  # left column
    [0x08, 0x10, 0x20, 0x80],  # right column
]

# Reference line characters, evenly trisecting the terminal row (top, middle, bottom).
_REF_CHARS = ("\u203e", "\u2500", "\u005f")


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
    individual_dots: bool = False,
    y_min_fixed: float | None = None,
    y_max_fixed: float | None = None,
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

    # Pre-compute numpy arrays.  History is accumulated in replica order within
    # each chunk, so x values may be out of chronological order.  Sort both
    # arrays together so np.searchsorted produces correct bin assignments.
    hist_arr = np.asarray(ground_u_history)
    if x_transform is math.log10:
        x_arr   = np.log10(np.asarray(ground_u_iters, dtype=float))
        x_total = float(np.log10(total_iters)) * 1.05
    elif x_transform is not None:
        x_arr   = np.asarray([x_transform(it) for it in ground_u_iters])
        x_total = x_transform(total_iters) * 1.05
    else:
        x_arr   = np.asarray(ground_u_iters, dtype=float)
        x_total = float(total_iters)
    sort_idx = np.argsort(x_arr, kind="stable")
    x_arr    = x_arr[sort_idx]
    hist_arr = hist_arr[sort_idx]

    if y_min_fixed is not None and y_max_fixed is not None:
        y_min, y_max = y_min_fixed, y_max_fixed
    else:
        y_min = float(hist_arr.min())
        y_max = float(hist_arr.max())
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

    def _ref_row_and_char(v: float) -> tuple[int, str]:
        """Map value to (terminal_row, reference_char) using a 3-way split.

        The three characters evenly trisect the terminal row, independently of
        the 4-sub-row braille mapping used for data dots."""
        frac      = (v - y_min) / (y_max - y_min)
        row_exact = (1.0 - frac) * (height - 1)
        r         = max(0, min(height - 1, round(row_exact)))
        within    = max(0.0, min(1.0 - 1e-9, row_exact - r + 0.5))
        return r, _REF_CHARS[int(within * 3)]

    # ── Sparse fallback: plain ●/│ characters ────────────────────────────
    if len(ground_u_history) < width:
        bin_edges  = np.linspace(0.0, x_total, width + 1)
        lo_indices = np.searchsorted(x_arr, bin_edges[:-1], side="left")
        hi_indices = np.searchsorted(x_arr, bin_edges[1:],  side="left")
        bin_counts = hi_indices - lo_indices
        n_nonempty = int(np.count_nonzero(bin_counts))
        mean_count = float(bin_counts.sum()) / max(1, n_nonempty)
        use_mean_range = not individual_dots and mean_count > _SPARK_DOTS_THRESHOLD
        for c in range(width):
            lo_idx = int(lo_indices[c])
            hi_idx = int(hi_indices[c])
            if lo_idx >= hi_idx:
                continue
            vals = hist_arr[lo_idx:hi_idx]  # O(1) view, no copy
            if use_mean_range:
                mean_r = _cell_row(float(vals.mean()))
                top_r  = _cell_row(float(vals.max()))
                bot_r  = _cell_row(float(vals.min()))
                for r in range(top_r, bot_r + 1):
                    grid[r][c] = ("●" if r == mean_r else "│", cp_mean if r == mean_r else cp_range)
            else:
                for v in vals:
                    r = _cell_row(float(v))
                    grid[r][c] = ("●", cp_mean)

        # Reference line: ‾/─/_ in empty cells only, position independent of data mapping.
        if ref_val is not None:
            ref_r, ref_ch = _ref_row_and_char(ref_val)
            for c in range(width):
                if grid[ref_r][c][0] == " ":
                    grid[ref_r][c] = (ref_ch, cp_ref)

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

    # Reference line — ‾/─/_ in empty cells, mapped independently of braille sub-rows.
    ref_row:  int | None = None
    ref_char: str        = ""
    if ref_val is not None:
        ref_row, ref_char = _ref_row_and_char(ref_val)

    # All bin boundaries in one searchsorted call, then loop over non-empty bins.
    bin_edges  = np.linspace(0.0, x_total, data_w + 1)
    lo_indices = np.searchsorted(x_arr, bin_edges[:-1], side="left")
    hi_indices = np.searchsorted(x_arr, bin_edges[1:],  side="left")
    bin_counts = hi_indices - lo_indices
    n_nonempty = int(np.count_nonzero(bin_counts))
    mean_count = float(bin_counts.sum()) / max(1, n_nonempty)
    use_mean_range = not individual_dots and x_transform is None and mean_count > _SPARK_DOTS_THRESHOLD

    for dc in range(data_w):
        lo_idx = int(lo_indices[dc])
        hi_idx = int(hi_indices[dc])
        if lo_idx >= hi_idx:
            continue
        vals = hist_arr[lo_idx:hi_idx]  # O(1) view, no copy
        if use_mean_range:
            mean_dr = _data_row(float(vals.mean()))
            top_dr  = _data_row(float(vals.max()))
            bot_dr  = _data_row(float(vals.min()))
            # Only topmost and bottommost range dots; mean dot in its own layer.
            _dot(mean_bits, dc, mean_dr)
            if top_dr != mean_dr:
                _dot(range_bits, dc, top_dr)
            if bot_dr != mean_dr:
                _dot(range_bits, dc, bot_dr)
        else:
            for v in vals:
                _dot(mean_bits, dc, _data_row(float(v)))

    # Compose: mean wins, then range; reference fills empty cells in its row.
    for r in range(height):
        for c in range(width):
            if mean_bits[r][c]:
                grid[r][c] = (chr(0x2800 + mean_bits[r][c]), cp_mean)
            elif range_bits[r][c]:
                grid[r][c] = (chr(0x2800 + range_bits[r][c]), cp_range)
            elif r == ref_row:
                grid[r][c] = (ref_char, cp_ref)

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


def _kabsch_align(
    P: np.ndarray[Any, np.dtype[Any]],
    Q: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Return P centred and rotated to best superpose onto centred Q (both N×3 in nm)."""
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return np.array(P_c @ R.T)


def _centroid_rmsd_vals(
    frames: list[np.ndarray[Any, np.dtype[Any]]],
) -> list[float]:
    """Per-frame RMSD (Å) to the centroid (mean superposed structure).

    All frames are first aligned to frame 0 via Kabsch superposition; the
    centroid is then the coordinate-wise mean of the aligned ensemble.  One
    Procrustes iteration is sufficient for monitoring purposes.
    """
    if not frames:
        return []
    ref = frames[0]
    aligned: list[np.ndarray[Any, np.dtype[Any]]] = [
        ref - ref.mean(axis=0)
    ] + [_kabsch_align(f, ref) for f in frames[1:]]
    centroid: np.ndarray[Any, np.dtype[Any]] = np.mean(aligned, axis=0)
    return [
        float(np.sqrt(np.mean(np.sum((a - centroid) ** 2, axis=-1)))) * 10.0
        for a in aligned
    ]


def _centroid_rmsd_gen(S: types.SimpleNamespace) -> Generator[None, None, None]:
    """Compute centroid RMSD incrementally, yielding after each Kabsch alignment.

    Snapshots ``S.pos_all_frames`` at entry so mid-run frame additions do not
    corrupt the ensemble.  Writes results to ``S.centroid_rmsd_cache`` and
    ``S.centroid_n_frames_cached`` atomically at the end.  Clears
    ``S.centroid_computing`` on exit (including on exception).
    """
    try:
        frames = list(S.pos_all_frames)   # snapshot — immune to concurrent appends
        n = len(frames)
        if n == 0:
            return
        ref = frames[0]
        aligned: list[np.ndarray[Any, np.dtype[Any]]] = [ref - ref.mean(axis=0)]
        for f in frames[1:]:
            aligned.append(_kabsch_align(f, ref))
            yield
        centroid: np.ndarray[Any, np.dtype[Any]] = np.mean(aligned, axis=0)
        S.centroid_rmsd_cache = [
            float(np.sqrt(np.mean(np.sum((a - centroid) ** 2, axis=-1)))) * 10.0
            for a in aligned
        ]
        S.centroid_n_frames_cached = n
    finally:
        S.centroid_computing = False


def _medoid_rmsd_gen(S: types.SimpleNamespace) -> Generator[None, None, None]:
    """Compute medoid RMSD from the pairwise matrix without any additional Kabsch SVDs.

    Snapshots ``S.pairwise_rmsd_mat`` at entry.  Builds the symmetric distance
    matrix one row at a time (yielding after each row), finds the frame that
    minimises the sum of pairwise distances (the medoid), then stores per-frame
    RMSDs to the medoid in ``S.medoid_rmsd_cache``.  Clears ``S.medoid_computing``
    on exit.
    """
    try:
        mat_rows = [list(row) for row in S.pairwise_rmsd_mat]  # snapshot
        n = len(mat_rows)
        if n < 2:
            return
        mat = np.zeros((n, n))
        for i, row in enumerate(mat_rows):
            mat[i, :len(row)] = row
            yield
        mat = mat + mat.T
        medoid_idx = int(np.argmin(mat.sum(axis=1)))
        S.medoid_rmsd_cache = (mat[:, medoid_idx] * 10.0).tolist()
        S.medoid_n_frames_cached = n
    finally:
        S.medoid_computing = False


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

    @property
    def n_atoms(self) -> int:
        """Number of atoms per replica in the positions array; 0 if unavailable."""
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

    def replica_states_range(self, start: int, end: int) -> np.ndarray[Any, np.dtype[Any]]:
        """(end-start, n_replicas) int: thermodynamic states for iterations start..end-1."""
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

    def get_title(self) -> str:
        """A short human-readable label for this trajectory (file name, run ID, …)."""
        ...

    def parse_topology(self) -> list[tuple[str, list[int]]]:
        """Return one entry per molecule: (hill_formula, sorted_atom_indices).

        Molecules are ordered by their lowest atom index.  Atom indices are
        0-based and may be non-contiguous (e.g. after solvent grouping).
        Raises if the topology cannot be determined from the trajectory file.
        """
        ...


class OpenmmToolsReader:
    """SimulationReader for openmmtools HREX netCDF4 files."""

    # ── NC-specific private helpers ────────────────────────────────────────

    def _read_thermo_state(self) -> dict[str, Any]:
        """Parse the first thermodynamic state from the open NC handle."""
        raw = b"".join(self._nc.groups["thermodynamic_states"].variables["state0"][:]).decode()
        return yaml.safe_load(raw)

    def _read_mcmc_move(self) -> dict[str, Any]:
        """Parse MCMC moves from the open NC handle.

        Asserts one identical move per replica and returns the first move's doc.
        """
        move_vars  = self._nc.groups["mcmc_moves"].variables
        n_replicas = self._nc.variables["energies"].shape[1]
        assert len(move_vars) == n_replicas and all(
            move_vars[f"move{i}"][0] == move_vars["move0"][0] for i in range(1, n_replicas)
        ), f"Expected {n_replicas} identical MCMC moves, got: {list(move_vars)}"
        return yaml.safe_load(str(move_vars["move0"][0]))

    def _system_xml(self) -> str:
        """Decompress and return the OpenMM System XML embedded in the NC file."""
        doc = self._read_thermo_state()
        return zlib.decompress(doc["standard_system"]).decode()

    # ── Construction ───────────────────────────────────────────────────────

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

        all_masses, all_constraints, has_cm_remover = _parse_masses_from_system_xml(
            self._system_xml()
        )
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

        move_doc = self._read_mcmc_move()
        assert "n_steps" in move_doc and "integrator" in move_doc
        integrator_root = ET.fromstring(move_doc["integrator"])
        assert "stepSize" in integrator_root.attrib
        self._timestep_ps   = float(integrator_root.attrib["stepSize"])
        self._steps_per_iter = int(move_doc["n_steps"])

        thermo_doc  = self._read_thermo_state()
        temp_entry  = thermo_doc["temperature"]
        assert isinstance(temp_entry, dict) and "value" in temp_entry
        self._ref_temp_k = float(temp_entry["value"])

        self._n_iterations: int | None = None
        try:
            opts = yaml.safe_load(str(self._nc.variables["options"][0]))
            self._n_iterations = int(opts["number_of_iterations"])
        except (KeyError, TypeError, ValueError, yaml.YAMLError):
            pass  # options variable absent or malformed — n_iterations stays None
        except Exception:
            _LOG.warning("unexpected error reading n_iterations from options", exc_info=True)

        self._pos_interval = 1
        try:
            self._pos_interval = int(self._nc.PositionInterval)
        except AttributeError:
            pass  # optional NC global attribute; default 1 (conservative: read every iteration)

        self._vel_interval = 1
        try:
            self._vel_interval = int(self._nc.VelocityInterval)
        except AttributeError:
            pass  # optional NC global attribute; default 1 (conservative: read every iteration)

        self._has_t_kin     = has_ap and "velocities" in self._nc.variables
        self._has_volume    = "volumes"    in self._nc.variables
        self._has_positions = "positions"  in self._nc.variables
        self._n_atoms       = (int(self._nc.variables["positions"].shape[2])  # type: ignore[union-attr]
                               if self._has_positions else 0)

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
    @property
    def n_atoms(self) -> int:          return self._n_atoms

    # ── File management ────────────────────────────────────────────────────

    def refresh(self) -> None:
        try:
            self._nc.close()
        except RuntimeError:
            pass  # netCDF4 raises RuntimeError if the file is already closed
        self._nc = netCDF4.Dataset(str(self._storage), "r")

    def close(self) -> None:
        try:
            self._nc.close()
        except RuntimeError:
            pass  # netCDF4 raises RuntimeError if the file is already closed

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

    def replica_states_range(self, start: int, end: int) -> np.ndarray[Any, np.dtype[Any]]:
        return self._nc.variables["states"][start:end, :].astype(int)  # type: ignore[union-attr]

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
        except (OSError, IndexError, RuntimeError, ValueError):
            return None  # missing or unreadable timestamp — caller treats None as unavailable
        except Exception:
            _LOG.warning("unexpected error reading timestamp at iteration %d", iteration, exc_info=True)
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

    def get_title(self) -> str:
        return self._storage.name

    def parse_topology(self) -> list[tuple[str, list[int]]]:
        return _parse_system_topology(self._system_xml())


# ── State initialisation and event loop ───────────────────────────────────────


def _init_state(
    reader: SimulationReader,
    n_iterations_arg: int | None,
    init_atom_sel: list[int] | None,
    interval: float,
    storage: "Path",
) -> types.SimpleNamespace:
    """Build initial mutable state namespace from a SimulationReader."""
    n_replicas   = reader.n_replicas
    n_states     = reader.n_states
    n_iterations = n_iterations_arg if n_iterations_arg is not None else reader.n_iterations

    return types.SimpleNamespace(
        # Derived from reader static properties
        storage_path=storage,
        title=reader.get_title(),
        flash_msg="",
        flash_until=0.0,
        interval=interval,
        n_replicas=n_replicas,
        n_states=n_states,
        n_steps=reader.steps_per_iter,
        timestep_ps=reader.timestep_ps,
        ref_temp_k=reader.ref_temp_k,
        kt_kjmol=reader.ref_temp_k * KB_KJMOL_PER_K,
        n_iterations=n_iterations,
        has_t_kin=reader.has_t_kin,
        has_volume=reader.has_volume,
        pos_interval=reader.pos_interval,
        n_atoms=reader.n_atoms,
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
        state0_replica_iters=[],      # iteration index for each state-0 replica record
        state0_replica_vals=[],       # float replica index at each recorded iteration
        state0_scan_iter=0,           # next iteration to scan for state-0 replica
        state0_scanning=False,        # True while _state0_scan_gen is running
        pos_frame_iters=[],
        pos_all_frames=[],
        pairwise_rmsd_mat=[],
        n_pos_frames=0,
        centroid_rmsd_cache=[],       # per-frame centroid RMSD (Å); recomputed when frame count changes
        centroid_n_frames_cached=0,   # len(pos_all_frames) when centroid_rmsd_cache was last computed
        centroid_computing=False,     # True while _centroid_rmsd_gen task is in the runner
        medoid_rmsd_cache=[],         # per-frame medoid RMSD (Å); recomputed when frame count changes
        medoid_n_frames_cached=0,     # len(pairwise_rmsd_mat) when medoid_rmsd_cache was last computed
        medoid_computing=False,       # True while _medoid_rmsd_gen task is in the runner
        cluster_k=None,               # int | None — k chosen by bootstrap stability
        k_choosing=False,             # True while _stability_choose_k_gen is running
        k_chosen_n_frames=0,          # n_pos_frames when cluster_k was last chosen
        cluster_labels=[],               # int cluster index per frame; written atomically by _cluster_gen
        cluster_medoids=[],              # frame indices of medoids
        cluster_outlier_mask=[],         # bool per frame: True if dist_to_medoid > mean+N*std
        cluster_intercluster_dists=[],   # k×k list[list[float]] of inter-medoid distances
        cluster_n_frames_cached=0,       # n_pos_frames when cluster_labels was last computed
        cluster_computing=False,         # True while _cluster_gen task is in the runner
        pos_scan_iter=0,      # next iteration index to attempt for RMSD accumulation
        history_computing=False,    # True while _history_scan_gen task is in the runner
        history_bulk_loading=True,  # False after the first history scan completes
        rmsd_computing=False,     # True while _rmsd_gen task is in the runner
        # UI state
        sparkline_mode=0,
        show_spark_help=False,
        rmsd_input_active=False,
        rmsd_input_buf="",
        rmsd_sel_error="",
        # Scrub state
        scrub_iter=None,          # None = live; int = pinned display iteration
        scrub_dirty=False,        # True when scrub_iter changed and data not yet read
        scrub_states=None,        # replica_states at scrub_iter
        scrub_energies=None,      # energies at scrub_iter
        scrub_acc_sum=None,       # (n_states, n_states) accepted counts up to scrub_iter
        scrub_prop_sum=None,      # (n_states, n_states) proposed counts up to scrub_iter
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


def _export_medoids_pdb(S: types.SimpleNamespace) -> str:
    """Write each cluster medoid as a MODEL in a PDB file. Returns output path or ''."""
    if not S.cluster_medoids or not S.pos_all_frames:
        return ""
    out_path = S.storage_path.with_name(S.storage_path.stem + "_medoids.pdb")
    with open(out_path, "w") as fh:
        for model_num, med_idx in enumerate(S.cluster_medoids, 1):
            if med_idx >= len(S.pos_all_frames):
                continue
            pos_ang = np.asarray(S.pos_all_frames[med_idx]) * 10.0  # nm → Å
            fh.write(f"MODEL     {model_num:4d}\n")
            fh.write(f"REMARK cluster {model_num}  frame {S.pos_frame_iters[med_idx]}\n")
            for serial, (x, y, z) in enumerate(pos_ang, 1):
                fh.write(
                    f"ATOM  {serial:5d}  CA  UNK A{serial:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
            fh.write("ENDMDL\n")
        fh.write("END\n")
    return str(out_path)


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
                pass  # user typed a non-integer; silently discard and close the prompt
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
                new_sel = _parse_atom_selection(S.rmsd_input_buf)
                if S.n_atoms > 0 and max(new_sel) >= S.n_atoms:
                    raise ValueError(f"atom index {max(new_sel)} out of range (max {S.n_atoms - 1})")
                S.solute_atom_sel = new_sel
                S.solute_sel_str = S.rmsd_input_buf
                S.pos_frame_iters           = []
                S.pos_all_frames            = []
                S.pairwise_rmsd_mat         = []
                S.n_pos_frames              = 0
                S.centroid_rmsd_cache       = []
                S.centroid_n_frames_cached  = 0
                S.medoid_rmsd_cache         = []
                S.medoid_n_frames_cached    = 0
                S.cluster_k                  = None
                S.k_choosing                 = False
                S.k_chosen_n_frames          = 0
                S.cluster_labels             = []
                S.cluster_medoids            = []
                S.cluster_outlier_mask       = []
                S.cluster_intercluster_dists = []
                S.cluster_n_frames_cached    = 0
                S.pos_scan_iter    = 0
                S.rmsd_computing     = False  # old tasks invalidated; runner drains naturally
                S.centroid_computing = False
                S.medoid_computing   = False
                S.cluster_computing  = False
                S.rmsd_sel_error   = ""
                S.rmsd_input_active = False
                S.rmsd_input_buf = ""
            except ValueError as exc:
                S.rmsd_sel_error = str(exc)
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
            S.sparkline_mode = (S.sparkline_mode + 1) % len(_SPARK_MODES)
            return True, False
        elif key == ord("S"):
            S.sparkline_mode = (S.sparkline_mode - 1) % len(_SPARK_MODES)
            return True, False
        elif key == ord("?"):
            S.show_spark_help = not S.show_spark_help
            return True, False
        elif key == ord("k") and not S.k_choosing and not S.cluster_computing:
            S.cluster_k                  = None
            S.k_chosen_n_frames          = 0
            S.cluster_labels             = []
            S.cluster_medoids            = []
            S.cluster_outlier_mask       = []
            S.cluster_intercluster_dists = []
            S.cluster_n_frames_cached    = 0
            return True, False
        elif key == ord("e") and S.cluster_medoids:
            path = _export_medoids_pdb(S)
            if path:
                S.flash_msg   = f"Medoids written → {path}"
            else:
                S.flash_msg   = "No clusters yet — nothing to export"
            S.flash_until = time.monotonic() + 6.0
            return True, False
        elif key == ord("a") and _SPARK_MODES[S.sparkline_mode].needs_atoms:
            S.rmsd_input_active = True
            S.rmsd_input_buf = S.solute_sel_str
            S.rmsd_sel_error = ""
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
                worked = True  # task finished — final cleanup may have mutated state
        return worked

    def __bool__(self) -> bool:
        return bool(self._tasks)


# ── Sparkline mode registry ────────────────────────────────────────────────────


@dataclass
class SparkMode:
    """All per-mode data for a single sparkline graph.

    Adding a new graph means adding one ``SparkMode`` entry to ``_SPARK_MODES``.
    Nothing else needs updating: mode count, key cycling, atom-selection guards,
    data dispatch, reference line, poll submission, and help text are all derived
    from this list automatically.
    """

    name: str
    unit: str
    help_text: str
    # Callable signatures:
    #   get_data(S, spark_w)  -> (iters, vals)
    #   get_ref(S, vals)      -> float | None
    #   get_grid(S, sp_iters, total_iters) -> (grid_total, xaxis_l, xaxis_r)
    #                            None = use standard time axis
    #   xaxis_m_fn(S)         -> centre-label string (overrides auto dot-density label)
    #   poll(S, runner)       -> None  (submit background recompute tasks as needed)
    get_data: Callable[..., tuple[list[int], list[float]]] = field(repr=False)
    get_ref:  Callable[..., float | None]                  = field(repr=False)
    needs_atoms:    bool = False
    individual_dots: bool = False
    x_transform: Callable[[float], float] | None = field(default=None, repr=False)
    multistate:   bool = False   # "State 0→N" prefix instead of "State 0"
    uses_history: bool = False   # show "loading history…" progress in title
    get_grid:    Callable[..., tuple[int, str, str]] | None = field(default=None, repr=False)
    xaxis_m_fn:  Callable[..., str] | None                  = field(default=None, repr=False)
    poll:        Callable[..., None] | None                  = field(default=None, repr=False)
    # When set, replaces _build_sparkline_grid for 2-D raster displays.
    # Signature: (S, spark_w, n_rows, total_iters, has_256) -> list[list[(char, attr)]]
    raster_fn:   Callable[..., list[list[tuple[str, int]]]] | None = field(default=None, repr=False)
    # Fixed y-axis range, bypassing data-driven scaling. Callable so it can depend on S.
    # Signature: (S,) -> (y_min, y_max)
    y_fixed_range: Callable[..., tuple[float, float]] | None = field(default=None, repr=False)


# ── get_data helpers ───────────────────────────────────────────────────────────

def _data_ground_u(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    return S.ground_u_iters, S.ground_u_history

def _data_ground_vol(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    return S.ground_vol_iters, S.ground_vol_history

def _data_fe(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    return getattr(S, "fe_iters", []), getattr(S, "fe_vals", [])

def _data_rmsd(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    if S.n_pos_frames:
        vals = [0.0] + [S.pairwise_rmsd_mat[i][0] for i in range(1, S.n_pos_frames)]
        return S.pos_frame_iters, vals
    return [], []

def _data_min_rmsd(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    if S.n_pos_frames > 1:
        return S.pos_frame_iters[1:], [min(S.pairwise_rmsd_mat[i]) for i in range(1, S.n_pos_frames)]
    return [], []

def _data_max_rmsd(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    if S.n_pos_frames > 1:
        return S.pos_frame_iters[1:], [max(S.pairwise_rmsd_mat[i]) for i in range(1, S.n_pos_frames)]
    return [], []

def _data_acf(S: types.SimpleNamespace, spark_w: int) -> tuple[list[int], list[float]]:
    if S.n_pos_frames < 2:
        return [], []
    max_lag = S.n_pos_frames // 2
    n_pts   = min(max_lag, max(4, 2 * spark_w))
    raw_lags = np.unique(np.round(np.geomspace(1, max_lag, n_pts)).astype(int))
    raw_lags = raw_lags[(raw_lags >= 1) & (raw_lags <= max_lag)]
    vals = [
        float(np.mean([S.pairwise_rmsd_mat[i + int(lag)][i]
                        for i in range(S.n_pos_frames - int(lag))]))
        for lag in raw_lags
    ]
    return [int(lag) for lag in raw_lags], vals

def _data_centroid_rmsd(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    n = S.centroid_n_frames_cached
    return S.pos_frame_iters[:n], S.centroid_rmsd_cache

def _data_medoid_rmsd(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    n = S.medoid_n_frames_cached
    return S.pos_frame_iters[:n], S.medoid_rmsd_cache

_STATE0_DOTS_PER_COL = 5  # target rendered points per terminal column in state 0 replica over time view

def _data_state0_replica(S: types.SimpleNamespace, spark_w: int) -> tuple[list[int], list[float]]:
    iters = S.state0_replica_iters
    vals  = S.state0_replica_vals
    n = len(iters)
    if n == 0:
        return [], []
    stride = max(1, n // (spark_w * _STATE0_DOTS_PER_COL))
    return iters[::stride], vals[::stride]


# ── get_ref helpers ────────────────────────────────────────────────────────────

def _ref_none(S: types.SimpleNamespace, vals: list[float]) -> float | None:
    return None

def _ref_first(S: types.SimpleNamespace, vals: list[float]) -> float | None:
    return vals[0] if vals else None

def _ref_mean(S: types.SimpleNamespace, vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None

def _ref_min_pairwise(S: types.SimpleNamespace, vals: list[float]) -> float | None:
    return (min(v for row in S.pairwise_rmsd_mat for v in row)
            if S.n_pos_frames >= 2 else None)

def _ref_max_pairwise(S: types.SimpleNamespace, vals: list[float]) -> float | None:
    return (max(v for row in S.pairwise_rmsd_mat for v in row)
            if S.n_pos_frames >= 2 else None)

def _ref_acf_plateau(S: types.SimpleNamespace, vals: list[float]) -> float | None:
    if len(vals) >= 4:
        half = len(vals) // 2
        return float(np.mean(vals[half:]))
    return None


# ── get_grid / xaxis_m helpers (ACF only needs custom grid) ───────────────────

def _grid_acf(
    S: types.SimpleNamespace, sp_iters: list[int], total_iters: int
) -> tuple[int, str, str]:
    frame_ps = S.pos_interval * S.n_steps * S.timestep_ps
    if not sp_iters:
        total_ns = total_iters * S.n_steps * S.timestep_ps / 1000
        return total_iters, "0 ns", f"{total_ns:.1f} ns"
    acf_max_lag = max(sp_iters)
    return (
        acf_max_lag,
        f"lag {_fmt_2sf(frame_ps / 1000)}ns (log)",
        f"lag {acf_max_lag * frame_ps / 1000:.1f}ns",
    )

def _xaxis_m_acf(S: types.SimpleNamespace) -> str:
    return f"{S.n_pos_frames} frames" if S.n_pos_frames else ""

def _data_cluster_occupancy(S: types.SimpleNamespace, _w: int) -> tuple[list[int], list[float]]:
    k = S.cluster_k
    if not k or not S.cluster_labels:
        return [], []
    n = len(S.cluster_labels)
    return list(range(k)), [S.cluster_labels.count(c) / n for c in range(k)]

def _grid_cluster_occupancy(
    S: types.SimpleNamespace, sp_iters: list[int], total_iters: int
) -> tuple[int, str, str]:
    k = S.cluster_k or 1
    return k - 1, "cluster 0", f"cluster {k - 1}"

def _xaxis_m_cluster_occupancy(S: types.SimpleNamespace) -> str:
    k   = S.cluster_k
    lab = "choosing k…" if S.k_choosing else (f"k={k}" if k else "")
    n   = len(S.cluster_labels)
    return f"{lab}  {n} frames" if n else lab


# ── poll helpers ───────────────────────────────────────────────────────────────

def _poll_centroid(S: types.SimpleNamespace, runner: TaskRunner) -> None:
    if (not S.centroid_computing
            and S.pos_all_frames
            and len(S.pos_all_frames) != S.centroid_n_frames_cached):
        S.centroid_computing = True
        runner.submit(_centroid_rmsd_gen(S))

def _poll_medoid(S: types.SimpleNamespace, runner: TaskRunner) -> None:
    if (not S.medoid_computing
            and S.n_pos_frames >= 2
            and S.n_pos_frames != S.medoid_n_frames_cached):
        S.medoid_computing = True
        runner.submit(_medoid_rmsd_gen(S))


_K_MIN_FRAMES          = 20   # minimum frames before attempting k-selection
_K_GROWTH_FACTOR       = 1.5  # retrigger k-selection when frame count grows by this factor
_K_MAX                 = 8    # never try more than this many clusters
_K_BOOTSTRAP_B         = 20   # bootstrap replicates per candidate k
_OUTLIER_STD_THRESHOLD = 2.0  # flag frames > mean + N*std distance to medoid as outliers


def _mds_1d(dist_mat: np.ndarray) -> np.ndarray:
    """Embed k points in 1D using classical MDS on a k×k distance matrix.

    Returns k positions (arbitrary scale/sign; caller normalises).  For k≤2
    the result is exact; for k>2 it uses the leading eigenvector of the
    double-centred squared-distance matrix.
    """
    k = len(dist_mat)
    if k == 1:
        return np.zeros(1)
    if k == 2:
        return np.array([0.0, float(dist_mat[0, 1])])
    D2 = np.asarray(dist_mat, dtype=float) ** 2
    H  = np.eye(k) - np.ones((k, k)) / k
    B  = -0.5 * H @ D2 @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    pos = eigvecs[:, -1] * np.sqrt(max(0.0, float(eigvals[-1])))
    return pos


def _kmedoids(dist_mat: np.ndarray, k: int, n_iter: int = 50) -> np.ndarray:
    """K-medoids on a precomputed symmetric distance matrix; returns label array."""
    n = len(dist_mat)
    # Greedy farthest-point initialisation
    medoids: list[int] = [0]
    for _ in range(k - 1):
        dists = dist_mat[:, medoids].min(axis=1)
        medoids.append(int(np.argmax(dists)))
    # PAM-style update
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        labels = np.argmin(dist_mat[:, medoids], axis=1)
        new_medoids = list(medoids)
        changed = False
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            costs = dist_mat[np.ix_(members, members)].sum(axis=1)
            new_med = int(members[np.argmin(costs)])
            if new_med != new_medoids[c]:
                changed = True
            new_medoids[c] = new_med
        medoids = new_medoids
        if not changed:
            break
    return np.argmin(dist_mat[:, medoids], axis=1)


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index between two label arrays.

    Corrects for chance: expected value is 0 for random labellings, 1 for
    identical.  Unlike the plain Rand index it does not systematically reward
    small k, so argmax over k gives an unbiased cluster count.
    """
    n = len(a)
    if n < 2:
        return 1.0
    # Contingency table via pair counting (avoids O(n²) boolean matrix)
    ka = int(a.max()) + 1
    kb = int(b.max()) + 1
    contingency = np.zeros((ka, kb), dtype=np.int64)
    for ai, bi in zip(a, b):
        contingency[ai, bi] += 1
    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) // 2
    sum_comb_c  = int(comb2(contingency).sum())
    sum_comb_a  = int(comb2(contingency.sum(axis=1)).sum())
    sum_comb_b  = int(comb2(contingency.sum(axis=0)).sum())
    total_pairs = n * (n - 1) // 2
    expected    = sum_comb_a * sum_comb_b / max(1, total_pairs)
    maximum     = (sum_comb_a + sum_comb_b) / 2.0
    denom = maximum - expected
    return (sum_comb_c - expected) / denom if denom > 0 else 1.0


def _stability_choose_k_gen(S: types.SimpleNamespace) -> Generator[None, None, None]:
    """Choose k via bootstrap stability using Adjusted Rand Index.

    ARI corrects for chance (expected value 0 for random labels, 1 for perfect
    agreement) so it does not systematically favour small k the way the plain
    Rand index does.  We pick the k with the highest mean ARI across
    _K_BOOTSTRAP_B subsampled replicates.  Writes S.cluster_k and
    S.k_chosen_n_frames when done.  Clears S.k_choosing on exit.
    """
    try:
        mat_rows = [list(row) for row in S.pairwise_rmsd_mat]  # snapshot
        n = len(mat_rows)
        k_max = min(_K_MAX, n // 5)
        if k_max < 2:
            return

        mat = np.zeros((n, n))
        for i, row in enumerate(mat_rows):
            mat[i, : len(row)] = row
            yield
        mat = mat + mat.T

        sub_size = max(4, int(0.8 * n))
        stabilities: dict[int, float] = {}

        for k in range(2, k_max + 1):
            labels_ref = _kmedoids(mat, k)
            yield
            scores: list[float] = []
            for _ in range(_K_BOOTSTRAP_B):
                idx = np.sort(np.random.choice(n, size=sub_size, replace=False))
                sub_mat = mat[np.ix_(idx, idx)]
                labels_sub = _kmedoids(sub_mat, k)
                scores.append(_adjusted_rand_index(labels_ref[idx], labels_sub))
                yield
            stabilities[k] = float(np.mean(scores))

        chosen_k = max(stabilities, key=lambda k: stabilities[k])
        S.cluster_k         = chosen_k
        S.k_chosen_n_frames = n
        _LOG.debug("k-stability (ARI): chose k=%d from %s", chosen_k, stabilities)
    finally:
        S.k_choosing = False


def _cluster_gen(S: types.SimpleNamespace) -> Generator[None, None, None]:
    """Assign cluster labels using the current S.cluster_k via k-medoids.

    Snapshots the pairwise matrix at entry, builds the symmetric form (yielding
    per row), then runs _kmedoids (yielding once after each k-1 farthest-point
    init step and once per PAM update per cluster).  Writes S.cluster_labels and
    S.cluster_n_frames_cached atomically at the end.  Clears S.cluster_computing.
    """
    try:
        k = S.cluster_k
        if k is None:
            return
        mat_rows = [list(row) for row in S.pairwise_rmsd_mat]  # snapshot
        n = len(mat_rows)
        if n < k:
            return
        mat = np.zeros((n, n))
        for i, row in enumerate(mat_rows):
            mat[i, : len(row)] = row
            yield
        mat = mat + mat.T

        # Greedy farthest-point initialisation
        medoids: list[int] = [0]
        for _ in range(k - 1):
            dists = mat[:, medoids].min(axis=1)
            medoids.append(int(np.argmax(dists)))
            yield
        # PAM update
        labels = np.zeros(n, dtype=int)
        for _ in range(50):
            labels = np.argmin(mat[:, medoids], axis=1)
            yield
            new_medoids = list(medoids)
            changed = False
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                costs = mat[np.ix_(members, members)].sum(axis=1)
                new_med = int(members[np.argmin(costs)])
                if new_med != new_medoids[c]:
                    changed = True
                new_medoids[c] = new_med
                yield
            medoids = new_medoids
            if not changed:
                break

        # Per-frame distance to assigned medoid
        medoid_arr = np.array(medoids)
        dist_to_med = mat[np.arange(n), medoid_arr[labels]]
        mean_d, std_d = float(dist_to_med.mean()), float(dist_to_med.std())
        threshold = mean_d + _OUTLIER_STD_THRESHOLD * std_d
        outlier_mask = (dist_to_med > threshold).tolist()

        # k×k inter-medoid distance matrix
        intercluster = mat[np.ix_(medoids, medoids)]

        S.cluster_labels            = labels.tolist()
        S.cluster_medoids           = list(medoids)
        S.cluster_outlier_mask      = outlier_mask
        S.cluster_intercluster_dists = intercluster.tolist()
        S.cluster_n_frames_cached   = n
    finally:
        S.cluster_computing = False


def _poll_cluster(S: types.SimpleNamespace, runner: TaskRunner) -> None:
    n = S.n_pos_frames
    if n < _K_MIN_FRAMES:
        return
    # Trigger k-selection when we have no k yet or frames have grown significantly.
    # Don't start a new k-selection while a previous one (or label assignment) is running.
    needs_new_k = S.cluster_k is None or n >= S.k_chosen_n_frames * _K_GROWTH_FACTOR
    if needs_new_k and not S.k_choosing and not S.cluster_computing:
        S.k_choosing = True
        runner.submit(_stability_choose_k_gen(S))
        return  # wait for k to settle before assigning labels
    # Assign labels whenever frame count changes and k is stable
    if (S.cluster_k is not None
            and not S.cluster_computing
            and not S.k_choosing
            and n != S.cluster_n_frames_cached):
        S.cluster_computing = True
        runner.submit(_cluster_gen(S))


def _raster_state0_replica(
    S: types.SimpleNamespace, spark_w: int, n_rows: int, total_iters: int, has_256: bool
) -> list[list[tuple[str, int]]]:
    """Scatter plot: x=time, y=physical replica occupying state 0, colour=green."""
    grid: list[list[tuple[str, int]]] = [[(" ", 0)] * spark_w for _ in range(n_rows)]
    if total_iters == 0 or not S.pos_frame_replicas:
        return grid
    attr = curses.color_pair(_CP_GREEN)
    for frame_iter, replica in zip(S.pos_frame_iters, S.pos_frame_replicas):
        col = min(spark_w - 1, int(frame_iter / total_iters * spark_w))
        if 0 <= replica < n_rows:
            grid[replica][col] = ("█", attr)
    return grid


def _bar_cluster_occupancy(
    S: types.SimpleNamespace, spark_w: int, n_rows: int, total_iters: int, has_256: bool
) -> list[list[tuple[str, int]]]:
    """Bar chart: x=cluster (+outlier), bar height=fraction of state-0 frames."""
    grid: list[list[tuple[str, int]]] = [[(" ", 0)] * spark_w for _ in range(n_rows)]
    k = S.cluster_k
    if not k or not S.cluster_labels:
        return grid
    labels       = S.cluster_labels
    outlier_mask = S.cluster_outlier_mask
    total        = len(labels)
    n_outliers   = sum(outlier_mask) if outlier_mask else 0
    # One bar per cluster, plus a grey outlier bar if any exist
    bars: list[tuple[int, int]] = [
        (sum(1 for i, l in enumerate(labels) if l == c and not (outlier_mask and outlier_mask[i])),
         _raster_color_attr(c, has_256))
        for c in range(k)
    ]
    if n_outliers:
        bars.append((n_outliers, curses.color_pair(_CP_GREY) | curses.A_DIM))
    n_bars = len(bars)
    for b, (count, attr) in enumerate(bars):
        col_lo = b * spark_w // n_bars
        col_hi = (b + 1) * spark_w // n_bars
        if col_lo >= col_hi:
            continue
        bar_height = count / total * n_rows
        full_rows  = int(bar_height)
        partial    = bar_height - full_rows
        for col in range(col_lo, col_hi):
            for row_from_bot in range(full_rows):
                grid[n_rows - 1 - row_from_bot][col] = ("█", attr)
            if partial > 0 and full_rows < n_rows:
                grid[n_rows - 1 - full_rows][col] = (_BAR_CHARS[max(1, round(partial * 8))], attr)
    return grid


def _raster_cluster_trajectory(
    S: types.SimpleNamespace, spark_w: int, n_rows: int, total_iters: int, has_256: bool
) -> list[list[tuple[str, int]]]:
    """Braille scatter: x=time, y=1D-MDS cluster position, colour=cluster (grey=outlier).

    Outlier frames are positioned via softmin-weighted average of cluster MDS
    positions using their distances to each medoid, so they appear between the
    clusters they are structurally intermediate to rather than at an arbitrary row.
    """
    grid: list[list[tuple[str, int]]] = [[(" ", 0)] * spark_w for _ in range(n_rows)]
    k = S.cluster_k
    if not k or not S.cluster_labels or total_iters == 0:
        return grid
    dists = S.cluster_intercluster_dists
    if not dists:
        return grid

    pos = _mds_1d(np.array(dists))
    pos_min, pos_max = float(pos.min()), float(pos.max())
    norm = (pos - pos_min) / (pos_max - pos_min) if pos_max > pos_min else np.full(k, 0.5)

    outlier_mask = S.cluster_outlier_mask
    medoids      = S.cluster_medoids
    mat_rows     = S.pairwise_rmsd_mat
    grey_attr    = curses.color_pair(_CP_GREY) | curses.A_DIM

    # Braille sub-cell resolution: 2 sub-cols × 4 sub-rows per terminal cell
    data_h = n_rows * 4
    data_w = spark_w * 2
    bits_grid: list[list[int]] = [[0] * spark_w for _ in range(n_rows)]
    attr_grid: list[list[int]] = [[0] * spark_w for _ in range(n_rows)]

    for i, (frame_iter, cluster) in enumerate(zip(S.pos_frame_iters, S.cluster_labels)):
        is_outlier = bool(outlier_mask[i]) if outlier_mask else False

        if is_outlier and medoids:
            # mat_rows is lower-triangular: dist(a,b) = mat_rows[max(a,b)][min(a,b)]
            dists_to_meds = np.array([
                0.0 if i == m else (float(mat_rows[i][m]) if m < i else float(mat_rows[m][i]))
                for m in medoids
            ])
            scale = float(dists_to_meds.mean()) or 1.0
            weights = np.exp(-dists_to_meds / scale)
            weights /= weights.sum()
            y_frac = float((weights * norm).sum())
        else:
            y_frac = float(norm[cluster])

        # High y_frac → top (row 0); low → bottom
        dc = min(data_w - 1, int(frame_iter / total_iters * data_w))
        dr = int(round((1.0 - y_frac) * (data_h - 1)))
        dr = max(0, min(data_h - 1, dr))
        tr, tc = dr // 4, dc // 2
        bits_grid[tr][tc] |= _BRAILLE_BITS[dc % 2][dr % 4]

        attr = grey_attr if is_outlier else _raster_color_attr(cluster, has_256)
        if attr_grid[tr][tc] == 0 or attr_grid[tr][tc] == grey_attr:
            attr_grid[tr][tc] = attr

    for r in range(n_rows):
        for c in range(spark_w):
            b = bits_grid[r][c]
            if b:
                grid[r][c] = (chr(0x2800 + b), attr_grid[r][c])
    return grid


# ── Mode registry ──────────────────────────────────────────────────────────────

_SPARK_MODES: list[SparkMode] = [
    SparkMode(
        name="reduced U", unit="kJ/mol",
        help_text=(
            "Energy of whichever replica currently occupies state 0, "
            "converted from dimensionless reduced potential (kT) to kJ/mol. "
            "Should fluctuate around a stable mean once the system is equilibrated; "
            "a sustained drift indicates incomplete equilibration. "
            "Yellow reference line: energy at the first trajectory frame."
        ),
        get_data=_data_ground_u, get_ref=_ref_first,
        uses_history=True,
    ),
    SparkMode(
        name="volume", unit="nm\u00b3",
        help_text=(
            "Box volume (nm\u00b3) of the replica currently in state 0. "
            "Should fluctuate around a stable mean under the NPT barostat; "
            "a drifting mean indicates the density has not yet equilibrated. "
            "Yellow reference line: running mean of all observed volumes \u2014 "
            "the best current estimate of the barostat equilibrium volume."
        ),
        get_data=_data_ground_vol, get_ref=_ref_mean,
        uses_history=True,
    ),
    SparkMode(
        name="online \u0394F", unit="kT",
        help_text=(
            "Total free energy difference from state 0 to state N-1, "
            "estimated by offline MBAR and updated at checkpoint intervals. "
            "Values should converge quickly and remain stable; "
            "large changes late in the run suggest insufficient sampling. "
            "No reference line."
        ),
        get_data=_data_fe, get_ref=_ref_none,
        multistate=True, uses_history=True,
    ),
    SparkMode(
        name="RMSD", unit="\u00c5",
        help_text=(
            "Kabsch RMSD (\u00c5) of the state-0 replica vs the very first trajectory frame, "
            "using optimal rigid-body superposition (translation + rotation). "
            "Shows structural drift from the starting conformation. "
            "Not a good convergence metric for flexible systems: "
            "a flexible peptide that re-visits its starting conformation looks converged "
            "even if large regions of conformational space are unexplored. "
            "No reference line."
        ),
        get_data=_data_rmsd, get_ref=_ref_none,
        needs_atoms=True,
    ),
    SparkMode(
        name="min-RMSD", unit="\u00c5",
        help_text=(
            "For each frame: minimum Kabsch RMSD (\u00c5) to any previously seen frame. "
            "High early on when every structure is novel; "
            "converges to a thermal noise floor once all accessible conformations "
            "have been visited at least once. "
            "Yellow reference line: all-pairs minimum RMSD \u2014 the closest any two frames "
            "have ever been, i.e. the thermal noise floor. "
            "When the curve plateaus at the reference line, the simulation has exhausted "
            "conformational space and is only revisiting structures within thermal fluctuations."
        ),
        get_data=_data_min_rmsd, get_ref=_ref_min_pairwise,
        needs_atoms=True,
    ),
    SparkMode(
        name="max-RMSD", unit="\u00c5",
        help_text=(
            "For each frame: maximum Kabsch RMSD (\u00c5) to any previously seen frame \u2014 "
            "the structural eccentricity of that frame within the explored ensemble. "
            "High when a frame is far from all known structures; "
            "stabilises once the ensemble diameter is fully covered. "
            "Unlike min-RMSD (which measures novelty) this measures reach: "
            "a central frame has small max-RMSD even if it is novel. "
            "Yellow reference line: all-pairs maximum RMSD observed so far \u2014 "
            "the structural diameter of the trajectory, which the curve converges toward."
        ),
        get_data=_data_max_rmsd, get_ref=_ref_max_pairwise,
        needs_atoms=True,
    ),
    SparkMode(
        name="centroid RMSD", unit="\u00c5",
        help_text=(
            "Kabsch RMSD (\u00c5) of each state-0 frame to the centroid (mean structure) of "
            "the entire trajectory. All frames are first superposed onto frame 0; the "
            "centroid is then the coordinate-wise mean of the aligned ensemble. "
            "Frames far from the centroid represent structural outliers; "
            "a flat, low plateau indicates tight conformational clustering around a "
            "single dominant structure, while large excursions suggest significant "
            "conformational heterogeneity. "
            "Yellow reference line: mean centroid RMSD across all frames."
        ),
        get_data=_data_centroid_rmsd, get_ref=_ref_mean,
        needs_atoms=True, poll=_poll_centroid,
    ),
    SparkMode(
        name="medoid RMSD", unit="\u00c5",
        help_text=(
            "Kabsch RMSD (\u00c5) of each state-0 frame to the medoid \u2014 the single observed "
            "frame that minimises the sum of pairwise RMSDs to all other frames. "
            "Unlike the coordinate-wise mean (which may be unphysical when multiple "
            "conformational states are visited), the medoid is always a real trajectory "
            "frame. Frames in the same conformational cluster as the medoid appear near "
            "zero; frames in a different cluster stand out as high-RMSD outliers, making "
            "this a sensitive indicator of multi-state behaviour. "
            "Yellow reference line: mean medoid RMSD across all frames."
        ),
        get_data=_data_medoid_rmsd, get_ref=_ref_mean,
        needs_atoms=True, poll=_poll_medoid,
    ),
    SparkMode(
        name="RMSD ACF", unit="\u00c5",
        help_text=(
            "Mean Kabsch RMSD (\u00c5) between all pairs of state-0 frames separated by a "
            "given lag time. X-axis is logarithmic lag time (not simulation time), "
            "so each decade of lag gets equal visual space \u2014 the rise from zero is "
            "clearly resolved even when the plateau spans orders of magnitude longer. "
            "Rises from 0 at lag=0 and plateaus at the structural variance of the ensemble. "
            "The lag at which it plateaus is the conformational decorrelation time. "
            "Faster decorrelation indicates more effective conformational sampling. "
            "Yellow reference line: estimated plateau value (mean of the highest-lag half) \u2014 "
            "converges toward the true ensemble structural variance as more frames accumulate."
        ),
        get_data=_data_acf, get_ref=_ref_acf_plateau,
        needs_atoms=True, individual_dots=True, x_transform=math.log10,
        get_grid=_grid_acf, xaxis_m_fn=_xaxis_m_acf,
    ),
    SparkMode(
        name="replica over time", unit="",
        help_text=(
            "Which physical replica is occupying thermodynamic state 0 at each sampled "
            "frame (braille scatter, individual dots).  X-axis is simulation time; "
            "Y-axis is physical replica index.  Good REMD mixing produces a scatter "
            "spread across all replica indices over time.  A single index dominating "
            "long stretches indicates a replica is stuck at state 0."
        ),
        get_data=_data_state0_replica,
        get_ref=lambda *_: None,
        individual_dots=True,
        y_fixed_range=lambda S: (-0.5, S.n_replicas - 0.5),
    ),
    SparkMode(
        name="cluster occupancy", unit="",
        help_text=(
            "Fraction of state-0 frames in each conformational cluster (k-medoids on "
            "pairwise RMSD).  Each bar is one cluster; bar height is the fraction of "
            "frames assigned to it.  Unequal bars indicate metastability.  k is chosen "
            "by bootstrap stability using Adjusted Rand Index (corrected for chance, so "
            "it does not systematically favour small k): for each candidate k, 20 "
            "subsampled replicates are clustered and their ARI vs the reference is "
            "averaged; the k with the highest mean ARI wins.  k is recomputed when "
            "frame count grows by _K_GROWTH_FACTOR; press 'k' to force a recompute."
        ),
        get_data=lambda S, _: ([], []), get_ref=lambda *_: None,
        needs_atoms=True, raster_fn=_bar_cluster_occupancy, poll=_poll_cluster,
    ),
    SparkMode(
        name="cluster trajectory", unit="",
        help_text=(
            "Which conformational cluster state-0 occupies at each sampled frame "
            "(x=time, y=cluster, colour=cluster, grey=outlier).  Clusters are "
            "positioned on the y-axis by 1D MDS of the inter-medoid RMSD matrix, "
            "so vertical spacing reflects structural distance between basins.  "
            "Grey dots are outliers: frames whose RMSD to their assigned medoid "
            f"exceeds the within-cluster mean + {_OUTLIER_STD_THRESHOLD:.0f}σ.  "
            "Persistent occupation of one cluster with rare transitions indicates "
            "metastability; rapid switching indicates good conformational sampling."
        ),
        get_data=lambda S, _: ([], []), get_ref=lambda *_: None,
        needs_atoms=True, raster_fn=_raster_cluster_trajectory, poll=_poll_cluster,
    ),
]


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
                    S.acc_sum.copy(),
                    S.prop_sum.copy(),
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
        S.history_bulk_loading = False


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
                _LOG.debug("skipping RMSD frame at iter %d", abs_i, exc_info=True)
            yield
    finally:
        S.rmsd_computing = False


def _state0_scan_gen(
    reader: SimulationReader,
    S: types.SimpleNamespace,
) -> Generator[None, None, None]:
    """Bulk-read replica_states for all unseen iterations in one NetCDF slice.

    Reading states[:, :] as a single 2-D array is orders of magnitude faster
    than one replica_states() call per iteration.  Yields after the read and
    after the numpy extraction so the render loop stays live.  Clears
    S.state0_scanning on exit.
    """
    try:
        start = S.state0_scan_iter
        end   = S.display_iter + 1
        if start >= end:
            return
        states = reader.replica_states_range(start, end)   # (n_iters, n_replicas)
        yield
        # For each iteration, argmax on the boolean mask gives the replica in state 0.
        replica_at_state0 = (states == 0).argmax(axis=1)   # (n_iters,)
        S.state0_replica_iters.extend(range(start, end))
        S.state0_replica_vals.extend(replica_at_state0.tolist())
        S.state0_scan_iter = end
        yield
    finally:
        S.state0_scanning = False



def _fetch_scrub_data(reader: SimulationReader, S: types.SimpleNamespace) -> None:
    """Read/compute all per-iteration data for the pinned scrub position."""
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
        _, state_counts, half_trips_arr, last_extreme, acc_sum, prop_sum = checkpoints[idx]
        state_counts = state_counts.copy()
        half_trips   = list(half_trips_arr)
        last_extreme = list(last_extreme)
        acc_sum      = acc_sum.copy()
        prop_sum     = prop_sum.copy()
        read_from    = checkpoints[idx][0] + 1
    else:
        state_counts = np.zeros((n_replicas, n_states), dtype=int)
        half_trips   = [0] * n_replicas
        last_extreme = [None] * n_replicas
        acc_sum      = np.zeros((n_states, n_states))
        prop_sum     = np.zeros((n_states, n_states))
        read_from    = 0

    if read_from <= si:
        acc_delta, prop_delta = reader.exchange_counts(slice(read_from, si + 1))
        acc_sum  += acc_delta
        prop_sum += prop_delta
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

    S.scrub_acc_sum      = acc_sum
    S.scrub_prop_sum     = prop_sum
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
        _LOG.warning("error computing timing/ETA string", exc_info=True)

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


def _render_topology(stdscr: curses.window, S: types.SimpleNamespace) -> None:
    """Render the topology molecule table in place of the exchange matrix.

    Spills into a second (or third…) column when the row count would exceed the
    available terminal height, up to as many columns as the terminal width allows.
    """
    from itertools import groupby

    mols: list[tuple[str, list[int]]] = getattr(S, "topology_molecules", [])
    n_total = sum(len(m[1]) for m in mols)

    _addstr(stdscr, f"  Topology  {n_total:,} atoms  (0-indexed, copy ranges verbatim)\n\n")

    if not mols:
        _addstr(stdscr, "    (not yet loaded)\n")
        return

    # Build all rows up-front so column widths can be computed globally.
    rows: list[tuple[str, str, int, str]] = []  # (range_str, formula, count, note)
    for formula, group in groupby(mols, key=lambda m: m[0]):
        group       = list(group)
        n           = len(group)
        all_indices = sorted(idx for _, mol in group for idx in mol)
        atoms_each  = len(group[0][1])
        note        = f"({atoms_each} atoms)" if n == 1 else f"({atoms_each} atoms each)"
        rows.append((_format_index_ranges(all_indices), formula, n, note))

    range_w   = max(len(r[0]) for r in rows)
    formula_w = max(len(r[1]) for r in rows)
    count_w   = max(len(str(r[2])) for r in rows)
    note_w    = max(len(r[3]) for r in rows)

    # Full width of one rendered column (including the leading "    " prefix).
    # "    " + range + "   " + formula + " × " + count + "  " + note
    cell_w  = 4 + range_w + 3 + formula_w + 3 + count_w + 2 + note_w
    col_sep = 4  # spaces between columns

    # Rows available for data: terminal height minus the fixed overhead above
    # (2 lines for the iteration header + 2 for the topology header) and below
    # (1 trailing blank + 1 for the input bar drawn at the last row).
    _term_rows, _term_cols = stdscr.getmaxyx()
    avail_rows = max(1, _term_rows - 6)

    # Minimum columns needed to fit within avail_rows; cap at what the terminal
    # width can accommodate.
    n_cols = math.ceil(len(rows) / avail_rows) if len(rows) > avail_rows else 1
    while n_cols > 1 and n_cols * cell_w + (n_cols - 1) * col_sep > _term_cols:
        n_cols -= 1
    n_cols = max(1, n_cols)

    rows_per_col = math.ceil(len(rows) / n_cols)
    columns      = [rows[i : i + rows_per_col] for i in range(0, len(rows), rows_per_col)]

    for row_idx in range(rows_per_col):
        for col_idx, col_rows in enumerate(columns):
            if col_idx > 0:
                _addstr(stdscr, " " * col_sep)
            if row_idx < len(col_rows):
                range_str, formula, count, note = col_rows[row_idx]
                _addstr(stdscr, f"    {range_str:>{range_w}}   ", curses.A_BOLD)
                # Pad note on all but the last column so subsequent columns align.
                note_str = f"{note:<{note_w}}" if col_idx < len(columns) - 1 else note
                _addstr(stdscr, f"{formula:<{formula_w}} × {count:>{count_w}}  {note_str}")
            elif col_idx < len(columns) - 1:
                _addstr(stdscr, " " * cell_w)  # blank cell to keep later columns aligned
        _addstr(stdscr, "\n")

    _addstr(stdscr, "\n")


def _render(stdscr: curses.window, S: types.SimpleNamespace, gradient: list[int] | None) -> None:
    """Redraw the entire screen from S. Pure display, no nc access."""
    stdscr.erase()

    _addstr(stdscr, f"{S.title}\n", curses.A_BOLD)

    if S.error_text:
        _addstr(stdscr, f"Unexpected error:\n{S.error_text}")
        stdscr.refresh()
        return

    if S.waiting:
        wall     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spin_ch  = _SPINNER[int(time.monotonic() * 4) % len(_SPINNER)]
        _addstr(stdscr, f"  Wall: {wall}\n\n")
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
    _addstr(stdscr, f"  Iteration: {_iter_str}    Time: {_sim_str}{S.timing_str}\n\n")

    if S.rmsd_input_active:
        _render_topology(stdscr, S)
        # Draw input bar at the last row and refresh, then skip the matrix body.
        _term_rows, _term_cols = stdscr.getmaxyx()
        _hints     = "   Enter confirm   Esc cancel"
        _prefix    = "  Atom selection: "
        _input_str = S.rmsd_input_buf + "\u2588"
        _err_str   = f"  ✗ {S.rmsd_sel_error}" if S.rmsd_sel_error else ""
        _lhs       = _prefix + _input_str
        _mid       = ("  e.g. 0-64,67,200-300" if not S.rmsd_sel_error else "")
        _rhs       = _mid + _err_str + _hints
        gap        = max(1, _term_cols - 1 - len(_lhs) - len(_rhs))
        bar        = (_lhs + " " * gap + _rhs).ljust(_term_cols - 1)
        try:
            stdscr.move(_term_rows - 1, 0)
            stdscr.addstr(bar, curses.A_REVERSE)
        except curses.error:
            pass
        if S.rmsd_sel_error:
            _err_attr = curses.color_pair(_CP_RED) | curses.A_BOLD | curses.A_REVERSE
            try:
                stdscr.move(_term_rows - 1, len(_prefix))
                stdscr.addstr(_input_str[:max(0, _term_cols - 1 - len(_prefix))], _err_attr)
                _err_col = len(bar) - len(_hints) - len(_err_str)
                if 0 <= _err_col < _term_cols - 1:
                    stdscr.move(_term_rows - 1, _err_col)
                    stdscr.addstr(_err_str[:max(0, _term_cols - 1 - _err_col)], _err_attr)
            except curses.error:
                pass
        stdscr.refresh()
        return

    col_w    = 7
    matrix_w = 2 + 5 + n_states * col_w
    sep      = "   "

    total_iters = (S.n_iterations - 1) if S.n_iterations else (display_iter + 1)
    _term_rows, _term_cols = stdscr.getmaxyx()
    spark_w = max(10, _term_cols - matrix_w - len(sep) - 1)

    # Derive sparkline data, reference line, and axis labels from the mode registry
    mode = S.sparkline_mode
    m    = _SPARK_MODES[mode]

    sp_iters, sp_vals = m.get_data(S, spark_w)
    sp_ref_val        = m.get_ref(S, sp_vals)

    # x-axis labels and grid_total (must precede n_per_unit which uses grid_total)
    total_ns = total_iters * S.n_steps * S.timestep_ps / 1000
    if m.get_grid is not None:
        grid_total, xaxis_l, xaxis_r = m.get_grid(S, sp_iters, total_iters)
    else:
        grid_total = total_iters
        xaxis_l = "0 ns"
        xaxis_r = f"{total_ns:.1f} ns"

    # Average data points per rendered dot.  Divide by the number of non-empty
    # bins (dots that actually get drawn), not total possible positions — dots
    # with no data are never shown so they shouldn't count in the denominator.
    n_per_unit: float | None = None
    xaxis_m = ""
    if m.raster_fn is None:
        if not m.individual_dots and spark_w > 0 and sp_vals and grid_total > 0:
            _use_braille = len(sp_vals) >= spark_w
            _n_bins = 2 * spark_w if _use_braille else spark_w
            _filled = len({min(int(it / grid_total * _n_bins), _n_bins - 1) for it in sp_iters})
            if _filled > 0:
                n_per_unit = len(sp_vals) / _filled
        if m.xaxis_m_fn is not None:
            xaxis_m = m.xaxis_m_fn(S)
        elif n_per_unit is not None and n_per_unit > _SPARK_DOTS_THRESHOLD:
            xaxis_m = f"~{_fmt_2sf(n_per_unit)}/dot"

    sp_prefix = (f"State 0\u2192{n_states-1} {m.name}" if m.multistate
                 else f"State 0 {m.name}")

    if m.raster_fn is not None:
        has_256 = gradient is not None
        spark_grid = m.raster_fn(S, spark_w, n_states, total_iters, has_256)
        spark_ymin = spark_ymax = float("nan")  # unused for raster
        if m.needs_atoms and S.solute_atom_sel is None:
            spark_title = f"{sp_prefix}  (press 'a' to set atom selection)"
        elif m.poll is None:
            # No background computation — replica scatter just shows what we have
            spark_title = (
                f"{sp_prefix}  {S.n_pos_frames} frames"
                if S.pos_frame_replicas else f"{sp_prefix}  (accumulating...)"
            )
        elif S.k_choosing:
            spark_title = f"{sp_prefix}  (choosing k…  {S.n_pos_frames} frames)"
        elif S.cluster_computing:
            spark_title = f"{sp_prefix}  k={S.cluster_k}  (clustering…)"
        elif S.cluster_labels:
            n_outliers = sum(S.cluster_outlier_mask) if S.cluster_outlier_mask else 0
            outlier_str = f"  {n_outliers} outliers" if n_outliers else ""
            if S.cluster_intercluster_dists and S.cluster_k and S.cluster_k > 1:
                mat = np.array(S.cluster_intercluster_dists)
                upper = mat[np.triu_indices(S.cluster_k, k=1)]
                d_min, d_max = float(upper.min()), float(upper.max())
                dist_str = f"  d={d_min:.1f}–{d_max:.1f}Å"
            else:
                dist_str = ""
            spark_title = (
                f"{sp_prefix}  k={S.cluster_k}{dist_str}"
                f"  {len(S.cluster_labels)} frames{outlier_str}"
            )
        elif S.n_pos_frames < _K_MIN_FRAMES:
            spark_title = f"{sp_prefix}  (need {_K_MIN_FRAMES} frames, have {S.n_pos_frames})"
        else:
            spark_title = f"{sp_prefix}  (accumulating...)"
        # Scrub cursor on raster — same logic as sparkline
        if _scrubbing and grid_total > 0:
            cur_col  = max(0, min(spark_w - 1, int(S.scrub_iter / grid_total * spark_w)))
            cur_attr = curses.color_pair(_CP_BLUE)
            for r in range(n_states):
                if spark_grid[r][cur_col][0] == " ":
                    spark_grid[r][cur_col] = ("│", cur_attr)
    else:
        _yfix = m.y_fixed_range(S) if m.y_fixed_range is not None else (None, None)
        spark_grid, spark_ymin, spark_ymax = _build_sparkline_grid(
            sp_iters, sp_vals,
            grid_total, n_states, spark_w,
            cp_mean=curses.color_pair(_CP_GREEN),
            cp_range=curses.color_pair(_CP_GREY),
            cp_ref=curses.color_pair(_CP_DIM),
            ref_val=sp_ref_val,
            x_transform=m.x_transform,
            individual_dots=m.individual_dots,
            y_min_fixed=_yfix[0],
            y_max_fixed=_yfix[1],
        )
        # Scrub cursor: blue vertical bar only on empty cells (never overwrites data)
        if _scrubbing and not m.individual_dots and grid_total > 0:
            cur_col  = max(0, min(spark_w - 1, int(S.scrub_iter / grid_total * spark_w)))
            cur_attr = curses.color_pair(_CP_BLUE)
            for r in range(n_states):
                if spark_grid[r][cur_col][0] == " ":
                    spark_grid[r][cur_col] = ("│", cur_attr)
        if m.needs_atoms and S.solute_atom_sel is None:
            spark_title = f"{sp_prefix}  (press 'a' to set atom selection)"
        elif m.needs_atoms:
            n_frames_total = display_iter // S.pos_interval + 1
            if S.n_pos_frames < n_frames_total:
                pct = S.n_pos_frames / n_frames_total * 100
                spark_title = (
                    f"{sp_prefix}  (computing... {pct:.0f}%"
                    f"  {S.n_pos_frames}/{n_frames_total} frames)"
                )
            elif sp_vals:
                spark_title = f"{sp_prefix}  [{spark_ymin:.4g}, {spark_ymax:.4g}] {m.unit}"
            else:
                spark_title = f"{sp_prefix}  (accumulating...)"
        elif S.history_computing and m.uses_history and S.history_bulk_loading:
            pct = (S.last_mixed_iter + 1) / (display_iter + 1) * 100
            spark_title = f"{sp_prefix}  (loading history... {pct:.0f}%)"
        elif sp_vals:
            spark_title = f"{sp_prefix}  [{spark_ymin:.4g}, {spark_ymax:.4g}] {m.unit}"
        else:
            spark_title = f"{sp_prefix}  (accumulating...)"

    # Title rows — spark title wraps freely on the right; the matrix header is
    # pinned to the last title line so it always sits directly above the column
    # headers regardless of how many lines the title takes.
    matrix_hdr = "  Exchange acceptance (%):"
    _title_lines = textwrap.wrap(spark_title, width=spark_w) or [""]
    for _tl in _title_lines[:-1]:
        _addstr(stdscr, " " * matrix_w + sep + _tl + "\n", curses.A_BOLD)
    _addstr(stdscr, matrix_hdr, curses.A_BOLD)
    _addstr(stdscr, " " * (matrix_w - len(matrix_hdr)) + sep)
    _addstr(stdscr, _title_lines[-1] + "\n", curses.A_BOLD)

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
    _render_acc  = S.scrub_acc_sum  if _scrubbing else S.acc_sum
    _render_prop = S.scrub_prop_sum if _scrubbing else S.prop_sum
    for i in range(n_states):
        _addstr(stdscr, f"  {i:>5}")
        for j in range(n_states):
            if i == j:
                _addstr(stdscr, " " * col_w)
            else:
                prop = _render_prop[i, j]
                rate = _render_acc[i, j] / prop * 100 if prop > 0 else float("nan")
                text = f"{'nan':>7}" if math.isnan(rate) else f"{rate:>7.1f}"
                _addstr(stdscr, text, curses.color_pair(_rate_colour_pair(rate, gradient)))
        _addstr(stdscr, sep)
        for col in range(spark_w):
            char, attr = spark_grid[i][col]
            _addstr(stdscr, char, attr)
        _addstr(stdscr, "\n")

    if S.show_spark_help:
        help_text = m.help_text
        if n_per_unit is not None and n_per_unit > _SPARK_DOTS_THRESHOLD:
            help_text += " Green dots are means; white dots are min/max range."
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

    # ── Simulation info box (drawn to the right of the replica table) ─────────
    # Table width: 2+7+2+5+2+18+2+10+2+13+2+visits_w+2+5 = 72+visits_w
    _table_w   = 72 + visits_w
    _box_avail = _term_cols - 1 - _table_w   # chars available for the box

    _iter_ps = S.n_steps * S.timestep_ps
    _pos_ps  = S.pos_interval * S.n_steps * S.timestep_ps

    def _fmt_ps(ps: float) -> str:
        return f"{_fmt_2sf(ps / 1000)} ns" if ps >= 1000 else f"{_fmt_2sf(ps)} ps"

    _sim_info: list[tuple[str, str]] = [
        ("Replicas",    str(n_replicas)),
        ("States",      str(n_states)),
        ("Ref. temp.",  f"{S.ref_temp_k:.4g} K"),
        ("Timestep",    f"{S.timestep_ps:.4g} ps"),
        ("Steps/iter",  f"{S.n_steps:,}"),
        ("Iter. time",  _fmt_ps(_iter_ps)),
    ]
    if S.pos_interval > 0:
        _sim_info.append(("Pos. every", _fmt_ps(_pos_ps)))
    if S.vel_interval > 0:
        _sim_info.append(("Vel. every", _fmt_ps(S.vel_interval * S.n_steps * S.timestep_ps)))
    if S.n_atoms > 0:
        _sim_info.append(("Atoms", f"{S.n_atoms:,}"))

    _lbl_w    = max(len(lbl) for lbl, _ in _sim_info)
    _val_w    = max(len(val) for _, val in _sim_info)
    _inner_w  = _lbl_w + 2 + _val_w   # "Label:  value"
    _box_w    = _inner_w + 4           # "│ " + inner + " │"
    _box_gap = 1
    _draw_box = _box_avail >= _box_w + _box_gap
    _box_col  = _table_w + _box_gap + (_box_avail - _box_w) // 2   # centred in available space

    _box_lines: list[str] = []
    if _draw_box:
        _box_lines.append("\u250c" + "\u2500" * (_inner_w + 2) + "\u2510")
        for _lbl, _val in _sim_info:
            _box_lines.append(f"\u2502 {_lbl:<{_lbl_w}}  {_val:>{_val_w}} \u2502")
        _box_lines.append("\u2514" + "\u2500" * (_inner_w + 2) + "\u2518")

    _addstr(stdscr, "\n")

    # Capture the row the header occupies, then render header/separator normally
    _box_start_row = stdscr.getyx()[0]

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

    # Draw info box to the right of the table using saved row position
    if _draw_box:
        for _bi, _bl in enumerate(_box_lines):
            _br = _box_start_row + _bi
            if _br >= _term_rows - 1:
                break
            try:
                stdscr.move(_br, _box_col)
                _avail = _term_cols - 1 - _box_col
                stdscr.addstr(_bl[:_avail])
            except curses.error:
                pass  # terminal too narrow; skip remaining box lines

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
    _prefix = _input_str = _hints = _err_str = ""  # set in rmsd_input_active branch
    if S.scrub_input_active:
        bar = f"  Jump to iteration: {S.scrub_input_buf}\u2588   (negative = from end)   Enter confirm   Esc cancel"
    elif S.rmsd_input_active:
        _hints     = "   Enter confirm   Esc cancel"
        _prefix    = "  Atom selection: "
        _input_str = S.rmsd_input_buf + "\u2588"
        _err_str   = f"  ✗ {S.rmsd_sel_error}" if S.rmsd_sel_error else ""
        _lhs       = _prefix + _input_str
        _mid       = ("  e.g. 0-64,67,200-300" if not S.rmsd_sel_error else "")
        _rhs       = _mid + _err_str + _hints
        gap        = max(1, _term_cols - 1 - len(_lhs) - len(_rhs))
        bar        = _lhs + " " * gap + _rhs
    else:
        _z_label = "z Live" if _scrubbing else "z Freeze"
        if m.needs_atoms:
            sel_hint = f" ({S.solute_sel_str})" if S.solute_sel_str else ""
            bar = f"  q Quit   s/S Sparkline   ? Explain   a Atoms{sel_hint}   k Re-cluster   w/e ±1   W/E ±5%   j Jump   {_z_label}"
        else:
            bar = f"  q Quit   s/S Sparkline   ? Explain   k Re-cluster   w/e ±1   W/E ±5%   j Jump   {_z_label}"
    bar = bar.ljust(_term_cols - 1)
    try:
        stdscr.move(_term_rows - 1, 0)
        stdscr.addstr(bar, curses.A_REVERSE)
    except curses.error:
        pass  # terminal too narrow to draw the full bar; truncation is acceptable
    if S.rmsd_input_active and S.rmsd_sel_error:
        _err_attr = curses.color_pair(_CP_RED) | curses.A_BOLD | curses.A_REVERSE
        try:
            # Redraw the input text in red+bold
            stdscr.move(_term_rows - 1, len(_prefix))
            stdscr.addstr(_input_str[:max(0, _term_cols - 1 - len(_prefix))], _err_attr)
            # Redraw the error string in red+bold (sits just before the key hints)
            _err_col = len(bar) - len(_hints) - len(_err_str)
            if 0 <= _err_col < _term_cols - 1:
                stdscr.move(_term_rows - 1, _err_col)
                stdscr.addstr(_err_str[:max(0, _term_cols - 1 - _err_col)], _err_attr)
        except curses.error:
            pass  # terminal too narrow to overlay error colouring; plain bar is still shown

    stdscr.refresh()


_LOOP_PERIOD = 0.05  # 50 ms — target wall time per main loop iteration


@app.default
def main(
    storage: Path = Path("cyclic_peptide.nc"),
    interval: float = 0.1,
    n_iterations: int | None = None,
    solute_n_atoms: int | None = None,
    log_file: Path | None = None,
    no_colorblind_mode: bool = False,
) -> None:
    """Monitor an REMD simulation.

    Pass --log-file monitor.log to enable debug timing output.
    Pass --no-colorblind-mode to use the classic red/green acceptance heatmap.
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
    _curses_main = lambda stdscr: _main(stdscr, reader, storage, interval, n_iterations, init_sel, no_colorblind_mode)
    try:
        curses.wrapper(_curses_main)
    except curses.error as e:
        if "terminfo" not in str(e):
            raise
        # Terminal type (e.g. xterm-ghostty) has no system terminfo entry; fall
        # back to xterm-256color which curses always knows about.
        _LOG.warning("TERM=%r has no terminfo entry; falling back to xterm-256color", os.environ.get("TERM"))
        os.environ["TERM"] = "xterm-256color"
        curses.wrapper(_curses_main)


def _main(
    stdscr: curses.window,
    reader: SimulationReader,
    storage: "Path",
    interval: float,
    n_iterations_arg: int | None,
    init_atom_sel: list[int] | None,
    no_colorblind_mode: bool = False,
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
    _key_points = _GRADIENT_KEY_POINTS_CLASSIC if no_colorblind_mode else _GRADIENT_KEY_POINTS
    gradient = _init_gradient_pairs(_key_points) if curses.COLORS >= 256 else None
    if curses.COLORS >= 256:
        _init_raster_pairs()

    S      = _init_state(reader, n_iterations_arg, init_atom_sel, interval, storage)
    try:
        S.topology_molecules = reader.parse_topology()
    except Exception:
        _LOG.warning("failed to parse system topology", exc_info=True)
        S.topology_molecules = []
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
                _LOG.warning("failed to fetch scrub data at iter %d; un-pinning", S.scrub_iter, exc_info=True)
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
            pass  # can't query terminal size (e.g. not a tty); skip resize check

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
                pass  # transient file error during refresh/poll; retry next interval
            last_poll = loop_start

        # ── Submit incremental tasks (idempotent: check flags before adding) ─
        if (
            not S.history_computing
            and not S.waiting
            and S.last_mixed_iter < S.display_iter
        ):
            S.history_computing = True
            runner.submit(_history_scan_gen(reader, S))

        if not S.state0_scanning and not S.waiting and S.state0_scan_iter <= S.display_iter:
            S.state0_scanning = True
            runner.submit(_state0_scan_gen(reader, S))

        if (
            not S.rmsd_computing
            and not S.waiting
            and S.solute_atom_sel is not None
            and S.pos_interval > 0
            and S.pos_scan_iter <= S.display_iter
        ):
            S.rmsd_computing = True
            runner.submit(_rmsd_gen(reader, S))

        for _m in _SPARK_MODES:
            if _m.poll:
                _m.poll(S, runner)

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

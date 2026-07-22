# ===========================================================================
# IUPAC-IUB atom (re)naming and per-frame canonicalisation
# ===========================================================================
#
# The topologies coming out of `process_storage()` have meaningless atom names
# (the OpenFF toolkit invented them when it wrote a heavy-atom-only PDB and then
# appended hydrogens) and an atom order in which each residue's hydrogens are
# scattered away from its heavy atoms. Per-atom `residue.name` is trustworthy,
# but the `Residue` *grouping*, atom names, and atom order are not.
#
# `canonicalize_iupac_names()` below rebuilds the peptide with:
#   * IUPAC-IUB atom names (Edsall et al. 1966; https://iupac.qmul.ac.uk/misc/ppep1.html),
#   * every residue's atoms contiguous, and
#   * prochiral / symmetric labels resolved *geometrically, per frame*.
#
# The last point matters because these are REST2 trajectories: high rungs run
# with weakened bonded terms, so a physical atom that is pro-R in one frame may
# have inverted by another. Since an mdtraj `Topology` binds one name to one
# index for the whole trajectory, "the atom named HB2" can only track the pro-R
# geometry if we permute the xyz columns of the affected atoms frame by frame.
# That is exactly what this code emits: a fixed IUPAC topology plus an xyz array
# whose columns are re-gathered per frame.
#
# Everything is derived from the bond graph and element identities alone (a
# residue's type comes from which template tiles its atoms, not from
# `residue.name`); nothing assumes the input atom order.

from dataclasses import dataclass
from typing import Never, Protocol, Self, TypeAlias, TypedDict, TypeVar

import mdtraj
import networkx
import numpy
from mdtraj.core import element as _mdtraj_element
from mdtraj.core.topology import Atom as _Atom
from networkx.algorithms import isomorphism as _isomorphism


class _AtomAttributes(TypedDict):
    """The attribute dict networkx stores on every node of these graphs.

    networkx exposes a node's data as its whole attribute mapping (``G.nodes[n]``),
    not a scalar; here that mapping holds exactly one key, the element symbol.
    """

    element: str


# networkx.Graph is Generic[node, node_data, edge_data], where node_data and
# edge_data are the per-node and per-edge *attribute mappings* (bound to
# Mapping[str, Any]), not scalar values. The two graphs used here differ only in
# node type (int atom indices for a real topology vs str atom names for a residue
# template); both give every node an _AtomAttributes dict and add edges with no
# attributes, so edge data is an always-empty mapping (dict[str, Never]). They are
# therefore one generic alias parameterised by node type.
_GraphNode = TypeVar("_GraphNode", int, str)

_LabelledGraph: TypeAlias = (
    "networkx.Graph[_GraphNode, _AtomAttributes, dict[str, Never]]"
)
"""An element-labelled graph over ``_GraphNode`` nodes (int atom indices for a real
topology, str atom names for a residue template). Every node carries an
``_AtomAttributes`` dict ``{"element": symbol}`` and edges carry no attributes.
Written as a forward-reference alias because networkx.Graph is not subscriptable at
runtime in this version; use it quoted and subscripted, e.g. ``"_LabelledGraph[int]"``."""


def _dihedral(
    a: numpy.ndarray,
    b: numpy.ndarray,
    c: numpy.ndarray,
    d: numpy.ndarray,
) -> numpy.ndarray:
    """Signed torsion angle theta(A,B,C,D), in degrees, vectorised over frames.

    Each argument is an ``(n_frames, 3)`` array of positions. The sign follows
    IUPAC-IUB rule 1.6: viewed along B->C, theta is positive when the front bond
    B-A must rotate *clockwise* to eclipse the rear bond C-D; equivalently, D
    lies ``+theta`` degrees clockwise of A about the B->C axis. (Verified against
    a hand-worked example: D placed +y off the B->C=+z axis with A along +x
    returns +90 deg.)
    """
    bc = c - b
    ba = a - b
    cd = d - c
    bc1 = bc / numpy.linalg.norm(bc, axis=-1, keepdims=True)
    # Compute the projections of ba and cd onto the plane perpendicular to bc1
    ba_proj = ba - numpy.sum(ba * bc1, axis=-1, keepdims=True) * bc1
    cd_proj = cd - numpy.sum(cd * bc1, axis=-1, keepdims=True) * bc1
    # Change of basis to 2d plane perpendicular to bc1 where x axis is ba_proj
    x_hat = ba_proj / numpy.linalg.norm(ba_proj, axis=-1, keepdims=True)
    z_hat = bc1
    y_hat = numpy.cross(z_hat, x_hat)
    cd_x = numpy.sum(x_hat * cd_proj, axis=-1)
    cd_y = numpy.sum(y_hat * cd_proj, axis=-1)
    # Angle from ba_proj to cd_proj in old basis is angle from x axis to cd in new basis
    return numpy.degrees(numpy.arctan2(cd_y, cd_x))


# --- IUPAC numbering rules ----------------------------------------------------
#
# Each rule resolves one prochiral / symmetric orbit and is a pure function of
# coordinates. ``reindex`` receives the per-frame coordinates of the rule's
# ``reference`` atoms (its geometric frame, never reordered) and its ``managed``
# atoms (the ones it permutes), and returns, for each frame, a permutation of
# ``0 .. n_managed - 1`` giving which managed atom's coordinates belong in each
# managed slot. Rules see no atom indices and no global state; the driver gathers
# the coordinates and applies the result. Each rule chooses its own managed layout.
# The driver resolves rules leaves-first (smallest managed set first), so every rule
# reads its atoms at their base positions and a parent subtree swap carries its
# children's results.


class _Rule(Protocol):
    """A geometric IUPAC-IUB numbering rule for one prochiral / symmetric orbit.

    ``reference_names`` are read only to build the geometric frame;
    ``managed_names`` are the atoms the rule permutes, in whatever layout that
    rule finds convenient. ``reindex`` maps their per-frame coordinates to a
    per-frame permutation of ``0 .. len(managed_names) - 1`` indexing the
    provided `managed_coords` argument.
    """

    # Read-only (property) members so a concrete @property satisfies them: a plain
    # annotated attribute is read-write, hence invariant, and rejects a property.
    @property
    def reference_names(self) -> tuple[str, ...]: ...

    @property
    def managed_names(self) -> tuple[str, ...]: ...

    def reindex(
        self,
        reference_coords: numpy.ndarray,
        managed_coords: numpy.ndarray,
    ) -> numpy.ndarray:
        """Return the ``(n_frames, n_managed)`` reordering of the managed atoms."""
        ...


@dataclass(frozen=True)
class _Methylene:
    """
    Implements IUPAC rule 2.2.2 case I for a methylene group.

    I paraphrase the relevant rules below:

        2.2.1: If, in a compound AB(P)(Q)C(E)(F)D, the sequence rule gives
        priorities A > (P, Q) and D > E > F, the "principal tortion angle" θ is
        ABCD and the branches from C are numbered CD: 1, CE: 2, CF: 3.

        2.2.2: D, E, and F are numbered in a clockwise sense when viewed in the
        direction B -> C. The reference atom from which this numbering proceeds
        is given in cases I, II, and III.

        2.2.2 case I: for identical branches (E, F) where the priorities of
        D > (E, F), D has the highest priority and is given the smallest number.

    (see <https://iupac.qmul.ac.uk/misc/ppep1.html>)

    In methylene, B is the previous heavy atom in the chain, D is the next heavy
    atom in the chain, and E, F are hydrogens. Therefore D is numbered 1, and E
    and F are numbered 2 and 3 moving clockwise from D.
    """

    center: str  # C
    parent: str  # B
    heavy: str  # D
    h2: str  # E or F, whichever is first moving clockwise from D
    h3: str  # E or F, whichever is second moving clockwise from D

    @property
    def reference_names(self) -> tuple[str, ...]:
        return (self.heavy, self.parent, self.center)

    @property
    def managed_names(self) -> tuple[str, ...]:
        return (self.h2, self.h3)

    def reindex(
        self,
        reference_coords: numpy.ndarray,
        managed_coords: numpy.ndarray,
    ) -> numpy.ndarray:
        # Unpack coordinates of each atom
        heavy, parent, center = reference_coords.transpose(1, 0, 2)
        h2c, h3c = managed_coords.transpose(1, 0, 2)
        # Compute angles from the heavy branch to each hydrogen in range (0, 360)
        clockwise_2 = numpy.mod(_dihedral(heavy, parent, center, h2c), 360.0)
        clockwise_3 = numpy.mod(_dihedral(heavy, parent, center, h3c), 360.0)
        # Reindex based on where each angle is greater
        return numpy.where((clockwise_3 < clockwise_2)[:, None], (1, 0), (0, 1))


@dataclass(frozen=True)
class _TetraPair:
    """
    Implements IUPAC rule 2.2.2 case II for two identical branches plus a lower priority branch.

    I paraphrase the relevant rules below:

        2.2.1: If, in a compound AB(P)(Q)C(E)(F)D, the sequence rule gives
        priorities A > (P, Q) and D > E > F, the "principal tortion angle" θ is
        ABCD and the branches from C are numbered CD: 1, CE: 2, CF: 3.

        2.2.2: D, E, and F are numbered in a clockwise sense when viewed in the
        direction B -> C. The reference atom from which this numbering proceeds
        is given in cases I, II, and III.

        2.2.2 case II: for identical branches (D, E) where the priorities of
        (D, E) > F, F has the lowest priority and is given the largest number.

    (see <https://iupac.qmul.ac.uk/misc/ppep1.html>)

    In a tetrahedral pair, B is the previous heavy atom in the chain, D and E
    are the two identical branches, and F is the light branch (or the first atom
    each thereof). Therefore F is numbered 3, and D and E are numbered 2 and 1
    moving counterclockwise from F (or equivalently, 1 and 2 moving clockwise
    from F, as they form a cycle).
    """

    center: str  # C
    parent: str  # B
    low: str  # F
    heavy_1: tuple[str, ...]  # D or E, starting with the atom bonded to center
    heavy_2: tuple[str, ...]  # D or E, starting with the atom bonded to center

    def __post_init__(self) -> None:
        # The branches are identical, so they must have the same number of atoms:
        if len(self.heavy_1) != len(self.heavy_2) or not self.heavy_1:
            raise ValueError(
                "identical branches must be non-empty and the same length, got "
                + f"{self.heavy_1} and {self.heavy_2}",
            )

    @property
    def reference_names(self) -> tuple[str, ...]:
        return (self.low, self.parent, self.center)

    @property
    def managed_names(self) -> tuple[str, ...]:
        return (*self.heavy_1, *self.heavy_2)

    def reindex(
        self,
        reference_coords: numpy.ndarray,
        managed_coords: numpy.ndarray,
    ) -> numpy.ndarray:
        n_branch = len(self.heavy_1)
        # Extract the coordinates for the atoms we'll compute on
        low, parent, center = reference_coords.transpose(1, 0, 2)
        branch_1, branch_2 = managed_coords[:, 0], managed_coords[:, n_branch]
        # Compute the angle between F and each branch seen from B->C
        clockwise_1 = numpy.mod(_dihedral(low, parent, center, branch_1), 360.0)
        clockwise_2 = numpy.mod(_dihedral(low, parent, center, branch_2), 360.0)
        # Reorder indices into managed_coords
        indices = numpy.arange(managed_coords.shape[1])
        return numpy.where(
            (clockwise_1 <= clockwise_2)[:, None],
            indices,
            numpy.roll(indices, n_branch),
        )


@dataclass(frozen=True)
class _Methyl:
    """
    Implements IUPAC rule 2.2.3 for a methyl (-CH3) group.

    I paraphrase the relevant rules below:

        1.6: The torsion θ(W, X, Y, Z) is the angle between the WXY and XYZ
        planes such that the eclipsed angle is 0 degrees and angles are measured
        in the range (-180, +180], with angles considered positive or negative
        when they are clockwise or counterclockwise to the eclipsed angle.

        2.2.1: If, in a compound AB(P)(Q)C(E)(F)D, the sequence rule gives
        priorities A > (P, Q) and D > E > F, the "principal tortion angle" [sic]
        θ is θ(A, B, C, D) and the branches from C are numbered CD: 1, CE: 2,
        CF: 3.

        2.2.3: If all three branches are identical, that giving the smallest
        absolute value of the principal torsion angle is normally
        assigned the highest priority and the lowest number (1); if two branches
        have torsion angles respectively +60 and -60°, the former is chosen. The
        others are numbered in a clockwise sense when viewed in the direction
        B->C.

    (see <https://iupac.qmul.ac.uk/misc/ppep1.html>)

    In a tetrahedral pair, B is the previous heavy atom in the chain, A is the
    highest priority neighbour of B other than C, C is the center, and D, E, and
    F are all hydrogen atoms. The smallest value of torsion θ(A, B, C, [DEF]) is
    numbered 1, with a +-60 degree tie favouring the positive, and the other
    hydrogens are numbered clockwise from that.
    """

    center: str  # C
    parent: str  # B
    ref: str  # A: highest-priority neighbour of B other than center
    hs: tuple[str, str, str]  # D, E, and F

    @property
    def reference_names(self) -> tuple[str, ...]:
        return (self.ref, self.parent, self.center)

    @property
    def managed_names(self) -> tuple[str, ...]:
        return self.hs

    def reindex(
        self,
        reference_coords: numpy.ndarray,
        managed_coords: numpy.ndarray,
    ) -> numpy.ndarray:
        reference, parent, center = reference_coords.transpose(1, 0, 2)
        # Get principal torsions of each hydrogen with shape (n_frames, 3)
        theta = numpy.stack(
            [
                _dihedral(reference, parent, center, h)
                for h in managed_coords.transpose(1, 0, 2)
            ],
            axis=1,
        )
        # Index all frames so we can broadcast our selection of H1 across them in indexing
        rows = numpy.arange(theta.shape[0])
        # Choose H1 by sorting first the absolute values of the angles `numpy.abs(theta)`
        # and breaking ties in favour of the positive value by secondly sorting by `-theta`.
        # This form lets us break ties that are not exactly +-60 degrees from FP
        # shenanigans
        first = numpy.lexsort((-theta, numpy.abs(theta)), axis=-1)[:, 0]
        # Recompute dihedrals to be in [0, 360) clockwise from H1
        clockwise = numpy.mod(theta - theta[rows, first][:, None], 360.0)
        # Drop H1 from clockwise without reindexing
        clockwise[rows, first] = numpy.inf
        # H2 is the first branch clockwise from H1
        second = numpy.argmin(clockwise, axis=-1)
        # H3 is the other one
        return numpy.stack([first, second, 3 - first - second], axis=1)


@dataclass(frozen=True)
class _PlanarPair:
    """
    Implements IUPAC rule 2.3.2 for a trigonal planar group with identical branches.

    I paraphrase the relevant rules below:

        1.6: The torsion θ(W, X, Y, Z) is the angle between the WXY and XYZ
        planes such that the eclipsed angle is 0 degrees and angles are measured
        in the range (-180, +180], with angles considered positive or negative
        when they are clockwise or counterclockwise to the eclipsed angle.

        2.3.1: If, in a compound AB(P)(Q)C(E)D such that B, C, D, and E are
        coplanar or nearly so, the sequence rule gives priorities A > (P, Q) and
        D > E, the principal torsion angle θ is θ(A, B, C, D) and the branches
        from C are numbered CD: 1, CE: 2.

        2.3.2: If the two branches are identical, that giving the smallest
        absolute value of the principal torsion angle is normally
        assigned the highest priority and the lowest number (1); if the two
        branches have equal absolute torsion angles (eg., +-90°), the positively
        angled branch is chosen.

    (see <https://iupac.qmul.ac.uk/misc/ppep1.html>)

    In an S1 trigonal planar centre, B is the previous heavy atom in the chain,
    A is the highest priority neighbour of B other than C, C is the center, and
    D and E are the two identical branches. The smallest value of torsion
    θ(A, B, C, [DE]) is numbered 1, with a +-90 degree tie favouring the
    positive.
    """

    a: str  # Highest priority neighbour of B (excepting C)
    b: str  # Previous heavy atom in chain B
    c: str  # Trigonal planar center C
    branch_1: tuple[str, ...]  # D or E, starting with the atom bonded to C
    branch_2: tuple[str, ...]  # D or E, starting with the atom bonded to C

    def __post_init__(self) -> None:
        # The branches are identical, so they must correspond atom for atom:
        # ``reindex`` exchanges them by rotating the managed layout by one branch
        # length, which only lands each atom on its counterpart if they match.
        if len(self.branch_1) != len(self.branch_2) or not self.branch_1:
            raise ValueError(
                f"identical branches must be non-empty and the same length, got "
                f"{self.branch_1} and {self.branch_2}",
            )

    @property
    def reference_names(self) -> tuple[str, ...]:
        return (self.a, self.b, self.c)

    @property
    def managed_names(self) -> tuple[str, ...]:
        return (*self.branch_1, *self.branch_2)

    def reindex(
        self,
        reference_coords: numpy.ndarray,
        managed_coords: numpy.ndarray,
    ) -> numpy.ndarray:
        # Extract coordinates of the atoms that go into the dihedrals
        atom_a, atom_b, atom_c = reference_coords.transpose(1, 0, 2)
        n_branch = len(self.branch_1)
        root_1, root_2 = managed_coords[:, 0], managed_coords[:, n_branch]
        # Compute the two dihedrals and their absolute values
        theta_1 = _dihedral(atom_a, atom_b, atom_c, root_1)
        theta_2 = _dihedral(atom_a, atom_b, atom_c, root_2)
        abs_1, abs_2 = numpy.abs(theta_1), numpy.abs(theta_2)
        # Order the indices for each frame according to the rule
        # Exchanging the two branches rotates the layout by one branch length
        indices = numpy.arange(managed_coords.shape[1])
        return numpy.where(
            (
                (abs_1 < abs_2) | ((abs_1 == abs_2) & (theta_1 > 0))
            )[:, None],
            indices,
            numpy.roll(indices, n_branch),
        )


def _rules_leaves_first(rules: tuple[_Rule, ...]) -> list[_Rule]:
    """Order rules so nested orbits resolve before the parent that carries them.

    Managed atom sets are laminar (any two are disjoint or nested -- a parent
    subtree contains its children), so sorting by managed-set size places each
    child before its parent. Raises if the sets overlap without nesting, which
    would make the size ordering unsound.
    """
    managed = [frozenset(rule.managed_names) for rule in rules]
    for i, child in enumerate(managed):
        for parent in managed[i + 1 :]:
            if child & parent and not (child <= parent or parent <= child):
                raise ValueError(
                    f"rule managed sets overlap without nesting: {child} & {parent}",
                )
    return sorted(rules, key=lambda rule: len(rule.managed_names))


@dataclass(frozen=True)
class _ResidueTemplate:
    """Constitutional description of one residue: element per atom name in canonical
    output order, the intra-residue bonds, and the ordered geometric rules that
    disambiguate its symmetric/prochiral atoms. `elements` is ordered (3.11+ dicts
    preserve insertion order), so its keys are the output atom order directly."""

    elements: dict[str, str]
    bonds: tuple[tuple[str, str], ...]
    rules: tuple[_Rule, ...]

    @classmethod
    def from_sidechain(
        cls,
        sidechain_bonds: tuple[tuple[str, str], ...],
        sidechain_elements: dict[str, str],
        rules: tuple[_Rule, ...],
    ) -> Self:
        """Assemble a template from the shared N-CA-C=O backbone plus a side chain.

        ``sidechain_elements`` lists the side-chain atoms beyond CA in output order;
        they are spliced between HA and C so that ``elements`` reads in canonical
        output order. ``sidechain_bonds`` are the side-chain bonds and ``rules`` the
        residue's geometric rules.
        """
        elements = {
            "N": "N",
            "H": "H",
            "CA": "C",
            "HA": "H",
            **sidechain_elements,
            "C": "C",
            "O": "O",
        }
        bonds = (
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
        ) + tuple(sidechain_bonds)
        return cls(elements, bonds, tuple(rules))


# These are *internal* (in-backbone) residue templates: each has a backbone N
# with exactly one amide H and a backbone C with one carbonyl O, i.e. it expects a
# peptide bond on both sides. There are deliberately no N-/C-terminal caps, so only
# fully bonded backbones are supported -- head-to-tail cyclic peptides today, and
# linear peptides ONLY once terminal-residue templates are added. Note well:
# adding a template is necessary but NOT sufficient to support new chemistry. Every
# residue is placed in one chain, inter-residue bonds are assumed to be backbone
# C-N (canonicalize_iupac_names raises on anything else, so cross-links such as
# disulfides are not tiled), and the geometric rules reference only same-residue
# atoms. Re-examine those assumptions before trusting output for a new backbone.
_TEMPLATES: dict[str, _ResidueTemplate] = {
    # Glycine: CA is a methylene (rule 2.2.2 case I with heavy branch = C').
    "GLY": _ResidueTemplate(
        elements={
            "N": "N",
            "H": "H",
            "CA": "C",
            "HA2": "H",
            "HA3": "H",
            "C": "C",
            "O": "O",
        },
        bonds=(
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA2"),
            ("CA", "HA3"),
            ("CA", "C"),
            ("C", "O"),
        ),
        rules=(_Methylene(center="CA", parent="N", heavy="C", h2="HA2", h3="HA3"),),
    ),
    "ALA": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "HB3")),
        sidechain_elements={"CB": "C", "HB1": "H", "HB2": "H", "HB3": "H"},
        rules=(_Methyl(center="CB", parent="CA", ref="N", hs=("HB1", "HB2", "HB3")),),
    ),
    "VAL": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB"),
            ("CB", "CG1"),
            ("CB", "CG2"),
            ("CG1", "HG11"),
            ("CG1", "HG12"),
            ("CG1", "HG13"),
            ("CG2", "HG21"),
            ("CG2", "HG22"),
            ("CG2", "HG23"),
        ),
        sidechain_elements={
            "CB": "C",
            "HB": "H",
            "CG1": "C",
            "HG11": "H",
            "HG12": "H",
            "HG13": "H",
            "CG2": "C",
            "HG21": "H",
            "HG22": "H",
            "HG23": "H",
        },
        rules=(
            _TetraPair(
                center="CB",
                parent="CA",
                low="HB",
                heavy_1=(
                    "CG1",
                    "HG11",
                    "HG12",
                    "HG13",
                ),
                heavy_2=(
                    "CG2",
                    "HG21",
                    "HG22",
                    "HG23",
                ),
            ),
            _Methyl(center="CG1", parent="CB", ref="CA", hs=("HG11", "HG12", "HG13")),
            _Methyl(center="CG2", parent="CB", ref="CA", hs=("HG21", "HG22", "HG23")),
        ),
    ),
    "THR": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB"),
            ("CB", "OG1"),
            ("CB", "CG2"),
            ("OG1", "HG1"),
            ("CG2", "HG21"),
            ("CG2", "HG22"),
            ("CG2", "HG23"),
        ),
        sidechain_elements={
            "CB": "C",
            "HB": "H",
            "OG1": "O",
            "HG1": "H",
            "CG2": "C",
            "HG21": "H",
            "HG22": "H",
            "HG23": "H",
        },
        # OG1 (O) vs CG2 (C) are fixed by priority; only the methyl needs geometry.
        rules=(
            _Methyl(center="CG2", parent="CB", ref="OG1", hs=("HG21", "HG22", "HG23")),
        ),
    ),
    "SER": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "OG"),
            ("OG", "HG"),
        ),
        sidechain_elements={"CB": "C", "HB2": "H", "HB3": "H", "OG": "O", "HG": "H"},
        rules=(_Methylene(center="CB", parent="CA", heavy="OG", h2="HB2", h3="HB3"),),
    ),
    "ASP": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "OD1"),
            ("CG", "OD2"),
        ),
        sidechain_elements={
            "CB": "C",
            "HB2": "H",
            "HB3": "H",
            "CG": "C",
            "OD1": "O",
            "OD2": "O",
        },
        rules=(
            _Methylene(center="CB", parent="CA", heavy="CG", h2="HB2", h3="HB3"),
            _PlanarPair(
                a="CA",
                b="CB",
                c="CG",
                branch_1=("OD1",),
                branch_2=("OD2",),
            ),
        ),
    ),
    "ASN": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "OD1"),
            ("CG", "ND2"),
            ("ND2", "HD21"),
            ("ND2", "HD22"),
        ),
        sidechain_elements={
            "CB": "C",
            "HB2": "H",
            "HB3": "H",
            "CG": "C",
            "OD1": "O",
            "ND2": "N",
            "HD21": "H",
            "HD22": "H",
        },
        rules=(
            _Methylene(center="CB", parent="CA", heavy="CG", h2="HB2", h3="HB3"),
            # OD1 (O) vs ND2 (N) fixed by priority; the amide hydrogens are planar.
            _PlanarPair(
                a="OD1",
                b="CG",
                c="ND2",
                branch_1=("HD21",),
                branch_2=("HD22",),
            ),
        ),
    ),
    "PHE": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CG", "CD2"),
            ("CD1", "HD1"),
            ("CD2", "HD2"),
            ("CD1", "CE1"),
            ("CD2", "CE2"),
            ("CE1", "HE1"),
            ("CE2", "HE2"),
            ("CE1", "CZ"),
            ("CE2", "CZ"),
            ("CZ", "HZ"),
        ),
        sidechain_elements={
            "CB": "C",
            "HB2": "H",
            "HB3": "H",
            "CG": "C",
            "CD1": "C",
            "HD1": "H",
            "CE1": "C",
            "HE1": "H",
            "CZ": "C",
            "HZ": "H",
            "CE2": "C",
            "HE2": "H",
            "CD2": "C",
            "HD2": "H",
        },
        rules=(
            _Methylene(center="CB", parent="CA", heavy="CG", h2="HB2", h3="HB3"),
            _PlanarPair(
                a="CA",
                b="CB",
                c="CG",
                branch_1=("CD1", "HD1", "CE1", "HE1"),
                branch_2=("CD2", "HD2", "CE2", "HE2"),
            ),
        ),
    ),
    "ARG": _ResidueTemplate.from_sidechain(
        sidechain_bonds=(
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "CD"),
            ("CD", "HD2"),
            ("CD", "HD3"),
            ("CD", "NE"),
            ("NE", "HE"),
            ("NE", "CZ"),
            ("CZ", "NH1"),
            ("CZ", "NH2"),
            ("NH1", "HH11"),
            ("NH1", "HH12"),
            ("NH2", "HH21"),
            ("NH2", "HH22"),
        ),
        sidechain_elements={
            "CB": "C",
            "HB2": "H",
            "HB3": "H",
            "CG": "C",
            "HG2": "H",
            "HG3": "H",
            "CD": "C",
            "HD2": "H",
            "HD3": "H",
            "NE": "N",
            "HE": "H",
            "CZ": "C",
            "NH1": "N",
            "HH11": "H",
            "HH12": "H",
            "NH2": "N",
            "HH21": "H",
            "HH22": "H",
        },
        rules=(
            _Methylene(center="CB", parent="CA", heavy="CG", h2="HB2", h3="HB3"),
            _Methylene(center="CG", parent="CB", heavy="CD", h2="HG2", h3="HG3"),
            _Methylene(center="CD", parent="CG", heavy="NE", h2="HD2", h3="HD3"),
            _PlanarPair(
                a="CD",
                b="NE",
                c="CZ",
                branch_1=("NH1", "HH11", "HH12"),
                branch_2=("NH2", "HH21", "HH22"),
            ),
            _PlanarPair(
                a="NE",
                b="CZ",
                c="NH1",
                branch_1=("HH11",),
                branch_2=("HH12",),
            ),
            _PlanarPair(
                a="NE",
                b="CZ",
                c="NH2",
                branch_1=("HH21",),
                branch_2=("HH22",),
            ),
        ),
    ),
}


def _bond_graph(topology: "mdtraj.Topology") -> "_LabelledGraph[int]":
    """Build an element-labelled networkx graph of a topology, keyed by atom index."""
    graph: "_LabelledGraph[int]" = networkx.Graph()
    for atom in topology.atoms:
        graph.add_node(atom.index, element=atom.element.symbol)
    for atom_i, atom_j in topology.bonds:
        graph.add_edge(atom_i.index, atom_j.index)
    return graph


def _element_match(node_a: dict[str, str], node_b: dict[str, str]) -> bool:
    """VF2 node comparator: two atoms match only when their element symbols agree."""
    return node_a["element"] == node_b["element"]


def _template_graph(template: _ResidueTemplate) -> "_LabelledGraph[str]":
    """Build the element-labelled networkx graph described by a residue template."""
    graph: "_LabelledGraph[str]" = networkx.Graph()
    for name, symbol in template.elements.items():
        graph.add_node(name, element=symbol)
    graph.add_edges_from(template.bonds)
    return graph


def _exact_cover(
    universe: frozenset[int],
    candidates: list[tuple[str, frozenset[int]]],
) -> list[tuple[str, frozenset[int]]]:
    """Return the unique sub-collection of ``candidates`` that partitions ``universe``.

    ``candidates`` are ``(residue_name, atom_set)`` template matches, which may
    overlap. A depth-first search finds a disjoint sub-collection whose union is
    exactly ``universe``. A valid peptide tiles its atoms uniquely, so exactly one
    cover is expected; ValueError is raised if there is none (an atom belongs to
    no known residue) or more than one (an ambiguous topology).

    This is Knuth's Algorithm X: at each step an uncovered atom is chosen and the
    search branches over every candidate covering it. The atom is picked by a
    most-constrained-atom (minimum-remaining-values) branching rule; as in
    Algorithm X this rule only reorders the search for speed and preserves
    completeness, so every exact cover is still enumerated and no ambiguity is
    missed.
    """
    # Map from atom indices to the candidate indices containing that atom
    containing: dict[int, list[int]] = {atom: [] for atom in universe}
    for index, (_name, atoms) in enumerate(candidates):
        for atom in atoms:
            containing[atom].append(index)

    solutions: list[list[int]] = []  # Lists of candidate indices that cover all atoms
    chosen: list[int] = []  # The list of candidate indices we're currently considering
    covered: set[int] = set()  # Set of atom indices covered by `chosen`

    def search() -> None:
        """Recurse, choosing a candidate for the most-constrained uncovered atom."""
        if len(solutions) > 1:
            # If there are more than one solutions, the template match is ambiguous
            # and we don't need to spend more time on it
            return
        if len(covered) == len(universe):
            # If the solution we're operating on covers all atoms, save it and return
            # as there's nothing to extend it with
            solutions.append(list(chosen))
            return
        # Choose the pivot to be the uncovered atom that has the fewest ways to
        # be covered by a new candidate.
        pivot = min(
            (atom for atom in universe if atom not in covered),
            key=lambda atom: sum(
                1
                for candidate_idx in containing[atom]
                if candidates[candidate_idx][1].isdisjoint(covered)
            ),
        )
        # Loop over the candidates that contain the pivot atom
        for candidate_idx in containing[pivot]:
            # Skip the candidate if any of its atoms are already covered
            atoms = candidates[candidate_idx][1]
            if not atoms.isdisjoint(covered):
                continue
            # We've found a candidate template that we can add to our chosen solution!
            chosen.append(candidate_idx)
            covered.update(atoms)
            # Check if the chosen solution covers all atoms, and if it doesn't
            # search for another candidate to add to the chosen solution.
            # The recursion depth is bounded by the number of residues because
            # each recursion adds another nonoverlapping template to the solution
            search()
            # We've exhaustively considered the solutions that involve `candidate_idx`,
            # so reset the solution we're currently working on
            covered.difference_update(atoms)
            chosen.pop()
            # and proceed with the next candidate in the next loop iteration

    search()
    if not solutions:
        raise ValueError("could not partition the peptide into known residue templates")
    if len(solutions) > 1:
        raise ValueError(
            "residue partition is ambiguous; templates tile the graph in multiple ways",
        )
    return [candidates[index] for index in solutions[0]]


def _partition_residues(
    graph: "_LabelledGraph[int]",
) -> list[tuple[str, dict[str, int]]]:
    """Tile every atom into residues by exact-cover matching against the templates.

    A residue's atoms induce a subgraph isomorphic to exactly one template (the
    inter-residue peptide bonds are the only edges leaving the set), so every
    node-induced template match is a candidate residue and the unique disjoint
    set of matches covering all atoms is the partition. This is robust to
    residues whose backbone nitrogen carries no hydrogen (e.g. proline), unlike
    peptide-bond pattern matching.

    Returns, in arbitrary order, one ``(residue_name, template-name ->
    physical-atom-index)`` tuple per residue. The map is the isomorphism found
    while matching, so the choice among symmetric atoms (e.g. HB2 vs HB3) is
    arbitrary and fixed later by the per-frame geometric rules.
    """
    # For each template collect every distinct atom set it matches, keeping one
    # representative name -> index isomorphism per set. subgraph_isomorphisms_iter
    # yields one mapping per graph automorphism, so a symmetric residue matches the
    # same atom set many times; keying on (resname, atoms) keeps only the first.
    name_map_of: dict[tuple[str, frozenset[int]], dict[str, int]] = {}
    candidates: list[tuple[str, frozenset[int]]] = []
    for resname, template in _TEMPLATES.items():
        matcher = _isomorphism.GraphMatcher(
            graph,
            _template_graph(template),
            node_match=_element_match,
        )
        for mapping in matcher.subgraph_isomorphisms_iter():
            # mapping: physical atom index -> template name
            atoms = frozenset(mapping)
            key = (resname, atoms)
            if key not in name_map_of:
                name_map_of[key] = {name: phys for phys, name in mapping.items()}
                candidates.append((resname, atoms))
    # Choose the unique exact cover of atoms by candidates, or raise.
    cover = _exact_cover(frozenset(graph.nodes), candidates)
    return [(resname, name_map_of[(resname, atoms)]) for resname, atoms in cover]


def canonicalize_iupac_names(
    traj: "mdtraj.Trajectory",
) -> "mdtraj.Trajectory":
    """Return a copy of ``traj`` with IUPAC-IUB atom names and a per-frame
    geometrically canonical atom order.

    The peptide is re-derived from the bond graph (no assumption about input
    atom order): atoms are tiled into residues by exact-cover template matching,
    each residue is matched to its IUPAC template by graph isomorphism, and every
    prochiral / symmetric label is resolved *for each frame* using the geometric
    rules of Edsall et al. (1966). Because a physical atom's prochirality can
    invert across a REST2 trajectory, the returned xyz array has its columns
    re-gathered frame by frame so that each fixed atom name always holds the
    geometrically correct atom.

    Parameters
    ----------
    traj:
        Solute-only trajectory from ``process_storage`` (solvent removed), whose
        topology carries the bonds inferred from the OpenMM ``System``.

    Returns
    -------
    mdtraj.Trajectory
        A new trajectory; ``traj`` is not modified.
    """
    graph = _bond_graph(traj.topology)

    # --- Partition atoms into residues by exact-cover template tiling ---------
    # Each residue arrives as (residue_name, template-name -> physical-atom-index);
    # the name map is the matching isomorphism, so no second graph match is needed.
    partition = _partition_residues(graph)
    name_maps = [name_map for _resname, name_map in partition]

    # Map every physical atom to its (residue index, atom name).
    residue_of_atom: dict[int, tuple[int, str]] = {}
    for res_idx, name_map in enumerate(name_maps):
        for atom_name, atom_idx in name_map.items():
            residue_of_atom[atom_idx] = (res_idx, atom_name)

    # Backbone connectivity: each inter-residue bond joins one residue's C to the
    # next residue's N. Collect these both to order the residues (by walking the
    # C->N successor chain) and to re-create the inter-residue bonds in the freshly
    # built output topology, whose template bonds are only intra-residue.
    successor: dict[int, int] = {}
    backbone_bonds: list[
        tuple[int, int]
    ] = []  # (carbon-side residue, nitrogen-side residue)
    for atom_idx_u, atom_idx_v in graph.edges:
        (res_idx_u, atom_name_u) = residue_of_atom[atom_idx_u]
        (res_idx_v, atom_name_v) = residue_of_atom[atom_idx_v]
        if res_idx_u == res_idx_v:  # intra-residue bond
            continue
        if (atom_name_u, atom_name_v) == ("C", "N"):
            carbon_side, nitrogen_side = res_idx_u, res_idx_v
        elif (atom_name_u, atom_name_v) == ("N", "C"):
            carbon_side, nitrogen_side = res_idx_v, res_idx_u
        else:
            raise ValueError(
                f"unexpected inter-residue bond between slots {atom_name_u!r} and {atom_name_v!r}",
            )
        successor[carbon_side] = nitrogen_side
        backbone_bonds.append((carbon_side, nitrogen_side))

    # Order residues by walking the C->N successor chain. A residue whose N has no
    # incoming peptide bond starts a chain: a cyclic peptide has none (so the start
    # is arbitrary; take the lowest atom index), a linear peptide has exactly one
    # (its N-terminus). More than one means the backbone is not a single chain.
    incoming = set(successor.values())
    chain_starts = [r for r in range(len(partition)) if r not in incoming]
    if not chain_starts:  # cyclic: no free terminus, start anywhere
        start = min(range(len(partition)), key=lambda p: min(name_maps[p].values()))
    elif len(chain_starts) == 1:  # linear: unique N-terminus
        start = chain_starts[0]
    else:
        raise ValueError(
            f"backbone is not a single chain or cycle: found {len(chain_starts)} "
            "residues with a free N-terminus",
        )

    order: list[int] = []
    seen: set[int] = set()
    res_idx: int | None = start
    while res_idx is not None and res_idx not in seen:
        seen.add(res_idx)
        order.append(res_idx)
        res_idx = successor.get(res_idx)  # None at the C-terminus of a linear peptide

    if len(order) != len(partition):
        raise ValueError(
            f"backbone walk covered {len(order)} of {len(partition)} residues; "
            "the peptide bonds do not form a single chain or cycle",
        )

    # --- Build the output topology (independent of coordinates) ---------------
    # Residues and atoms are added in canonical (residue, template) order, so the
    # i-th residue of `new_topology` is `order[i]` and each new atom's index is its
    # output column. `atom_map_of_res_idx` keeps the new Atom objects by name.
    new_topology = mdtraj.Topology()
    chain = new_topology.add_chain()
    atom_map_of_res_idx: dict[int, dict[str, _Atom]] = {}
    for residue_sequence, res_idx in enumerate(order, start=1):
        resname, _name_map = partition[res_idx]
        template = _TEMPLATES[resname]
        residue = new_topology.add_residue(resname, chain, resSeq=residue_sequence)
        atom_map = {
            name: new_topology.add_atom(
                name,
                _mdtraj_element.get_by_symbol(symbol),
                residue,
            )
            for name, symbol in template.elements.items()
        }
        atom_map_of_res_idx[res_idx] = atom_map

        # Intra-residue bonds from the template
        for name_i, name_j in template.bonds:
            new_topology.add_bond(atom_map[name_i], atom_map[name_j])

    # Inter-residue backbone bonds (C of one residue to N of the next)
    for carbon_side, nitrogen_side in backbone_bonds:
        new_topology.add_bond(
            atom_map_of_res_idx[carbon_side]["C"],
            atom_map_of_res_idx[nitrogen_side]["N"],
        )

    # --- Reorder the coordinates to match the new topology --------------------
    xyz = numpy.asarray(traj.xyz)
    n_frames = xyz.shape[0]

    # Map from new indices before geometric reordering to original indices
    # Before geometric reordering, so same for all frames
    # Only used to construct `perm`
    base_atom_idcs: list[int] = [
        name_maps[order[atom.residue.index]][atom.name] for atom in new_topology.atoms
    ]
    # Map from new indices to original indices for each frame
    # Will be modified over the course of the geometric reordering
    perm = numpy.tile(numpy.asarray(base_atom_idcs), (n_frames, 1))

    # Reorder some atoms in some frames so that the IUPAC naming rules are upheld
    # Note that this abandons any notion of realistic dynamics as atoms appear
    # to be frozen in their relative stereochemistry
    for res_idx in order:
        resname, name_map = partition[res_idx]
        atom_map = atom_map_of_res_idx[res_idx]
        for rule in _rules_leaves_first(_TEMPLATES[resname].rules):
            # Select the reference and managed coords from the original xyz
            reference_coords = numpy.stack(
                [xyz[:, name_map[name]] for name in rule.reference_names],
                axis=1,
            )
            managed_coords = numpy.stack(
                [xyz[:, name_map[name]] for name in rule.managed_names],
                axis=1,
            )
            # Allow the rule to choose new indices for the managed coords
            new_managed_coords_order = rule.reindex(reference_coords, managed_coords)
            # Modify `perm` given the reindexing
            managed_atom_idcs: list[int] = [
                atom_map[name].index for name in rule.managed_names
            ]
            perm[:, managed_atom_idcs] = numpy.take_along_axis(
                perm[:, managed_atom_idcs],
                new_managed_coords_order,
                axis=1,
            )
    # Use `perm` to reorder `xyz` in one shot
    new_xyz = numpy.take_along_axis(xyz, perm[:, :, None], axis=1)

    # --- Combine reordered coordinates, topology, and box into a trajectory ---
    return mdtraj.Trajectory(
        new_xyz,
        topology=new_topology,
        time=traj.time,
        unitcell_lengths=traj.unitcell_lengths,
        unitcell_angles=traj.unitcell_angles,
    )

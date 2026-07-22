import logging
import sys
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Self, TypeVar

import cyclopts
import mdtraj
import netCDF4
import numpy
import openmm
import yaml
from cyclic_peptides_pydantic import BenchmarkConfig
from openmmtools.utils import quantity_from_string
from peptide_iupac_namer import canonicalize_iupac_names

_log = logging.getLogger(__name__)


class ThermodynamicStateYamlLoader(yaml.Loader):
    """PyYAML loader for the `!Quantity`/`!ndarray` tags openmmtools uses to serialize
    `ThermodynamicState`s into multistate `.nc` storage files.

    Vendored from `openmmtools.multistate.multistatereporter._DictYamlLoader`, since
    that class is private and `openmmtools.storage.iodrivers._DictYamlLoader` is a
    same-named but incompatible sibling that expects different mapping keys.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_constructor("!Quantity", self.quantity_constructor)
        self.add_constructor("!ndarray", self.ndarray_constructor)

    @staticmethod
    def quantity_constructor(
        loader: yaml.Loader,
        node: yaml.Node,
    ) -> openmm.unit.Quantity:
        loaded_mapping = loader.construct_mapping(node)
        data_unit = quantity_from_string(loaded_mapping["unit"])
        data_value = loaded_mapping["value"]
        return data_value * data_unit

    @staticmethod
    def ndarray_constructor(loader: yaml.Loader, node: yaml.Node) -> numpy.ndarray:
        loaded_mapping = loader.construct_mapping(node, deep=True)
        data_type = numpy.dtype(loaded_mapping["type"])
        data_shape = loaded_mapping["shape"]
        data_values = loaded_mapping["values"]
        data = numpy.ndarray(shape=data_shape, dtype=data_type)
        if 0 not in data_shape:
            data[:] = data_values
        return data


def main(
    config_json: Path,
    results: Path,
    debug: bool,
    overwrite_files: bool,
    log_stdout: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s via %(name)s)",
        stream=sys.stdout if log_stdout else sys.stderr,
    )

    config = BenchmarkConfig.from_json_path(config_json)
    for target in config.targets:
        target_path = results / target.sequence
        for replica_dir in target_path.glob("replica-*"):
            storage_path = replica_dir / f"{target.sequence}-storage.nc"
            checkpoint_path = replica_dir / f"{target.sequence}-checkpoint.nc"
            topology_path = config_json.parent / f"{target.name}-rung0.pdb"
            storage_to_dcd_pdb(storage_path, checkpoint_path, topology_path, overwrite_files)


def storage_to_dcd_pdb(
    storage_path: Path,
    checkpoint_path: Path,
    topology_path: Path,
    overwrite_files: bool,
):
    assert storage_path.exists()
    assert checkpoint_path.exists()

    nc = netCDF4.Dataset(str(storage_path), "r")
    options = yaml.load(nc.variables["options"][0], Loader=ThermodynamicStateYamlLoader)

    target_iters = options["number_of_iterations"]
    completed_iters = nc.variables["last_iteration"][0]

    _log.info(
        "%s: %s/%s iterations completed", storage_path, completed_iters, target_iters
    )
    if target_iters > completed_iters:
        _, _, i_repl = storage_path.parent.name.partition("replica-")
        resume_manifest = unwrap(storage_path.parent.glob(f"*-{i_repl}.yaml"))
        _log.warning(
            "%s has not completed; resume with: kubectl apply -f %s",
            storage_path,
            resume_manifest,
        )

    # Load the system and get a list of bonds
    system = system_from_openmmtools_storage(nc)
    bonds: list[tuple[int, int]] = []
    for force in system.getForces():
        if isinstance(force, (openmm.HarmonicBondForce, openmm.CustomBondForce)):
            for bond_idx in range(force.getNumBonds()):
                i, j, *_ = force.getBondParameters(bond_idx)
                bonds.append((i, j))
    for constraint_idx in range(system.getNumConstraints()):
        i, j, _distance = system.getConstraintParameters(constraint_idx)
        bonds.append((i, j))

    # Load the topology
    pdb_traj = mdtraj.load_pdb(topology_path)
    topology = pdb_traj.topology
    for i, j in bonds:
        topology.add_bond(topology.atom(i), topology.atom(j))

    # get positions and box vectors of the bottom-most rung
    positions = nc.variables["positions"]
    box_vectors = nc.variables["box_vectors"]
    positions = numpy.asarray(positions)[:, 0, :, :]
    box_vecs = numpy.asarray(box_vectors)[:, 0, :, :]

    # Compile into a DCD file
    traj = mdtraj.Trajectory(
        positions,
        topology=topology,
    )
    traj.unitcell_vectors = box_vecs
    traj.make_molecules_whole(inplace=True)
    traj.remove_solvent(inplace=True)
    traj.superpose(pdb_traj.remove_solvent())

    traj = canonicalize_iupac_names(traj)

    dcd_path = storage_path.with_suffix(".aligned.reindexed.dcd")
    pdb_path = storage_path.with_suffix(".aligned.reindexed.pdb")
    traj.save_dcd(dcd_path, force_overwrite=overwrite_files)
    _log.info("Saved %s with %s frames, %s atoms", dcd_path, traj.n_frames, traj.n_atoms)
    traj[0].save_pdb(pdb_path, force_overwrite=overwrite_files)
    _log.info("Saved %s with %s atoms", pdb_path, traj.n_atoms)


def system_from_openmmtools_storage(nc: netCDF4.Dataset) -> openmm.System:
    system_raw = b"".join(
        nc.groups["thermodynamic_states"].variables["state0"][:],
    ).decode()
    system_doc = yaml.load(system_raw, Loader=ThermodynamicStateYamlLoader)
    system_xml = zlib.decompress(system_doc["standard_system"]).decode()
    return openmm.XmlSerializer.deserialize(system_xml)


T = TypeVar("T")


def unwrap(iterable: Iterable[T]) -> T:
    Sentinel = type("Sentinel", (), {})
    sentinel = Sentinel()
    iterator = iter(iterable)
    val = next(iterator, sentinel)
    assert not isinstance(val, Sentinel)
    assert isinstance(next(iterator, sentinel), Sentinel)
    return val

if __name__ == "__main__":
    cyclopts.run(main)

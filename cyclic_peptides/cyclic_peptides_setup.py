#! /usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Any, Literal

import cyclopts
import numpy as np
import openmm
import openmm.app
import openmmtools
from openff.interchange import Interchange
from openff.interchange.components._packmol import (
    RHOMBIC_DODECAHEDRON,
    RHOMBIC_DODECAHEDRON_XYHEX,
    UNIT_CUBE,
    pack_box,
    solvate_topology,
)
from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openmm.app.simulation import Simulation
from openmmforcefields.generators.template_generators import SMIRNOFFTemplateGenerator
from proteinbenchmark import OpenMMHrexEnsemble, OpenMMSimulation
from proteinbenchmark import read_xml as read_system_xml
from proteinbenchmark import write_xml as write_system_xml

LOGGER = logging.getLogger(__name__)


WATER = Molecule.from_smiles("O")
WATER.generate_conformers(n_conformers=1)
SODIUM = Molecule.from_smiles("[Na+]")
SODIUM.generate_conformers(n_conformers=1)
CHLORIDE = Molecule.from_smiles("[Cl-]")
CHLORIDE.generate_conformers(n_conformers=1)

BOX_SHAPES = {
    "RHOMBIC_DODECAHEDRON": RHOMBIC_DODECAHEDRON,
    "CUBE": UNIT_CUBE,
    "RHOMBIC_DODECAHEDRON_XYHEX": RHOMBIC_DODECAHEDRON_XYHEX,
}


def main(config_json: Path, debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s via %(name)s)",
    )
    config = load_config(config_json)
    n_replicas = config["configuration"]["ensemble"]["n_replicas"]

    for target_idx, target in enumerate(config["targets"]):

        def ffn(s: str) -> Path:
            return format_filename(
                config["configuration"]["file_names"][s],
                config,
                target_idx,
            )

        storage = ffn("storage_file")
        checkpoint = ffn("checkpoint_file")
        if storage.exists() and checkpoint.exists():
            continue

        LOGGER.info(f"Preparing target {target_idx}: {target['name']}...")
        LOGGER.info("  Generating box...")
        interchange, positions, boxes = prep_target_modeller(
            n_replicas=n_replicas,
            **target,
            **config["configuration"]["solvation"],
            force_field=ForceField(*config["configuration"]["force_field_files"]),
            vizualization_file=ffn("visualization_file"),
        )
        target_name = interchange.topology.molecule(0).properties["name"]

        LOGGER.info("  Parametrizing and serializing System...")

        base_system = interchange.to_openmm_system(
            hydrogen_mass=config["configuration"]["integration"].get(
                "hydrogen_mass",
                Quantity(1.00784, "Da"),
            ),
        )
        base_system_xml_file = f"{target_name}-system.xml"
        write_system_xml(openmm_system=base_system, xml_file_name=base_system_xml_file)
        for i, (this_positions, box) in enumerate(zip(positions, boxes)):
            interchange.positions = this_positions
            interchange.box = box
            interchange.to_pdb(f"{target_name}-rung{i}.pdb")

        LOGGER.info("  Preparing OpenMMSimulation...")

        integration_config = config["configuration"]["integration"]
        lengths_config = config["configuration"]["lengths"]
        ensemble_config = config["configuration"]["ensemble"]
        base_simulation = OpenMMSimulation(
            openmm_system_file=base_system_xml_file,
            initial_pdb_file=f"{target_name}-rung0.pdb",
            dcd_reporter_file=str(storage),
            state_reporter_file=str(storage),
            checkpoint_file=str(checkpoint),
            save_state_prefix=str(ffn("save_state_prefix")),
            temperature=integration_config["temperature"] * openmm.unit.kelvin,
            pressure=integration_config["pressure"] * openmm.unit.atmosphere,
            langevin_friction=(
                integration_config["langevin_friction"] / openmm.unit.picosecond
            ),
            barostat_frequency=integration_config["barostat_frequency"],
            timestep=integration_config["timestep_fs"] * openmm.unit.femtosecond,
            traj_length=Quantity(
                lengths_config["traj_length_ns"],
                "nanosecond",
            ).to_openmm(),
            frame_length=Quantity(
                lengths_config["frame_length_ns"],
                "nanosecond",
            ).to_openmm(),
            checkpoint_length=Quantity(
                lengths_config["checkpoint_length_ns"],
                "nanosecond",
            ).to_openmm(),
            save_state_length=Quantity(
                (
                    ensemble_config[
                        "steps_between_exchange_attempts"
                    ]
                    * integration_config["timestep_fs"]
                ),
                "femtosecond",
            ).to_openmm(),
        )

        LOGGER.info("  Scaling...")

        ensemble = OpenMMHrexEnsemble.construct_rest2(
            n_replicas=n_replicas,
            base_simulation=base_simulation,
            tempered_atom_idcs=list(range(interchange.topology.molecule(0).n_atoms)),
            steps_between_exchange_attempts=ensemble_config[
                "steps_between_exchange_attempts"
            ],
            max_effective_temperature=Quantity(
                ensemble_config["max_effective_temperature"],
                "Kelvin",
            ),
        )

        LOGGER.info("  Checking...")
        test_ensemble(ensemble, interchange, positions)

        LOGGER.info("  Preparing sampler...")
        sampler = ensemble.setup_simulation(
            require_gpu=False,
        )

        LOGGER.info("  set states for sampler...")
        sampler.sampler_states = [
            openmmtools.states.SamplerState(
                positions=this_positions.to_openmm(),
                box_vectors=box.to_openmm(),
            )
            for this_positions, box in zip(positions, boxes)
        ]
        LOGGER.info("  Energy minimize...")
        sampler.minimize()
        LOGGER.info("  Equilibrate...")
        sampler.equilibrate(
            int(
                np.ceil(
                    lengths_config["equilibration_length_ns"]
                    * 1_000_000
                    / integration_config["timestep_fs"]
                    / ensemble_config[
                        "steps_between_exchange_attempts"
                    ],
                ),
            ),
        )
        LOGGER.info("  Done!")


def prep_target_packmol(
    *,
    n_replicas: int,
    box_shape: Literal["RHOMBIC_DODECAHEDRON", "CUBE", "RHOMBIC_DODECAHEDRON_XYHEX"],
    target_density: float,
    nacl_molarity: float,
    solvent_padding_nm: float,
    smiles: str,
    sequence: str,
    name: str,
    force_field: ForceField,
    vizualization_file: Path,
) -> tuple[Interchange, list[Quantity]]:
    peptide = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    peptide.properties["sequence"] = sequence
    peptide.properties["name"] = name
    peptide.perceive_residues()
    vizualization_file.write_text(
        peptide.visualize(backend="rdkit", show_all_hydrogens=False).data,
    )

    peptide.generate_conformers(n_conformers=n_replicas)
    pack_box_kwargs = None
    positions: list[Quantity] = []
    topology: Topology = peptide.to_topology()
    solvated_top: Topology
    for conf in peptide.conformers:
        topology.set_positions(conf)
        if pack_box_kwargs is None:
            solvated_top = solvate_topology(
                topology=topology,
                box_shape=BOX_SHAPES[box_shape],
                nacl_conc=Quantity(nacl_molarity, "molar"),
                padding=Quantity(solvent_padding_nm, "nanometer"),
                target_density=Quantity(target_density, "g/L"),
            )
            n_waters = sum(1 for mol in solvated_top.molecules if mol == WATER)
            n_sodium = sum(1 for mol in solvated_top.molecules if mol == SODIUM)
            n_chloride = sum(1 for mol in solvated_top.molecules if mol == CHLORIDE)
            pack_box_kwargs = dict(
                molecules=[WATER, SODIUM, CHLORIDE],
                number_of_copies=[n_waters, n_sodium, n_chloride],
                box_vectors=solvated_top.box_vectors,
            )
            print(pack_box_kwargs)
        else:
            solvated_top = pack_box(
                solute=topology,
                **pack_box_kwargs,
            )
        this_positions = solvated_top.get_positions()
        assert isinstance(this_positions, Quantity)
        positions.append(this_positions)

    base_interchange = Interchange.from_smirnoff(
        force_field=force_field,
        topology=solvated_top,
    )

    return (base_interchange, positions)


UNIT_BOX = {
    "cube": np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    ),
    "dodecahedron": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, np.sqrt(2.0) / 2.0],
        ],
    ),
}


# # If you specify box_vectors, none of the other box arguments may be
# # specified
# @overload
# def solvate_with_modeller(
#     topology: Topology,
#     *,
#     box_vectors: Quantity,
#     salt_conc: Quantity = Quantity(0.0, "mole/liter"),
#     positive_ion: Molecule = Molecule.from_smiles("[Na+]"),
#     negative_ion: Molecule = Molecule.from_smiles("[Cl-]"),
#     neutralize: bool = True,
# ) -> None: ...


# # If you specify box_padding, you must also specify box_shape, and none of
# # the other box arguments may be specified
# @overload
# def solvate_with_modeller(
#     topology: Topology,
#     *,
#     box_shape: Literal["cube", "dodecahedron", None],
#     box_padding: Quantity,
#     salt_conc: Quantity = Quantity(0.0, "mole/liter"),
#     positive_ion: Molecule = Molecule.from_smiles("[Na+]"),
#     negative_ion: Molecule = Molecule.from_smiles("[Cl-]"),
#     neutralize: bool = True,
# ) -> None: ...


# # If you specify box_width, you must also specify box_shape, and none of
# # the other box arguments may be specified
# @overload
# def solvate_with_modeller(
#     topology: Topology,
#     *,
#     box_shape: Literal["cube", "dodecahedron", None],
#     box_width: Quantity,
#     salt_conc: Quantity = Quantity(0.0, "mole/liter"),
#     positive_ion: Molecule = Molecule.from_smiles("[Na+]"),
#     negative_ion: Molecule = Molecule.from_smiles("[Cl-]"),
#     neutralize: bool = True,
# ) -> None: ...


# # If you specify n_waters, you must also specify box_shape, and none of
# # the other box arguments may be specified. box_n_waters is the final total
# # number of waters; we need to subtract the number of waters in the initial
# # topology and the number of ions we want before we pass it to numAdded
# @overload
# def solvate_with_modeller(
#     topology: Topology,
#     *,
#     box_shape: Literal["cube", "dodecahedron", None],
#     box_n_solvent: Quantity,
#     salt_conc: Quantity = Quantity(0.0, "mole/liter"),
#     positive_ion: Molecule = Molecule.from_smiles("[Na+]"),
#     negative_ion: Molecule = Molecule.from_smiles("[Cl-]"),
#     neutralize: bool = True,
# ) -> None: ...


# # If you specify none of the box arguments, the topology's box vectors are
# # used
# @overload
# def solvate_with_modeller(
#     topology: Topology,
#     *,
#     salt_conc: Quantity = Quantity(0.0, "mole/liter"),
#     positive_ion: Molecule = Molecule.from_smiles("[Na+]"),
#     negative_ion: Molecule = Molecule.from_smiles("[Cl-]"),
#     neutralize: bool = True,
# ) -> None: ...


def solvate_with_modeller(
    topology: Topology,
    *,
    box_vectors: Quantity | None = None,
    box_shape: Literal["RHOMBIC_DODECAHEDRON", "CUBE", None],
    box_padding: Quantity | None = None,
    box_width: Quantity | None = None,
    box_n_solvent: int | None = None,
    salt_conc: Quantity = Quantity(0.0, "mole/liter"),
    neutralize: bool = True,
) -> Topology:
    # TODO: Make this work when residues are defined
    if box_shape is not None and (
        box_padding is None and box_width is None and box_n_solvent is None
    ):
        raise ValueError(
            "Cannot specify box shape without one of box_padding, box_width, box_n_solvent",
        )
    if (
        len(
            [
                x
                for x in (box_vectors, box_padding, box_width, box_n_solvent)
                if x is not None
            ],
        )
        > 1
    ):
        raise ValueError(
            "box_vectors, box_padding, box_width, and box_n_solvent are mutually exclusive",
        )
    if box_shape is None and (
        box_padding is not None or box_width is not None or box_n_solvent is not None
    ):
        raise ValueError(
            "box_shape is required when box_padding, box_width, or box_n_solvent is specified",
        )
    if box_width is not None:
        assert box_shape is not None
        box_vectors = UNIT_BOX[box_shape] * box_width
        box_width = None
        box_shape = None

    # Remove residue info so that SMIRNOFFTemplateGenerator can operate on
    # whole molecules - we'll add it back after
    topology = Topology(topology)
    original_residue_names = [
        atom.metadata.pop("residue_name", None) for atom in topology.atoms
    ]
    original_residue_numbers = [
        atom.metadata.pop("residue_number", None) for atom in topology.atoms
    ]

    modeller = openmm.app.Modeller(
        topology.to_openmm(),
        topology.get_positions().to_openmm(),
    )
    ommff = openmm.app.ForceField("amber/tip3p_standard.xml")
    smirnoff = SMIRNOFFTemplateGenerator(
        forcefield="openff-2.3.0.offxml",
        molecules=[mol for mol in topology.unique_molecules if mol != WATER],
    )
    ommff.registerTemplateGenerator(smirnoff.generator)
    modeller.addSolvent(
        forcefield=ommff,
        model="tip3p",
        boxVectors=box_vectors.to_openmm() if box_vectors is not None else None,
        boxShape={"RHOMBIC_DODECAHEDRON": "dodecahedron", "CUBE": "cube", None: None}[
            box_shape
        ],
        padding=box_padding.to_openmm() if box_padding is not None else None,
        numAdded=box_n_solvent,
        ionicStrength=salt_conc.to_openmm(),
        positiveIon="Na+",
        negativeIon="Cl-",
        neutralize=neutralize,
    )
    topology = Topology.from_openmm(
        modeller.topology,
        unique_molecules={*topology.unique_molecules, WATER, SODIUM, CHLORIDE},
    )
    positions = Quantity(modeller.positions.value_in_unit(openmm.unit.nanometer), "nm")
    topology.set_positions(positions)

    # Restore residue info
    for atom, resname, resnum in zip(
        topology.atoms,
        original_residue_names,
        original_residue_numbers,
    ):
        if resname is not None:
            atom.metadata["residue_name"] = resname
            atom.metadata["residue_number"] = resnum

    return topology


def prep_target_modeller(
    *,
    n_replicas: int,
    box_shape: Literal["RHOMBIC_DODECAHEDRON", "CUBE"],
    nacl_molarity: float,
    solvent_padding_nm: float,
    smiles: str,
    sequence: str,
    name: str,
    force_field: ForceField,
    vizualization_file: Path,
) -> tuple[Interchange, list[Quantity], list[Quantity]]:
    peptide = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    peptide.properties["sequence"] = sequence
    peptide.properties["name"] = name
    vizualization_file.write_text(
        peptide.visualize(backend="rdkit", show_all_hydrogens=False).data,
    )

    peptide.generate_conformers(n_conformers=n_replicas)
    pack_box_kwargs = None
    positions: list[Quantity] = []
    boxes: list[Quantity] = []
    topology: Topology = peptide.to_topology()
    solvated_top: Topology
    for conf in peptide.conformers:
        topology.set_positions(conf)
        if pack_box_kwargs is None:
            solvated_top = solvate_with_modeller(
                topology=topology,
                box_shape=box_shape,
                box_padding=Quantity(solvent_padding_nm, "nanometer"),
                salt_conc=Quantity(nacl_molarity, "molar"),
            )
            n_waters = sum(1 for mol in solvated_top.molecules if mol == WATER)
            n_sodium = sum(1 for mol in solvated_top.molecules if mol == SODIUM)
            n_chloride = sum(1 for mol in solvated_top.molecules if mol == CHLORIDE)
            pack_box_kwargs = dict(
                box_n_solvent=n_waters + n_sodium + n_chloride,
                salt_conc=Quantity(nacl_molarity, "molar"),
                box_shape=box_shape,
            )
            print(pack_box_kwargs)
        else:
            solvated_top = solvate_with_modeller(
                topology,
                **pack_box_kwargs,
            )
        this_positions = solvated_top.get_positions()
        assert isinstance(this_positions, Quantity)
        positions.append(this_positions)
        boxes.append(solvated_top.box_vectors)

    solvated_top.molecule(0).perceive_residues()
    base_interchange = Interchange.from_smirnoff(
        force_field=force_field,
        topology=solvated_top,
    )

    return (base_interchange, positions, boxes)


def format_filename(fn: str, config: dict[str, Any], target: int) -> Path:
    smiles = config["targets"][target]["smiles"]
    sequence = config["targets"][target]["sequence"]
    name = config["targets"][target]["sequence"]
    n_replicas = config["configuration"]["ensemble"]["n_replicas"]
    timestep_fs = config["configuration"]["integration"]["timestep_fs"]
    return Path(
        fn.replace("{smiles}", f"{smiles}")
        .replace("{sequence}", f"{sequence}")
        .replace("{name}", f"{name}")
        .replace("{n_replicas}", f"{n_replicas}")
        .replace("{timestep_fs}", f"{timestep_fs}"),
    )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_bytes())
    config["targets"] = [
        {"name": target["sequence"], **target} for target in config["targets"]
    ]
    config["configuration"]["integration"].setdefault("hydrogen_mass", 1.00784)
    config["configuration"]["file_names"] = {
        "storage_file": "{name}-storage.nc",
        "checkpoint_file": "{name}-checkpoint.nc",
        "save_state_prefix": "{name}-state_",
        "visualization_file": "peptide_{name}.svg",
        **config["configuration"].get("file_names", {}),
    }
    return config


def test_ensemble(
    ensemble: OpenMMHrexEnsemble,
    interchange: Interchange,
    positions: list[Quantity],
):

    if ensemble.base_simulation.n_steps % ensemble.steps_between_exchange_attempts != 0:
        raise ValueError(
            "steps_between_exchange_attempts must evenly divide n_steps",
        )
    if (
        ensemble.base_simulation.checkpoint_frequency
        % ensemble.steps_between_exchange_attempts
        != 0
    ):
        raise ValueError(
            "steps_between_exchange_attempts must evenly divide checkpoint_frequency",
        )
    if (
        ensemble.base_simulation.output_frequency
        % ensemble.steps_between_exchange_attempts
        != 0
    ):
        raise ValueError(
            "steps_between_exchange_attempts must evenly divide output_frequency",
        )

    # Load OpenMM systems and initial PDB
    openmm_systems = [read_system_xml(fn) for fn in ensemble.system_fn_ladder]
    # initial_pdb = openmm.app.PDBFile(ensemble.base_simulation.initial_pdb_file)

    LOGGER.info("    Unscaled base system without barostat")
    base_system_simulation = set_up_simulation(
        interchange,
        read_system_xml(ensemble.base_simulation.openmm_system_file),
        ensemble.base_simulation,
    )
    check_forces(base_system_simulation)

    # Set up Monte Carlo barostat
    if ensemble.base_simulation.pressure.value_in_unit(openmm.unit.atmosphere) > 0:
        for system in openmm_systems:
            system.addForce(
                openmm.MonteCarloBarostat(
                    ensemble.base_simulation.pressure,
                    ensemble.base_simulation.temperature,
                    ensemble.base_simulation.barostat_frequency,
                ),
            )

    # Set up simulations
    simulations: list[Simulation] = []
    for i, (this_positions, system) in enumerate(zip(positions, openmm_systems)):
        LOGGER.info(f"    Rung {i}")
        interchange.positions = this_positions
        simulation = set_up_simulation(interchange, system, ensemble.base_simulation)
        check_forces(simulation)
        simulations.append(simulation)

    # for i, simulation in enumerate(simulations):
    #     LOGGER.info(f"    Energy minimize rung {i}")
    #     simulation.minimizeEnergy()
    #     simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())
    #     LOGGER.info(f"    Integrate rung {i} 10000 steps")
    #     simulation.step(10000)


def get_platform_property_dict(
    context: openmm.Context,
) -> dict[str, Any]:
    platform = context.getPlatform()
    return {
        name: platform.getPropertyValue(context, name)
        for name in platform.getPropertyNames()
    }


def check_forces(simulation) -> None:
    for force_idx, force in enumerate(simulation.system.getForces()):
        state = simulation.context.getState(
            getEnergy=True,
            getForces=True,
            groups={force_idx},
        )
        assert isinstance(state, openmm.State)
        potential = state.getPotentialEnergy()
        forces = state.getForces(asNumpy=True)
        rms_force = np.sqrt(np.power(forces, 2).sum(axis=-1).mean())
        LOGGER.info(f"      Force {force.getName()} has {potential=} and {rms_force=} ")


def set_up_simulation(
    interchange: Interchange,
    system: openmm.System,
    base_simulation: OpenMMSimulation,
) -> openmm.app.Simulation:
    # Set up BAOAB Langevin integrator from openmmtools with VRORV splitting
    integrator = openmm.LangevinMiddleIntegrator(
        base_simulation.temperature,
        base_simulation.langevin_friction,
        base_simulation.timestep,
    )

    for force_idx, force in enumerate(system.getForces()):
        force.setForceGroup(force_idx)

    simulation = openmm.app.Simulation(
        topology=interchange.to_openmm_topology(),
        system=system,
        integrator=integrator,
    )
    simulation.context.setPositions(interchange.positions.to_openmm())
    simulation.context.setPeriodicBoxVectors(*interchange.box.to_openmm())

    return simulation


if __name__ == "__main__":
    cyclopts.run(main)

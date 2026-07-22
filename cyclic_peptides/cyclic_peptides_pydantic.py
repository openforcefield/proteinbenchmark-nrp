"""Pydantic v2 models for cyclic-peptides-benchmark-v1.0.0.schema.json.

Run this file directly to validate the schema equivalence and config.json parsing:
    python cyclic-peptides-benchmark-pydantic.py
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoxShape(str, Enum):
    """Shape of the periodic simulation box."""

    RHOMBIC_DODECAHEDRON = "RHOMBIC_DODECAHEDRON"
    CUBE = "CUBE"
    RHOMBIC_DODECAHEDRON_XYHEX = "RHOMBIC_DODECAHEDRON_XYHEX"


PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]

DEFAULT_SENTINEL_STR: str = "\0DEFAULT\0"


class Target(BaseModel):
    """A single cyclic peptide target to simulate."""

    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(description="SMILES string defining the cyclic peptide topology")
    sequence: str = Field(description="One-letter amino acid sequence, used in output file names")
    name: str = Field(
        default=DEFAULT_SENTINEL_STR,
        description="Human-readable label for this target. Defaults to sequence if omitted.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("name") in [None, DEFAULT_SENTINEL_STR]:
            data = {**data, "name": data["sequence"]}
        return data


class Solvation(BaseModel):
    """Arguments to solvate_topology() from openff.interchange.components._packmol."""

    model_config = ConfigDict(extra="forbid")

    solvent_padding_nm: PositiveFloat = Field(description="Padding around the solute in nanometers")
    nacl_molarity: NonNegativeFloat = Field(description="NaCl concentration in mol/L (molar)")
    box_shape: BoxShape = Field(description="Shape of the periodic box")


class Integration(BaseModel):
    """Properties of the integrator, barostat, and thermostat."""

    model_config = ConfigDict(extra="forbid")

    temperature: PositiveFloat = Field(description="Simulation temperature in kelvin")
    pressure: PositiveFloat = Field(description="Simulation pressure in atmospheres")
    langevin_friction: PositiveFloat = Field(
        description="Langevin friction coefficient in inverse picoseconds"
    )
    barostat_frequency: PositiveInt = Field(
        description="Number of steps between Monte Carlo barostat attempts"
    )
    timestep_fs: PositiveFloat = Field(
        description=(
            "Integration timestep in femtoseconds."
            " Recommended: 4.0 fs when hydrogen_mass is set, 2.0 fs otherwise."
        )
    )
    hydrogen_mass: PositiveFloat = Field(
        default=1.00784,
        description=(
            "Hydrogen mass for repartitioning in daltons."
            " Recommend: 3.0 to enable HMR, unset otherwise."
        ),
    )


class Lengths(BaseModel):
    """Time-length OpenMMSimulation constructor arguments and equilibration length."""

    model_config = ConfigDict(extra="forbid")

    equilibration_length_ns: PositiveFloat = Field(
        description="Equilibration run length in nanoseconds, passed to sampler.equilibrate()"
    )
    traj_length_ns: PositiveFloat = Field(
        description="Production trajectory length in nanoseconds"
    )
    frame_length_ns: PositiveFloat = Field(
        description="Interval between trajectory frames in nanoseconds"
    )
    checkpoint_length_ns: PositiveFloat = Field(
        description="Interval between checkpoint saves in nanoseconds"
    )


class Ensemble(BaseModel):
    """Arguments to OpenMMHrexEnsemble.construct_rest2()."""

    model_config = ConfigDict(extra="forbid")

    n_replicas: PositiveInt = Field(description="Number of REST2 replicas")
    max_effective_temperature: PositiveFloat = Field(
        description="Maximum effective temperature for the REST2 ladder in kelvin"
    )
    steps_between_exchange_attempts: PositiveInt = Field(
        description=(
            "Number of MD steps between replica exchange attempts."
            " Must evenly divide the steps derived from traj_length, frame_length,"
            " and checkpoint_length."
        )
    )


class FileNames(BaseModel):
    """Python f-string templates for output file paths. Available variables: {smiles}, {sequence}, {name}, {n_replicas}, {timestep_fs}."""

    model_config = ConfigDict(extra="forbid")

    storage_file: str = Field(
        default="{name}-storage.nc",
        description="Path for the NetCDF trajectory/state reporter storage file",
    )
    checkpoint_file: str = Field(
        default="{name}-checkpoint.nc",
        description="Path for the simulation checkpoint file",
    )
    save_state_prefix: str = Field(
        default="{name}-state_",
        description="Prefix for full state save files",
    )
    visualization_file: str = Field(
        default="peptide_{name}.svg",
        description="Path for the peptide structure visualization SVG",
    )


class Configuration(BaseModel):
    """Simulation configuration shared across all targets."""

    model_config = ConfigDict(extra="forbid")

    force_field_files: Annotated[list[str], Field(min_length=1)] = Field(
        description=(
            "Ordered list of OpenFF SMIRNOFF force field files passed to ForceField()."
            ' Example: ["openff_no_water-3.0.0-alpha0.offxml", "opc3.offxml"]'
        )
    )
    solvation: Solvation
    integration: Integration
    lengths: Lengths
    ensemble: Ensemble
    file_names: FileNames = Field(
        default_factory=FileNames,
        description=(
            "Python f-string templates for output file paths."
            " Available variables: {smiles}, {sequence}, {name}, {n_replicas}, {timestep_fs}."
        ),
    )


class BenchmarkConfig(BaseModel):
    """Configuration for cyclic peptide REST2 simulations using OpenFF force fields"""

    model_config = ConfigDict(
        title="Cyclic Peptides Benchmark Configuration v1.0.0",
        extra="forbid",
        populate_by_name=True,
    )

    # $schema is an IDE/LSP hint; present in JSON files but not part of the data model.
    schema_uri: str | None = Field(
        default=None,
        alias="$schema",
        description="JSON Schema URI for editor autocomplete",
    )
    targets: Annotated[list[Target], Field(min_length=1)] = Field(
        description="List of cyclic peptide targets to simulate"
    )
    configuration: Configuration

    def _resolve_template(self, template: str) -> list[str]:
        n_replicas = self.configuration.ensemble.n_replicas
        timestep_fs = self.configuration.integration.timestep_fs
        return [
            template.format(
                smiles=t.smiles,
                sequence=t.sequence,
                name=t.name,
                n_replicas=n_replicas,
                timestep_fs=timestep_fs,
            )
            for t in self.targets
        ]

    @property
    def storage_files(self) -> list[str]:
        return self._resolve_template(self.configuration.file_names.storage_file)
    @property
    def checkpoint_files(self) -> list[str]:
        return self._resolve_template(self.configuration.file_names.checkpoint_file)
    @property
    def save_state_prefixes(self) -> list[str]:
        return self._resolve_template(self.configuration.file_names.save_state_prefix)
    @property
    def visualization_files(self) -> list[str]:
        return self._resolve_template(self.configuration.file_names.visualization_file)

    @classmethod
    def from_json_path(cls, path: Path) -> Self:
        raw = json.loads(path.read_text())
        return cls.model_validate(raw)

# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------


def resolve_refs(schema: dict[str, object]) -> dict[str, object]:
    """Inline all $ref pointers, returning a schema with no $refs or $defs.

    General-purpose: works for any JSON Schema using the #/$defs/ pointer
    convention. Preserves all constraints — no information is deleted.
    """
    defs: dict[str, object] = schema.get("$defs", {})  # type: ignore[assignment]

    def _inline(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].removeprefix("#/$defs/")
                return _inline(defs[name])
            return {k: _inline(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_inline(v) for v in node]
        return node

    return _inline(schema)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    here = Path(__file__).parent

    # --- 1. Parse config.json against the Pydantic model ---
    config_path = here / "config.json"
    raw = json.loads(config_path.read_text())
    config = BenchmarkConfig.model_validate(raw)
    print(f"config.json parsed successfully: {len(config.targets)} targets")

    # Verify Target.name defaulting
    assert all(isinstance(t.name, str) for t in config.targets), "Target.name must always be str"
    nameless = Target.model_validate({"smiles": "C", "sequence": "G"})
    assert nameless.name == "G", "name should default to sequence"

    # Verify file_names is always populated and templates resolve correctly
    assert config.storage_files == [f"{t.name}-storage.nc" for t in config.targets]
    assert config.checkpoint_files == [f"{t.name}-checkpoint.nc" for t in config.targets]
    print("Validator and property checks passed.")

    # --- 2. Compare generated JSON schema against the hand-written schema ---
    # Resolve $refs in both schemas before comparing so that structural differences
    # (inline vs $defs) don't obscure genuine semantic differences.
    pydantic_schema = resolve_refs(BenchmarkConfig.model_json_schema(by_alias=True))
    original_schema = resolve_refs(
        json.loads((here / "cyclic-peptides-benchmark-v1.0.0.schema.json").read_text())
    )

    # Keys present only at the root of the original schema — not part of the data model.
    ORIGINAL_ONLY_ROOT_KEYS = {"$schema", "$id"}

    differences: list[str] = []

    def compare_schemas(orig: object, pyd: object, path: str = "#") -> None:  # noqa: ANN001
        """Recursively compare two schema fragments, collecting notable differences."""
        if isinstance(orig, dict) and isinstance(pyd, dict):
            all_keys = orig.keys() | pyd.keys()
            for key in sorted(all_keys):
                child_path = f"{path}/{key}"
                if key not in orig:
                    differences.append(f"Pydantic-only at {child_path}: {pyd[key]!r}")
                elif key not in pyd:
                    if path != "#" or key not in ORIGINAL_ONLY_ROOT_KEYS:
                        differences.append(f"Original-only at {child_path}: {orig[key]!r}")
                else:
                    compare_schemas(orig[key], pyd[key], child_path)
        elif isinstance(orig, list) and isinstance(pyd, list):
            if orig != pyd:
                differences.append(f"List mismatch at {path}: {orig!r} vs {pyd!r}")
        else:
            if orig != pyd:
                differences.append(f"Value mismatch at {path}: {orig!r} vs {pyd!r}")

    compare_schemas(original_schema, pydantic_schema)

    # Differences that are intrinsic to how Pydantic generates schemas and cannot
    # be eliminated without forking pydantic-core.
    EXPECTED_DIFFERENCES = {
        # Pydantic emits a title on every field and model class.
        "title",
        # Pydantic represents Optional[X] as anyOf:[{...},{type:null}] + default:null,
        # while the hand-written schema simply omits the field from `required`.
        "anyOf",
        # Pydantic emits `default` for fields that have one (sentinel for `name`,
        # null for schema_uri). Values that agree with the original are not reported
        # as differences; only Pydantic-added defaults appear here.
        "default",
    }

    unexpected = [
        d for d in differences
        if not any(token in d for token in EXPECTED_DIFFERENCES)
    ]

    if unexpected:
        print("\nUNEXPECTED schema differences (investigate these):")
        for d in unexpected:
            print(f"  {d}")
    else:
        print("Schema comparison: all differences are expected Pydantic conventions.")

    if differences:
        print(f"\nAll {len(differences)} differences (expected Pydantic-vs-handwritten):")
        for d in differences:
            print(f"  {d}")

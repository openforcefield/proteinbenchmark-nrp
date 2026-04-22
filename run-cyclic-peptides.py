#!/usr/bin/env python3

import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import shutil

import yaml

INITIALS = "am"
SCRIPT_PATH = Path("cyclic_peptides/cyclic_peptides_run.py")
STORAGE_PATHS= list(Path("cyclic_peptides").glob("*-storage.nc"))
CHECKPOINT_PATHS= list(Path("cyclic_peptides").glob("*-checkpoint.nc"))
FF="3.0.0-a0/OPC3"
TARGET_PATTERN = r"^.*/(.+)-storage.nc$"
LOCAL_RESULT_DIR = Path("cyclic_peptides/results")
N_REPLICATES = 3

def main():
    script_commit = get_script_commit(SCRIPT_PATH)
    with open("k8s_template.yaml") as f:
        template = yaml.safe_load(f)
    for src_storage, src_checkpoint in zip(STORAGE_PATHS, CHECKPOINT_PATHS, strict=True):
        match = re.match(TARGET_PATTERN, str(src_storage))
        assert match is not None, f"{TARGET_PATTERN!r} not in {str(src_storage)!r}"
        target = match.group(1)
        for replica in range(1, N_REPLICATES + 1):
            k8s_manifest_path = (
                LOCAL_RESULT_DIR
                / FF
                / f"{target}"
                / f"replica-{replica}"
                / f"{target}-{FF}-{replica}.yaml"
            )

            storage = Path(f"{target}/replica-{replica}/{src_storage.name}")
            checkpoint = Path(f"{target}/replica-{replica}/{src_checkpoint.name}")
            shutil.copyfile(src_storage, storage)
            shutil.copyfile(src_checkpoint, checkpoint)

            manifest = add_env_to_template(
                template,
                {
                    "PROTBENCH_REPLICA": replica,
                    "PROTBENCH_TARGET": target,
                    "PROTBENCH_FF": FF,
                    "PROTBENCH_WINDOW": 0,
                    "PROTBENCH_SCRIPT_COMMIT": script_commit,
                    "PROTBENCH_SCRIPT_PATH": SCRIPT_PATH,
                    "PROTBENCH_REQUIRED_FILES": "\n".join([
                        str(storage),
                        str(checkpoint),
                    ]),
                },
            )

            manifest.setdefault("metadata", {})["name"] = (
                f"pb-{INITIALS}-{target}-{FF}-{replica}".replace(".", "")
            )

            if "--dry-run" in sys.argv:
                yaml.safe_dump(
                    manifest,
                    sys.stdout,
                )
            else:
                k8s_manifest_path.parent.mkdir(parents=True, exist_ok=True)

                with open(k8s_manifest_path, "x") as f:
                    yaml.safe_dump(manifest, f)

                # subprocess.run(
                #     [
                #         "kubectl",
                #         "apply",
                #         "-f",
                #         k8s_manifest_path,
                #     ],
                #     check=True,
                # )


def get_script_commit(script_path: Path) -> str:
    script_is_ignored = (
        subprocess.run(
            ["git", "check-ignore", script_path],
            check=False,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )
    script_is_checked_in = (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", script_path],
            check=False,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )
    script_is_unmodified = (
        subprocess.run(
            ["git", "status", "--porcelain", script_path],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        == ""
    )

    if not (script_is_checked_in and script_is_unmodified) or script_is_ignored:
        print(script_is_checked_in, script_is_unmodified, script_is_ignored)
        raise ValueError(
            f"script {script_path} must be checked in to git so that the Kubernetes job can find it"
        )

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def get_containers(manifest):
    for container in manifest["spec"]["template"]["spec"].get("initContainers", []):
        yield container
    for container in manifest["spec"]["template"]["spec"].get("containers", []):
        yield container


def add_env_to_template(template: dict, envvars: dict[str, Any]) -> dict:
    output = deepcopy(template)
    for key, value in envvars.items():
        for container in get_containers(output):
            container.setdefault("env", []).append(
                {
                    "name": key,
                    "value": str(value),
                }
            )
    return output


if __name__ == "__main__":
    main()

import logging
from pathlib import Path

import cyclopts
import openmm
import openmmtools

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s via %(name)s)",
)


def main(storage: Path, checkpoint: Path):
    LOGGER.info(f"Loading sampler from {storage}")
    reporter = openmmtools.multistate.MultiStateReporter(
        storage=str(storage),
        checkpoint_storage=str(checkpoint),
    )
    sampler = openmmtools.multistate.ReplicaExchangeSampler.from_storage(reporter)

    LOGGER.info("Configuring platform")
    platform = openmm.Platform.getPlatformByName("CUDA")
    context_cache_dict = dict(
        platform=platform,
        platform_properties={"Precision": "mixed"},
        capacity=None,
        time_to_live=None,
    )
    sampler.energy_context_cache = openmmtools.cache.ContextCache(
        **context_cache_dict,
    )
    sampler.sampler_context_cache = openmmtools.cache.ContextCache(
        **context_cache_dict,
    )

    LOGGER.info("Beginning production run")
    sampler.run()


if __name__ == "__main__":
    cyclopts.run(main)

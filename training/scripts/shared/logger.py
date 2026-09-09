import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import torch.distributed as dist

if TYPE_CHECKING:
    from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig


class RankZeroFilter(logging.Filter):
    """Optional: Restrict log records to Rank 0 in distributed setups."""

    def filter(self, record):
        return int(os.environ.get("RANK", 0)) == 0


class RankAdapter(logging.LoggerAdapter):
    """Automatically prefixes every log message with the current process rank."""

    def process(self, msg, kwargs):
        rank = os.environ.get("RANK", "0")
        return f"[Rank {rank}] {msg}", kwargs


def setup_logging(
    cfg: "BenchmarkConfig",
    level=logging.INFO,
):
    rank = int(os.environ.get("RANK", 0))
    RUNID = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d%H%M%S"))
    RUNJD = os.environ.get("SLURM_STEP_ID", "0")
    LOG_DIR = os.path.join(
        os.environ.get("LOG_DIR", os.path.join("outputs", "logs", "pyft")),
        RUNID,
        f"{cfg.model.name}_{cfg.framework.name}_{cfg.framework.parallelism_name}_{cfg.dataset.name}_nodes{cfg.slurm.sbatch.nodes}",
    )

    # Prevent filesystem race condition: let Rank 0 create the directory
    if rank == 0:
        os.makedirs(LOG_DIR, exist_ok=True)

    # Wait for Rank 0 to finish creating the directory if using torch.distributed
    if dist.is_initialized():
        dist.barrier()

    # Configure parent namespace logger instead of root
    parent_logger = logging.getLogger(
        "MINERVA_BENCH"
    )  # Replace with your top-level project name
    parent_logger.setLevel(level)

    if parent_logger.hasHandlers():
        return parent_logger

    log_file = os.path.join(LOG_DIR, f"minerva-step{RUNJD}.log")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s : %(message)s")
    )
    file_handler.addFilter(RankZeroFilter())  # Ensure ONLY Rank 0 writes to disk

    parent_logger.addHandler(file_handler)

    return parent_logger

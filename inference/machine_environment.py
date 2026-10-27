"""Shared machine selection for inference result tools."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def load_machine_environment(base_dir):
    """Load .env-<machine>, chosen by --machine or $MACHINE."""
    parser = argparse.ArgumentParser(
        description="Process inference results for a supercomputer configuration."
    )
    parser.add_argument(
        "--machine",
        default=os.getenv("MACHINE"),
        help="Configuration name, e.g. leonardo or bsc-mn5-acc (default: $MACHINE).",
    )
    machine = parser.parse_args().machine
    if not machine:
        parser.error("Select the configuration with --machine or export MACHINE.")

    env_file = Path(base_dir) / f".env-{machine}"
    if not env_file.is_file():
        parser.error(f"Missing configuration file for {machine}: {env_file}")
    load_dotenv(env_file, override=True)

    for key in ("SUPCOMPUTER_NAME", "PARTITION_NAME"):
        if not os.getenv(key, "").strip():
            parser.error(f"{key} is not set in {env_file}")
    return machine

"""Shared machine selection for inference result tools."""

import argparse
import os
import socket
from pathlib import Path

from dotenv import load_dotenv


MACHINES = (
    "leonardo",
    "bsc-mn5-acc",
    "bsc-mn5-gpp",
    "cines-adastra-mi250",
    "cines-adastra-mi300",
    "idris-jeanzay-h100",
    "csc-lumi-gpu",
)


def detect_machine():
    hostname = socket.gethostname().lower()
    if "leonardo" in hostname:
        return "leonardo"
    if "jean-zay" in hostname:
        return "idris-jeanzay-h100"
    if hostname == "lumi" or hostname.startswith(("lumi.", "lumi-")):
        return "csc-lumi-gpu"
    if "adastra" in hostname:
        raise ValueError(
            "Adastra detected, but MI250/MI300 cannot be determined. "
            "Use --machine cines-adastra-mi250 or --machine cines-adastra-mi300."
        )
    if os.getenv("BSC_MACHINE", "").lower() == "mn5":
        raise ValueError(
            "MN5 detected, but ACC/GPP cannot be determined. "
            "Use --machine bsc-mn5-acc or --machine bsc-mn5-gpp."
        )
    raise ValueError(
        f"Cannot determine the machine configuration for host {hostname!r}. "
        "Select it explicitly with --machine."
    )


def load_machine_environment(base_dir):
    parser = argparse.ArgumentParser(
        description="Process inference results for a supercomputer configuration."
    )
    parser.add_argument(
        "--machine",
        choices=MACHINES,
        help="Configuration to use; otherwise try automatic host detection.",
    )
    args = parser.parse_args()
    try:
        machine = args.machine or detect_machine()
    except ValueError as error:
        parser.error(str(error))

    base_dir = Path(base_dir)
    env_file = base_dir / f".env-{machine}"
    model_map = base_dir / f"configs-{machine}" / "model_type_map.json"
    for path in (env_file, model_map):
        if not path.is_file():
            parser.error(f"Missing configuration file for {machine}: {path}")

    # The selected file must not inherit metadata from a previously sourced machine.
    load_dotenv(env_file, override=True)
    os.environ["MACHINE"] = machine
    return machine

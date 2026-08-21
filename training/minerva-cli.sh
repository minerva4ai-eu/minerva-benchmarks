#!/bin/bash

USE_SINGULARITY=0
cli_args="$@"

VENV_PATH="envs/cli/.venv/bin/activate"
source "$VENV_PATH"
python -m scripts.slurm.cli $cli_args

exit 0
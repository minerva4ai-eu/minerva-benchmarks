#!/bin/bash

cli_args="$@"

#VENV_PATH="envs/cli/.venv/bin/activate"
#source "$VENV_PATH"
source "scripts/shared/runtime_environment.sh"
PYTHON=$(get_cli_python)
echo "PYTHON: $PYTHON"

# Create logs directory and set log file path
#LOG_DIR="outputs/logs/$(date '+%Y-%m-%d')"
#mkdir -p "$LOG_DIR"

$PYTHON -m scripts.slurm.cli $cli_args 
exit 0
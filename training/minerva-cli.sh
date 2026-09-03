#!/bin/bash

cli_args="$@"

VENV_PATH="envs/cli/.venv/bin/activate"
source "$VENV_PATH"

# Create logs directory and set log file path
LOG_DIR="cli-logs/$(date '+%Y-%m-%d')"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date '+%H:%M:%S').log"

export PYTHONUNBUFFERED=1

# Use 'script' if available (forces TTY → full colors on terminal, strips ANSI for log file).
# Otherwise fall back to plain logging (no color stripping).
if command -v script &>/dev/null; then
    script -q -c "python -m scripts.slurm.cli $cli_args" /dev/null | tee >(
        sed -u -E 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\r//g' > "$LOG_FILE"
    )
else
    python -m scripts.slurm.cli $cli_args 2>&1 | tee "$LOG_FILE"
fi
exit 0
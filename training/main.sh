#!/bin/bash

# MINERVA Benchmarks CLI Shell Script

# Source common configuration
source config.sh

# Default values
export DRY_RUN=false
export CONFIGS_PATH="."
CONFIG_NAME=default
export CONFIG_ENV=singularity
export OUT_DIR="benchmark-runs"
export MINI_MODE=false

# Function to display help
show_help() {
    echo "MINERVA Benchmarks CLI"
    echo ""
    echo "Usage: ./main.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  submit   Submit new benchmark jobs"
    echo "  rerun    Rerun failed/pending jobs"
    echo "  status   Check status of a run"
    echo "  cancel   Cancel jobs of running/pending jobs of a run"
    echo "  help     Show this help"
    echo ""
    echo "Use './minerva-cli.sh [COMMAND] --help' for more information about a command."
}


# If no arguments provided, show help
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Parse the first argument as command
command="$1"

case "$command" in
    submit)
        shift
        ./submit.sh "$@"
        ;;
    rerun)
        shift
        # ./rerun.sh "$@"
        echo rerun not yet implemented
        ;;
    status)
        shift
        # ./status.sh "$@"
        echo status not yet implemented
        ;;
    cancel)
        shift
        ./cancel.sh "$@"
        echo cancel not yet implemented
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $command${RESET}" >&2
        show_help
        exit 1
        ;;
esac
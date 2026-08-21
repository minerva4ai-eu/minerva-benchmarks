#!/bin/bash

# MINERVA Benchmarks CLI Shell Script

# Source common configuration
source config.sh

# Default values
export DRY_RUN=false
export CONFIGS_PATH="."
CONFIG_NAME=default
# export CONFIG_ENV=singularity
export OUT_DIR="benchmark-runs"
export DEBUG=false

# Function to display help
show_help() {
    echo "MINERVA Benchmarks CLI"
    echo ""
    echo "Usage: ./main.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  submit   Submit new benchmark job"
    echo "  status   NotYetImplemented"
    echo "  cancel   NotYetImplemented"
    echo "  help     Show this help"
    echo ""
    echo "Use './main.sh [COMMAND] --help' for more information about a command."
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
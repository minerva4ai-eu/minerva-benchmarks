#!/bin/bash

training_detect_execution_mode() {
    case "${EXECUTION_MODE:-auto}" in
        singularity|SINGULARITY)
            echo "singularity"
            ;;
        venv|virtualenv|VIRTUALENV)
            echo "venv"
            ;;
        host|HOST|none|NONE)
            echo "host"
            ;;
        auto|AUTO|"")
            if [[ -n "${SINGULARITY_CONTAINER:-}" ]]; then
                echo "singularity"
            elif [[ -n "${ENVIRONMENT_FINETUNING:-}" ]]; then
                echo "venv"
            else
                echo "host"
            fi
            ;;
        *)
            echo "No EXECUTION_MODE auto detected, defaulting to 'host'"
            ;;
    esac
}

training_build_runtime_prefix() {
    local execution_mode
    execution_mode="$(training_detect_execution_mode)"

    case "$execution_mode" in
        singularity)
            if [[ -z "${SINGULARITY_CONTAINER:-}" ]]; then
                echo "Singularity execution requested but SINGULARITY_CONTAINER is empty." >&2
                return 1
            fi

            local runtime_prefix="singularity exec"
            if [[ -n "${SINGULARITY_ARGS:-}" ]]; then
                runtime_prefix+=" ${SINGULARITY_ARGS}"
            fi
            if [[ -n "${SINGULARITY_BINDS:-}" ]]; then
                runtime_prefix+=" ${SINGULARITY_BINDS}"
            fi
            runtime_prefix+=" --home ${LAUNCH_FOLDER:-$PWD} ${SINGULARITY_CONTAINER}"
            echo "$runtime_prefix"
            ;;
        venv|host)
            echo ""
            ;;
        *)
            echo "Unknown execution mode: $execution_mode" >&2
            return 1
            ;;
    esac
}

training_activate_runtime_environment() {
    local execution_mode
    execution_mode="$(training_detect_execution_mode)"

    case "$execution_mode" in
        venv)
            if [[ -z "${ENVIRONMENT_FINETUNING:-}" ]]; then
                echo "Virtualenv execution requested but ENVIRONMENT_FINETUNING is empty." >&2
                return 1
            fi
            if [[ ! -f "${ENVIRONMENT_FINETUNING}/bin/activate" ]]; then
                echo "Virtualenv activation script not found: ${ENVIRONMENT_FINETUNING}/bin/activate" >&2
                return 1
            fi
            # shellcheck disable=SC1090
            source "${ENVIRONMENT_FINETUNING}/bin/activate"
            ;;
        singularity|host)
            ;;
        *)
            echo "Unknown execution mode: $execution_mode" >&2
            return 1
            ;;
    esac
}
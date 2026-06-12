#!/bin/bash

#######################################################
# RESUBMIT ONLY FAILED BENCHMARKS
#######################################################
# A launch is resubmitted ONLY if ALL of these hold:
#   1. Its launch folder exists (it was attempted before).
#   2. It is not complete (some Concurrency_*.json missing).
#   3. It has no job currently PENDING/RUNNING (checked via squeue workdir).
#   4. Its most recent Slurm job (looked up by workdir via sacct)
#      ended in a failure state: FAILED, TIMEOUT, OUT_OF_MEMORY,
#      NODE_FAIL, BOOT_FAIL, DEADLINE, PREEMPTED.
#      Jobs that were CANCELLED before ever starting are ignored when
#      determining the latest state: they say nothing about the
#      benchmark itself (e.g. a duplicate resubmission killed while
#      queued must not mask an earlier real failure).
# Everything else (CANCELLED, COMPLETED, pending, running, unknown)
# is skipped. If squeue or sacct cannot be reached after retries the
# script ABORTS instead of guessing.
#
# Preview without submitting anything:
#   DRY_RUN=true ./run_failed_benchmaks.sh
#######################################################
FRAMEWORKS=("vllm" "sglang" "deepspeed")
DATASETS=("sharegpt" "sonnet")
MODELS=("Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct" "Llama-3.1-405B-Instruct" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3" "ALIA-40b-instruct-2605")
NUMBER_OF_NODES=(1 2 4)
MAX_MODEL_LENGTHS=(4096 16384 32768)
REPEATS=3
MACHINE="leonardo"
MACHINE_TYPE="cuda" # "cuda" or "rocm"
# When true, only relaunch combinations that already have a launch folder.
ONLY_EXISTING_LAUNCHES=true
# When true, launches whose latest job was CANCELLED are also resubmitted.
RETRY_CANCELLED=false
# How far back sacct looks for job history.
SACCT_WINDOW="${SACCT_WINDOW:-now-60days}"
# When true, print what would be submitted without touching anything.
DRY_RUN="${DRY_RUN:-false}"
#######################################################

set -a
source .env-$MACHINE
set +a

source scripts/utils.sh

EXPECTED_CONCURRENCIES=(150 250 300 500 1000)

declare -A ACTIVE_LAUNCHES   # workdir -> job id currently pending/running
declare -A LATEST_JOB        # workdir -> most recent job id known to sacct
declare -A LATEST_STATE      # workdir -> state of that job
declare -A JOB_STATE_CACHE   # job id  -> state (fallback lookups)
declare -A SKIP_COUNTS

# Run a command, retrying on failure (slurmdbd on this machine fails
# intermittently with "Resource temporarily unavailable").
run_with_retries() {
  local attempt
  for attempt in 1 2 3; do
    if "$@" 2>/dev/null; then
      return 0
    fi
    sleep 3
  done
  return 1
}

load_slurm_state() {
  local out job_id state workdir

  if ! out=$(run_with_retries squeue -h -u "$USER" -t PD,R,CF,CG,S -o "%A %Z"); then
    echo "ERROR: squeue not responding after retries; aborting to avoid duplicate submissions." >&2
    exit 1
  fi
  while read -r job_id workdir; do
    [[ -z "$workdir" ]] && continue
    ACTIVE_LAUNCHES["$workdir"]="$job_id"
  done <<< "$out"

  if ! out=$(run_with_retries sacct -u "$USER" -X -S "$SACCT_WINDOW" --format=JobID,State,Start,WorkDir -n -P); then
    echo "ERROR: sacct (accounting DB) not responding after retries; aborting to avoid misclassifying jobs." >&2
    exit 1
  fi
  while IFS='|' read -r job_id state start workdir; do
    [[ -z "$workdir" ]] && continue
    job_id="${job_id%%[!0-9]*}"
    [[ -z "$job_id" ]] && continue
    # Cancelled before it ever started: carries no information about the
    # benchmark, do not let it shadow an earlier real failure.
    if [[ "$state" == CANCELLED* && ( -z "$start" || "$start" == "Unknown" || "$start" == "None" ) ]]; then
      continue
    fi
    if (( job_id > ${LATEST_JOB["$workdir"]:-0} )); then
      LATEST_JOB["$workdir"]="$job_id"
      LATEST_STATE["$workdir"]="$state"
    fi
  done <<< "$out"
}

is_launch_complete() {
  local launch_folder="$1"
  local conc

  for conc in "${EXPECTED_CONCURRENCIES[@]}"; do
    if [[ ! -s "$launch_folder/Concurrency_${conc}.json" ]]; then
      return 1
    fi
  done

  return 0
}

# Fallback for attempts older than SACCT_WINDOW: job ids from run-*.out files.
get_latest_launch_job_id() {
  local launch_folder="$1"
  local file
  local job_id
  local latest=0

  shopt -s nullglob
  for file in "$launch_folder"/run-*.out "$launch_folder"/run-*.err; do
    job_id="${file##*/run-}"
    job_id="${job_id%%.*}"
    if [[ "$job_id" =~ ^[0-9]+$ ]] && (( job_id > latest )); then
      latest=$job_id
    fi
  done
  shopt -u nullglob

  if (( latest > 0 )); then
    echo "$latest"
  fi
}

# Sets RESOLVED_STATE to the latest known Slurm state for this launch
# ("" if no job history can be found at all).
resolve_latest_state() {
  local launch_folder="$1"
  local job_id

  RESOLVED_STATE="${LATEST_STATE[$launch_folder]}"
  [[ -n "$RESOLVED_STATE" ]] && return

  job_id=$(get_latest_launch_job_id "$launch_folder")
  [[ -z "$job_id" ]] && return

  if [[ -z "${JOB_STATE_CACHE[$job_id]+x}" ]]; then
    JOB_STATE_CACHE["$job_id"]=$(run_with_retries sacct -j "$job_id" --format=State -n -P | head -n 1 | cut -d'|' -f1)
  fi
  RESOLVED_STATE="${JOB_STATE_CACHE[$job_id]}"
}

# Returns 0 (skip) with SKIP_REASON set, or 1 (resubmit) with SUBMIT_STATE set.
should_skip_launch() {
  local launch_folder="$1"

  if [[ "$ONLY_EXISTING_LAUNCHES" == true && ! -d "$launch_folder" ]]; then
    SKIP_REASON="not_attempted"
    return 0
  fi

  if is_launch_complete "$launch_folder"; then
    SKIP_REASON="completed"
    return 0
  fi

  if [[ -n "${ACTIVE_LAUNCHES[$launch_folder]}" ]]; then
    SKIP_REASON="pending_or_running"
    return 0
  fi

  resolve_latest_state "$launch_folder"

  case "$RESOLVED_STATE" in
    FAILED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|BOOT_FAIL*|DEADLINE*|PREEMPTED*)
      SUBMIT_STATE="$RESOLVED_STATE"
      return 1
      ;;
    CANCELLED*)
      if [[ "$RETRY_CANCELLED" == true ]]; then
        SUBMIT_STATE="$RESOLVED_STATE"
        return 1
      fi
      SKIP_REASON="cancelled"
      return 0
      ;;
    COMPLETED*)
      # Job exited 0 but result files are missing: needs manual inspection.
      SKIP_REASON="completed_job_but_missing_results"
      return 0
      ;;
    "")
      SKIP_REASON="no_job_history"
      return 0
      ;;
    *)
      SKIP_REASON="state_${RESOLVED_STATE%% *}"
      return 0
      ;;
  esac
}

CURRENT_DIR=$(pwd)
SUBMITTED=0

load_slurm_state
echo "Loaded ${#ACTIVE_LAUNCHES[@]} active (pending/running) launches and ${#LATEST_JOB[@]} launch folders with job history (window: $SACCT_WINDOW)."
[[ "$DRY_RUN" == true ]] && echo "DRY RUN: nothing will be submitted."

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for NODES in "${NUMBER_OF_NODES[@]}"; do
        if [[ "$NODES" -eq 1 ]]; then
          GPU_CONFIGS=(1 $GPUS_PER_NODE)
        else
          GPU_CONFIGS=($GPUS_PER_NODE)
        fi

        for GPU_NODE in "${GPU_CONFIGS[@]}"; do
          for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
            TOTAL_GPUS=$((NODES * GPU_NODE))
            TOTAL_CPUS=$((GPUS_PER_NODE * CPUS_PER_GPU))
            TENSOR_PARALLEL=$GPU_NODE
            PIPELINE_PARALLEL=$NODES

            BASE_FOLDER="results/${framework}/${dataset}/${model}"
            RUN_FOLDER="Nodes_${NODES}-GPUs_${TOTAL_GPUS}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}-MaxModelLength_${MAX_MODEL_LENGTH}"
            FULL_FOLDER="${BASE_FOLDER}/${RUN_FOLDER}"

            MODEL_TYPE=$(get_model_type "$model" "configs-$MACHINE/model_type_map.json")
            MODEL_DIRECTORY=$(get_model_directory "$MODEL_TYPE" "configs-$MACHINE/model_type_directories_map.json")
            MODEL_PATH="${MODEL_DIRECTORY}/${model}"

            if [[ -z "$MODEL_DIRECTORY" ]]; then
              echo "Unknown model '$model' (resolved type '$MODEL_TYPE') or missing directory mapping. Exiting."
              exit 1
            fi

            DATASET_PATH=$(get_dataset_path "$dataset" "configs-$MACHINE/config_datasets_paths_map.json")

            if [[ "$framework" == "vllm" ]]; then
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
                if [[ "$NODES" -lt 4 ]]; then
                  continue
                fi
              fi
              ADDITIONAL_ARGS="--disable-log-requests"

              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"

                if should_skip_launch "$LAUNCH_FOLDER"; then
                  echo "Skipping ($SKIP_REASON): $LAUNCH_FOLDER"
                  ((SKIP_COUNTS[$SKIP_REASON]++))
                  continue
                fi

                if [[ "$DRY_RUN" == true ]]; then
                  echo "[DRY-RUN] Would resubmit (last state: $SUBMIT_STATE): $LAUNCH_FOLDER"
                  ((SUBMITTED++))
                  continue
                fi

                echo "Resubmitting (last state: $SUBMIT_STATE): $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"

                cp scripts/vllm/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/vllm/run_mp_vllm.sh "$LAUNCH_FOLDER"

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH
                export ADDITIONAL_ARGS
                export MODULES
                export MACHINE
                export MACHINE_TYPE
                export CPUS_PER_NODE VRAM_PER_NODE
                export CUR_DIR=$(pwd)

                JOB_ID=$(sbatch --parsable \
                    --chdir=$(pwd) \
                    --nodes=$NODES \
                    --gres=gpu:$GPUS_PER_NODE \
                    --cpus-per-task=$TOTAL_CPUS \
                    --output=run-%j.out \
                    --error=run-%j.err \
                    -A $ACCOUNT \
                    -q $QOS \
                    --time=$TIME_LIMIT \
                    --partition=$PARTITION_NAME \
                    run_mp_vllm.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH" "$MACHINE" "$MACHINE_TYPE")

                echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
                ACTIVE_LAUNCHES["$LAUNCH_FOLDER"]="$JOB_ID"
                ((SUBMITTED++))

                cd - > /dev/null
                sleep 5
              done
            fi

            if [[ "$framework" == "deepspeed" ]]; then
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" || "$model" == "Llama-3.70B-Instruct" ]]; then
                continue
              fi
              if [[ "$model" == "gemma-3-12b-it" ]]; then
                continue
              fi
              if [[ "$model" == "Mistral-7B-Instruct-v0.3" ]]; then
                continue
              fi
              if [[ "$NODES" != 1 ]]; then
                continue
              fi
              ADDITIONAL_ARGS=""

              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"

                if should_skip_launch "$LAUNCH_FOLDER"; then
                  echo "Skipping ($SKIP_REASON): $LAUNCH_FOLDER"
                  ((SKIP_COUNTS[$SKIP_REASON]++))
                  continue
                fi

                if [[ "$DRY_RUN" == true ]]; then
                  echo "[DRY-RUN] Would resubmit (last state: $SUBMIT_STATE): $LAUNCH_FOLDER"
                  ((SUBMITTED++))
                  continue
                fi

                echo "Resubmitting (last state: $SUBMIT_STATE): $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"

                cp scripts/deepspeed/serve_deepspeed_mii.py "$LAUNCH_FOLDER"
                cp scripts/deepspeed/deepspeed-mii_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
                cp scripts/deepspeed/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH
                export ADDITIONAL_ARGS
                export MODULES
                export MACHINE
                export MACHINE_TYPE

                JOB_ID=$(sbatch --parsable \
                    --chdir=$(pwd) \
                    --nodes=$NODES \
                    --gres=gpu:$GPUS_PER_NODE \
                    --cpus-per-task=$TOTAL_CPUS \
                    --output=run-%j.out \
                    --error=run-%j.err \
                    -A $ACCOUNT \
                    -q $QOS \
                    --time=$TIME_LIMIT \
                    --partition=$PARTITION_NAME \
                    deepspeed-mii_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

                echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
                ACTIVE_LAUNCHES["$LAUNCH_FOLDER"]="$JOB_ID"
                ((SUBMITTED++))

                cd - > /dev/null
                sleep 5
              done
            fi

            if [[ "$framework" == "sglang" ]]; then
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
                if [[ "$NODES" -lt 4 ]]; then
                  continue
                fi
              fi
              if [[ "$model" == "gemma-3-12b-it" && "$NODES" -gt 1 ]]; then
                continue
              fi
              ADDITIONAL_ARGS=""

              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"

                if should_skip_launch "$LAUNCH_FOLDER"; then
                  echo "Skipping ($SKIP_REASON): $LAUNCH_FOLDER"
                  ((SKIP_COUNTS[$SKIP_REASON]++))
                  continue
                fi

                if [[ "$DRY_RUN" == true ]]; then
                  echo "[DRY-RUN] Would resubmit (last state: $SUBMIT_STATE): $LAUNCH_FOLDER"
                  ((SUBMITTED++))
                  continue
                fi

                echo "Resubmitting (last state: $SUBMIT_STATE): $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"

                cp scripts/sglang/sglang_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
                cp scripts/sglang/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/sglang/serve.sh "$LAUNCH_FOLDER"
                cp scripts/sglang/wrapper_singularity.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH
                export ADDITIONAL_ARGS
                export MODULES
                export MACHINE
                export MACHINE_TYPE

                JOB_ID=$(sbatch --parsable \
                    --chdir=$(pwd) \
                    --nodes=$NODES \
                    --gres=gpu:$GPUS_PER_NODE \
                    --cpus-per-task=$TOTAL_CPUS \
                    --output=run-%j.out \
                    --error=run-%j.err \
                    -A $ACCOUNT \
                    -q $QOS \
                    --time=$TIME_LIMIT \
                    --partition=$PARTITION_NAME \
                    sglang_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

                echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
                ACTIVE_LAUNCHES["$LAUNCH_FOLDER"]="$JOB_ID"
                ((SUBMITTED++))

                cd - > /dev/null
                sleep 5
              done
            fi
          done
        done
      done
    done
  done
done

echo ""
echo "Summary: submitted=$SUBMITTED"
for reason in "${!SKIP_COUNTS[@]}"; do
  echo "  skipped ($reason): ${SKIP_COUNTS[$reason]}"
done

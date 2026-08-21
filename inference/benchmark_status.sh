#!/bin/bash

#######################################################
# STATUS TABLE FOR THE INFERENCE BENCHMARK SUITE
#######################################################
# Read-only: classifies every launch of the benchmark matrix as
#   completed     all Concurrency_*.json files present
#   running       job currently RUNNING (squeue, matched by workdir)
#   pending       job currently queued (squeue, matched by workdir)
#   failed        latest real attempt ended in FAILED/TIMEOUT/OOM/...
#   cancelled     latest real attempt was cancelled after starting
#   no_history    folder exists but no job ever actually ran
#   not_attempted no launch folder
# Jobs cancelled before they ever started are ignored, exactly like
# in run_failed_benchmaks.sh.
#######################################################
FRAMEWORKS=("vllm" "sglang" "deepspeed")
DATASETS=("sharegpt" "sonnet")
MODELS=("Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct" "Llama-3.1-405B-Instruct" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3" "ALIA-40b-instruct-2605")
NUMBER_OF_NODES=(1 2 4)
MAX_MODEL_LENGTHS=(4096 16384 32768)
REPEATS=3
MACHINE="leonardo"
SACCT_WINDOW="${SACCT_WINDOW:-now-60days}"
#######################################################

set -a
source .env-$MACHINE
set +a

EXPECTED_CONCURRENCIES=(150 250 300 500 1000)
STATUSES=(completed running pending failed cancelled no_history not_attempted)

declare -A QUEUE_STATE     # workdir -> RUNNING / PENDING / ...
declare -A LATEST_JOB      # workdir -> most recent started job id
declare -A LATEST_STATE    # workdir -> state of that job
declare -A COUNTS          # "fw|ds|model|status" -> count
declare -A TOTALS          # status -> count

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
  local out job_id state start workdir

  if ! out=$(run_with_retries squeue -h -u "$USER" -t PD,R,CF,CG,S -o "%T %Z"); then
    echo "ERROR: squeue not responding after retries." >&2
    exit 1
  fi
  while read -r state workdir; do
    [[ -z "$workdir" ]] && continue
    QUEUE_STATE["$workdir"]="$state"
  done <<< "$out"

  if ! out=$(run_with_retries sacct -u "$USER" -X -S "$SACCT_WINDOW" --format=JobID,State,Start,WorkDir -n -P); then
    echo "ERROR: sacct (accounting DB) not responding after retries." >&2
    exit 1
  fi
  while IFS='|' read -r job_id state start workdir; do
    [[ -z "$workdir" ]] && continue
    job_id="${job_id%%[!0-9]*}"
    [[ -z "$job_id" ]] && continue
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

classify_launch() {
  local launch_folder="$1"

  if [[ ! -d "$launch_folder" ]]; then
    echo "not_attempted"
    return
  fi
  if is_launch_complete "$launch_folder"; then
    echo "completed"
    return
  fi
  case "${QUEUE_STATE[$launch_folder]}" in
    RUNNING|COMPLETING|CONFIGURING|SUSPENDED)
      echo "running"
      return
      ;;
    PENDING)
      echo "pending"
      return
      ;;
  esac
  case "${LATEST_STATE[$launch_folder]}" in
    FAILED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|BOOT_FAIL*|DEADLINE*|PREEMPTED*)
      echo "failed"
      ;;
    CANCELLED*)
      echo "cancelled"
      ;;
    "")
      echo "no_history"
      ;;
    *)
      # e.g. COMPLETED job but missing result files
      echo "no_history"
      ;;
  esac
}

CURRENT_DIR=$(pwd)
load_slurm_state

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
            TENSOR_PARALLEL=$GPU_NODE
            PIPELINE_PARALLEL=$NODES

            # Same combination exclusions as the submission scripts.
            if [[ "$framework" == "vllm" || "$framework" == "sglang" ]]; then
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]] && [[ "$NODES" -lt 4 ]]; then
                continue
              fi
            fi
            if [[ "$framework" == "sglang" && "$model" == "gemma-3-12b-it" && "$NODES" -gt 1 ]]; then
              continue
            fi
            if [[ "$framework" == "deepspeed" ]]; then
              case "$model" in
                Llama-3.1-405B|Llama-3.1-405B-Instruct|Llama-3.70B-Instruct|gemma-3-12b-it|Mistral-7B-Instruct-v0.3) continue ;;
              esac
              [[ "$NODES" != 1 ]] && continue
            fi

            RUN_FOLDER="Nodes_${NODES}-GPUs_${TOTAL_GPUS}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}-MaxModelLength_${MAX_MODEL_LENGTH}"

            for (( run_id=1; run_id<=REPEATS; run_id++ )); do
              LAUNCH_FOLDER="${CURRENT_DIR}/results/${framework}/${dataset}/${model}/${RUN_FOLDER}/launch-${run_id}"
              status=$(classify_launch "$LAUNCH_FOLDER")
              key="${framework}|${dataset}|${model}"
              ((COUNTS["$key|$status"]++))
              ((TOTALS["$status"]++))
            done
          done
        done
      done
    done
  done
done

fmt="%-10s %-9s %-26s %10s %8s %8s %7s %10s %11s %14s %6s\n"
printf "$fmt" FRAMEWORK DATASET MODEL COMPLETED RUNNING PENDING FAILED CANCELLED NO_HISTORY NOT_ATTEMPTED TOTAL
printf "$fmt" --------- ------- ----- --------- ------- ------- ------ --------- ---------- ------------- -----

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      key="${framework}|${dataset}|${model}"
      row_total=0
      declare -A c=()
      for s in "${STATUSES[@]}"; do
        c[$s]=${COUNTS["$key|$s"]:-0}
        ((row_total += c[$s]))
      done
      (( row_total == 0 )) && continue
      printf "$fmt" "$framework" "$dataset" "$model" \
        "${c[completed]}" "${c[running]}" "${c[pending]}" "${c[failed]}" \
        "${c[cancelled]}" "${c[no_history]}" "${c[not_attempted]}" "$row_total"
    done
  done
done

printf "$fmt" --------- ------- ----- --------- ------- ------- ------ --------- ---------- ------------- -----
grand_total=0
for s in "${STATUSES[@]}"; do ((grand_total += ${TOTALS[$s]:-0})); done
printf "$fmt" TOTAL "" "" \
  "${TOTALS[completed]:-0}" "${TOTALS[running]:-0}" "${TOTALS[pending]:-0}" "${TOTALS[failed]:-0}" \
  "${TOTALS[cancelled]:-0}" "${TOTALS[no_history]:-0}" "${TOTALS[not_attempted]:-0}" "$grand_total"

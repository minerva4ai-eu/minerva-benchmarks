#!/bin/bash

scontrol show step $SLURM_JOB_ID.$SLURM_STEP_ID
nvidia-smi

##################################################
### Environment Variables Setup
##################################################

LAUNCH_FOLDER=$1
BENCHMARK_FILE=$2
DATASET=$3
DATASET_PATH=$4
MODEL_PATH=$5
MAX_MODEL_LEN=$6

echo "LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "BENCHMARK_FILE: $BENCHMARK_FILE"
echo "DATASET: $DATASET"
echo "DATASET_PATH: $DATASET_PATH"
echo "MODEL_PATH: $MODEL_PATH"
echo "VLLM_IMAGE: $VLLM_IMAGE"

# NCCL
source $SCRIPTS_DIR/activate-env-variables-per-supercomputer.sh

# vLLM variables
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

##################################################
### Run vLLM inside Singularity
##################################################

TP_SIZE=$TENSOR_PARALLEL
PP_SIZE=$PIPELINE_PARALLEL
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-0}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-0}
export ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL ENABLE_EXPERT_PARALLEL ENFORCE_EAGER DISABLE_CUSTOM_ALL_REDUCE

echo TP_SIZE= $TP_SIZE
echo PP_SIZE= $PP_SIZE
echo MAX_MODEL_LEN= $MAX_MODEL_LEN
# Disable allreduce+rms fusion path that can require multicast-capable symmetric memory
#export COMPILATION_CONFIG='{"pass_config":{"fuse_allreduce_rms":true}}'

# Get Head HOST
export HEAD_NODE=$(scontrol show hostnames ${SLURM_STEP_NODELIST} | head -n 1)
echo "[INFO] Head Node: $HEAD_NODE"

export MASTER_PORT=29500
echo MASTER_PORT= $MASTER_PORT

export NUM_NODES=$SLURM_STEP_NUM_NODES

##################################################
### RUN vLLM SERVE 
##################################################
echo "BINDINGS_SINGULARITY: $BINDINGS_SINGULARITY"

#  --env VLLM_ENABLE_CUDA_COMPATIBILITY="1" \
#  --env VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-13.0/compat" \
#  --env LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat" \
#  --env TRITON_PTXAS_PATH="/usr/local/cuda-13.0/compat/ptxas" \
# setsid: run the server in its own process group so it (and all its spawned
# workers) can be signaled as a unit at shutdown without touching this script.
setsid singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS \
  -B "$MODEL_PATH":"$MODEL_PATH" \
  --env LC_ALL="C" \
  --env LANG="C.UTF-8" \
  "$VLLM_IMAGE" \
  bash -c '
      echo "SLURM_NODEID=$SLURM_NODEID"
      echo "SLURM_PROCID=$SLURM_PROCID"
      echo "SLURM_LOCALID=$SLURM_LOCALID"
      NODE_RANK=$SLURM_NODEID
      HOST_IP=$(ip -o -4 addr show "$NET_IFACE" 2>/dev/null | awk "{split(\$4,a,\"/\"); print a[1]; exit}")
      if [ -z "$HOST_IP" ]; then
          HOST_IP=$(hostname -I | awk "{print \$1}")
      fi
      export VLLM_HOST_IP=$HOST_IP

      ENGINE_EXTRA_ARGS=()
      if [ "$ENABLE_PREFIX_CACHING" -eq 1 ]; then
          ENGINE_EXTRA_ARGS+=(--enable-prefix-caching)
      fi
      if [ "$ENABLE_CHUNKED_PREFILL" -eq 1 ]; then
          ENGINE_EXTRA_ARGS+=(--enable-chunked-prefill)
      fi
      ENGINE_EXTRA_ARGS+=(--enforce-eager)
      if [ "$ENABLE_EXPERT_PARALLEL" -eq 1 ]; then
          ENGINE_EXTRA_ARGS+=(--enable-expert-parallel)
      fi
      if [ "$DISABLE_CUSTOM_ALL_REDUCE" -eq 1 ]; then
          ENGINE_EXTRA_ARGS+=(--disable-custom-all-reduce)
      fi
      echo $MACHINE_TYPE
      # Error without this flag: [multiproc_executor.py:888] ValueError: Available memory on node 0 (108.22/125.65 GiB) on startup is less than desired CPU memory utilization (0.92, 115.59 GiB). On the CPU backend, the `--gpu-memory-utilization` flag controls the fraction of CPU memory reserved (despite its name). To resolve: decrease `--gpu-memory-utilization` (e.g. `--gpu-memory-utilization 0.5`) or reduce CPU memory used by other processes.
      if [ "$MACHINE_TYPE" = "cpu" ]; then
          echo here in machine type cpu
          ENGINE_EXTRA_ARGS+=(--gpu-memory-utilization 0.5)
      fi

      echo "[$(hostname)] Node rank: $NODE_RANK / '"$NUM_NODES"'"
      echo "[$(hostname)] NET_IFACE=$NET_IFACE VLLM_HOST_IP=$VLLM_HOST_IP"
      
      if [ "$NODE_RANK" -eq 0 ]; then
        echo "[$(hostname)][RANK $NODE_RANK] Launching vLLM with DP_SIZE=1 TP_SIZE='"$TP_SIZE"' PP_SIZE='"$PP_SIZE"'"
        echo "[$(hostname)][RANK $NODE_RANK] max_model_len='$MAX_MODEL_LEN'"
        echo "[$(hostname)][RANK $NODE_RANK] ENGINE_EXTRA_ARGS=${ENGINE_EXTRA_ARGS[*]}"
        echo "[$(hostname)][RANK $NODE_RANK] NCCL settings: NET=$NCCL_NET NVLS_ENABLE=$NCCL_NVLS_ENABLE IB_TIMEOUT=$NCCL_IB_TIMEOUT IB_RETRY_CNT=$NCCL_IB_RETRY_CNT SOCKET_IFNAME=$NCCL_SOCKET_IFNAME IB_HCA=$NCCL_IB_HCA IB_DISABLE=$NCCL_IB_DISABLE DEBUG=$NCCL_DEBUG DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS P2P_DISABLE=$NCCL_P2P_DISABLE"
      fi

      if [ "$NUM_NODES" -gt 1 ]; then
          main_args=('"$MODEL_PATH"' \
            --tensor-parallel-size '"$TP_SIZE"' \
            --pipeline-parallel-size '"$PP_SIZE"' \
            --max-model-len '"$MAX_MODEL_LEN"' \
            --nnodes '"$NUM_NODES"' \
            --master-addr '"$HEAD_NODE"' \
            --master-port '"$MASTER_PORT"' \
          )
      else
          main_args=('"$MODEL_PATH"' \
            --tensor-parallel-size '"$TP_SIZE"' \
            --pipeline-parallel-size '"$PP_SIZE"' \
            --max-model-len '"$MAX_MODEL_LEN"' \
            --disable-custom-all-reduce \
          )
      fi


      if [ "$NODE_RANK" -eq 0 ]; then
          # Head node: runs the API server
          vllm serve "${main_args[@]}" \
            "${ENGINE_EXTRA_ARGS[@]}" \
            --node-rank 0 \
            --host 0.0.0.0 \
            --port '$PORT'
      else
          # Worker nodes: headless (no API server)
          vllm serve "${main_args[@]}" \
            "${ENGINE_EXTRA_ARGS[@]}" \
            --node-rank $NODE_RANK \
            --headless
      fi
  ' &

VLLM_PID=$!
# With setsid, the server is its own process group leader; capture the PGID
# so shutdown can signal the whole tree (singularity -> bash -> vllm -> workers).
VLLM_PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
# Safety: never group-kill our own process group.
MY_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
[ "$VLLM_PGID" = "$MY_PGID" ] && VLLM_PGID=""

# Gracefully stop the vllm serve process tree (SIGTERM, wait, SIGKILL).
stop_vllm_tree() {
    local pid=$1 pgid=$2
    if kill -0 "$pid" 2>/dev/null; then
        if [ -n "$pgid" ]; then
            kill -TERM -"$pgid" 2>/dev/null
        fi
        kill -TERM "$pid" 2>/dev/null
        # Wait up to 60s for a clean exit before forcing.
        for _ in $(seq 1 60); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARN] vLLM did not exit after SIGTERM, sending SIGKILL..."
            [ -n "$pgid" ] && kill -KILL -"$pgid" 2>/dev/null
            kill -KILL "$pid" 2>/dev/null
        fi
    fi
    wait "$pid" 2>/dev/null
}

# Sentinel files on the shared filesystem, used to coordinate an ordered,
# worker-first shutdown (LAUNCH_FOLDER is visible from all step nodes).
SHUTDOWN_SENTINEL="$LAUNCH_FOLDER/.vllm_shutdown"
WORKER_DONE_SENTINEL="$LAUNCH_FOLDER/.vllm_worker_done"
rm -f "$SHUTDOWN_SENTINEL" "$WORKER_DONE_SENTINEL"

if [ "$SLURM_NODEID" -ne 0 ]; then
    # Worker node: no API server runs here, so the readiness check below can
    # never succeed. Headless vllm workers do NOT exit on their own when the
    # head node shuts down (known vLLM multi-node gap): their NCCL
    # HeartbeatMonitor keeps polling the dead TCPStore and spams c10d
    # "Broken pipe" errors until Slurm kills the step.
    #
    # Shutdown order matters: the worker must exit WHILE the head's TCPStore
    # is still alive, otherwise the worker ranks poll a dead store and emit
    # c10d Broken pipe warnings. So the worker waits for the head's shutdown
    # sentinel (or, as a crash fallback, for the head's store port to go
    # down), then stops itself and signals completion.
    echo "[Worker node] Waiting for shutdown signal from head node $HEAD_NODE ..."
    SEEN_UP=0
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        if [ -f "$SHUTDOWN_SENTINEL" ]; then
            echo "[Worker node] Shutdown sentinel detected; stopping worker..."
            break
        fi
        # Fallback: if the head node crashed, its TCPStore port goes down.
        if timeout 3 bash -c "exec 3<>/dev/tcp/$HEAD_NODE/$MASTER_PORT" 2>/dev/null; then
            SEEN_UP=1
        elif [ "$SEEN_UP" -eq 1 ]; then
            echo "[Worker node] Head node store at $HEAD_NODE:$MASTER_PORT is down; stopping worker..."
            break
        fi
        sleep 2
    done
    stop_vllm_tree "$VLLM_PID" "$VLLM_PGID"
    touch "$WORKER_DONE_SENTINEL"
    echo "[Worker node] vLLM worker exited."
    echo "Benchmark finished"
    exit 0
fi

echo "Waiting for vLLM server to be ready..."


##################################################
### WAIT until vLLM Server is UP 
##################################################
echo "Waiting for vLLM server to be ready..."
until curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; do
  # Check if vLLM process has exited
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "❌ vLLM serve failed to start or crashed!"
    exit 1
  fi
  sleep 5
done

sleep 10

##################################################
### Activate Environment
##################################################
# module purge
source $SCRIPTS_DIR/activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM
echo host python:
which python
which python3
echo vllm container python:
singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $VLLM_IMAGE which python
singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $VLLM_IMAGE which python3
echo metrics container python:
singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $METRICS_IMAGE which python
singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $METRICS_IMAGE which python3

echo "Starting vLLM serve inside Singularity (head node only)"

concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do

    echo "=============================="
    echo "Running concurrency $conc"
    echo "=============================="

    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"

    ##################################################
    # GPU MONITOR (inside container)
    ##################################################
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $VLLM_IMAGE \
            python3 $SCRIPTS_DIR/vllm/gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 &
    GPU_MON_PID=$!

    ##################################################
    # BENCHMARK (inside container)
    ##################################################
    # #python3 $BENCHMARK_FILE \
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $METRICS_IMAGE \
            python3 $BENCHMARK_FILE \
            --backend vllm \
            --host localhost \
            --port $PORT \
            --model $MODEL_PATH \
            --dataset-name $DATASET \
            --dataset-path $DATASET_PATH \
            --max-concurrency $conc \
            --num-prompts 1000 \
            --save-result \
            --result-filename "$LAUNCH_FOLDER/Concurrency_${conc}.json" \
            > "$LAUNCH_FOLDER/logs_benchmark_${conc}.log"

    ##################################################
    # STOP MONITOR
    ##################################################
    kill $GPU_MON_PID
    sleep 5

done

##################################################
### Graceful vLLM shutdown (worker first, then head)
##################################################
# Shut the server down cleanly instead of letting Slurm SIGKILL the step's
# processes. The worker node shuts down FIRST (while this node's TCPStore is
# still alive, so worker ranks never poll a dead store), then the head stops.
echo "Signaling worker node to shut down..."
touch "$SHUTDOWN_SENTINEL"
# Wait up to 120s for the worker to confirm shutdown before stopping ourselves.
WORKER_OK=0
for _ in $(seq 1 120); do
    if [ -f "$WORKER_DONE_SENTINEL" ]; then
        WORKER_OK=1
        break
    fi
    sleep 1
done
if [ "$WORKER_OK" -eq 1 ]; then
    echo "Worker node confirmed shutdown."
else
    echo "[WARN] Worker did not confirm shutdown within 120s; proceeding."
fi

echo "Stopping vLLM server (PID $VLLM_PID, PGID ${VLLM_PGID:-unknown})..."
stop_vllm_tree "$VLLM_PID" "$VLLM_PGID"
echo "vLLM server stopped."
rm -f "$SHUTDOWN_SENTINEL" "$WORKER_DONE_SENTINEL"

# Remove Temporary dir.
echo "Remove TMPDIR"
rm -rf $TMPDIR
echo "$TMPDIR Removed!"
sleep 10


echo "Benchmark finished"

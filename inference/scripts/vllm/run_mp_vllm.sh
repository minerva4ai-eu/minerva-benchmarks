#!/bin/bash
#SBATCH --job-name=VLLM_DYNAMIC
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00

module load $MODULES

# Tmp dir
# Create Temporal directories
TMPDIR=$CUR_DIR/tmp
mkdir $TMPDIR
chmod -R 777 $TMPDIR
export SINGULARITY_CACHEDIR=$TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR

##################################################
### Environment Variables Setup
##################################################

LAUNCH_FOLDER=$1
BENCHMARK_FILE=$2
DATASET=$3
DATASET_PATH=$4
MACHINE=$5
MACHINE_TYPE=$6

echo "LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "BENCHMARK_FILE: $BENCHMARK_FILE"
echo "DATASET: $DATASET"
echo "DATASET_PATH: $DATASET_PATH"
echo "MACHINE: $MACHINE"
echo "MACHINE_TYPE: $MACHINE_TYPE"
echo "MODEL_PATH: $MODEL_PATH"
echo "VLLM_IMAGE: $VLLM_IMAGE"

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# NCCL
source activate-env-variables-per-supercomputer.sh

# vLLM variables
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn


##################################################
### Node discovery
##################################################

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "Head node: $head_node ($head_node_ip)"

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
export ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL ENFORCE_EAGER DISABLE_CUSTOM_ALL_REDUCE

# Disable allreduce+rms fusion path that can require multicast-capable symmetric memory
#export COMPILATION_CONFIG='{"pass_config":{"fuse_allreduce_rms":true}}'

export MASTER_PORT=29500

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
NODE_COUNT=${#NODES[@]}
# Apply node count limit if specified
if [ -n "$NODE_COUNT" ] && [ "$NODE_COUNT" -gt 0 ]; then
    NODES=( "${NODES[@]:0:$NODE_COUNT}" )
fi

export NUM_NODES=${#NODES[@]}
export NODELIST=$(IFS=,; echo "${NODES[*]}")
export MASTER_ADDR="${NODES[0]}"


##################################################
### RUN vLLM SERVE 
##################################################
echo "BINDINGS_SINGULARITY: $BINDINGS_SINGULARITY"

srun --nodes="$NUM_NODES" --ntasks-per-node=1 --nodelist="$NODELIST" --export=ALL \
 singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS \
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
            --master-addr '"$MASTER_ADDR"' \
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
module purge
source activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM

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
    #singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $VLLM_IMAGE \
    python gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 &
    GPU_MON_PID=$!

    ##################################################
    # BENCHMARK (inside container)
    ##################################################
    # #python3 $BENCHMARK_FILE \
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $VLLM_IMAGE \
    	python $BENCHMARK_FILE \
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

# Remove Temporary dir.
echo "Remove TMPDIR"
rm -rf $TMPDIR
echo "$TMPDIR Removed!"
sleep 10


echo "Benchmark finished"

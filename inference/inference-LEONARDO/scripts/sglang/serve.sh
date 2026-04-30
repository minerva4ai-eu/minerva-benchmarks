#!/bin/bash

# Singularity variables
#module load $SINGULARITY_MODULE


source activate-env-per-supercomputer.sh $ENVIRONMENT_SGLANG
source activate-env-variables-per-supercomputer.sh $ENVIRONMENT_SGLANG


echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"

# Create Temporal directories
TMPDIR=/leonardo_work/MNRVA_bench/minerva-benchmarks/inference/inference-LEONARDO/tmp
mkdir $TMPDIR
chmod -R 777 $TMPDIR
export FLASHINFER_WORKSPACE_BASE=/leonardo_work/MNRVA_bench/minerva-benchmarks/inference/inference-LEONARDO

#export SINGULARITY_CACHEDIR=$TMPDIR
#export SINGULARITY_TMPDIR=$TMPDIR

# If the model is 'gemma-3-12b' -> pp and tp aren't implemented yet!
if [[ "$MODEL_PATH" == *"gemma-3-12b-it"* ]]; then
    # Run without --pp-size and --tp
    #$CUR_DIR/wrapper_singularity.sh
    # $SINGULARITY_EXEC_COMMAND \
    #singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
    sglang serve \
        --attention-backend triton \
        --model-path "$MODEL_PATH" \
        --context-length "$MAX_MODEL_LENGTH" \
        --port "$PORT" \
        --grammar-backend "xgrammar" \
        --dist-init-addr "$NCCL_INIT_ADDR" \
        --model-impl transformers \
        $ADDITIONAL_ARGS \
        --nnodes $NODES \
        --node-rank "$SLURM_NODEID" &
elif [[ "$MODEL_PATH" == *"Llama-3.1-405B"* || "$MODEL_PATH" == *"Llama-3.1-405B-Instruct"* ]]; then
    # Run with --pp-size and --tp
    #$CUR_DIR/wrapper_singularity.sh
    # $SINGULARITY_EXEC_COMMAND \
    #singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
    sglang serve \
        --attention-backend triton \
        --model-path "$MODEL_PATH" \
        --pp-size "$PIPELINE_PARALLEL" \
        --tp-size "$TENSOR_PARALLEL" \
        --context-length "$MAX_MODEL_LENGTH" \
        --port "$PORT" \
        --grammar-backend "xgrammar" \
        --dist-init-addr "$NCCL_INIT_ADDR" \
        --mem-fraction-static 0.80 --chunked-prefill-size 4096 \
        $ADDITIONAL_ARGS \
        --nnodes $NODES \
        --node-rank "$SLURM_NODEID" &
        
elif [[ "$MODEL_PATH" == *"Mistral-7B-Instruct-v0.3"* ]]; then
    # Run with --pp-size and --tp
    #$CUR_DIR/wrapper_singularity.sh
    #singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
      sglang serve \
        --model-path "$MODEL_PATH" \
        --attention-backend triton \
        --pp-size 1 \
        --tp-size 4 \
        --context-length "$MAX_MODEL_LENGTH" \
        --port "$PORT" \
        --nnodes $NODES \
        --dist-init-addr "$NCCL_INIT_ADDR" \
        --node-rank "$SLURM_NODEID" &
        #--mem-fraction-static 0.50 \
        #--cuda-graph-max-bs 4 \
        # --disable-cuda-graph \
        #--grammar-backend "xgrammar" \
        #--attention-backend torch_native \


# else
#     # Run with --pp-size and --tp-size
#     #$CUR_DIR/wrapper_singularity.sh
#     # $SINGULARITY_EXEC_COMMAND \
#     #singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
#       sglang serve \
#         --attention-backend triton \
#         --model-path "$MODEL_PATH" \
#         --pp-size "$PIPELINE_PARALLEL" \
#         --tp-size "$TENSOR_PARALLEL" \
#         --context-length "$MAX_MODEL_LENGTH" \
#         --port "$PORT" \
#         --grammar-backend "xgrammar" \
#         --dist-init-addr "$NCCL_INIT_ADDR" \
#         $ADDITIONAL_ARGS \
#         --nnodes $NODES \
#         #--node-rank "$SLURM_NODEID" &
fi

SGLANG_PID=$!
echo "Waiting for SGLANG server to be ready..."


echo "Waiting for SGLANG server to be ready..."
until curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; do
  # Check if SGLANG process has exited
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "❌ SGLANG serve failed to start or crashed!"
    exit 1
  fi
  sleep 5
done

sleep 10


# Activate environment
module purge
source activate-env-per-supercomputer.sh $ENVIRONMENT_SGLANG
echo "ci siamoooooooooooooooooooo"
# # Get current hostnames and Head Node
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  
concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc"
    
    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"
    
    # Run in GPU monitor in background.
    python gpu_summary_monitor-cuda.py "$SUMMARY_FILE" 0.10 & #> "$LOG_FILE" 2>&1 &
    GPU_MON_PID=$!

    # Run benchmark stressing the sglang server.
    python $BENCHMARK_FILE --backend 'sglang' \
        --host localhost \
        --port $PORT \
        --model $MODEL_PATH \
        --dataset-name $DATASET \
        --dataset-path $DATASET_PATH \
        --max-concurrency $conc \
        --num-prompts 1000 \
        --save-result \
        --result-filename "$LAUNCH_FOLDER/Concurrency_$conc.json" \
        > "$LAUNCH_FOLDER/logs_benchmarking_$conc-concurrency.log"

    # Stop monitoring
    kill "$GPU_MON_PID"

    # Wait for nvidia-smi to exit cleanly
    sleep 2
done

sleep 10

# Remove Temporal directories.
echo "Remove TMPDIR"
rm -rf $TMPDIR
echo "$TMPDIR Removed!"
sleep 10


exit 0

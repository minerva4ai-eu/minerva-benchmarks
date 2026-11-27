#!/bin/bash

source $SCRIPTS_DIR/activate-env-variables-per-supercomputer.sh

echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"
echo "BENCHMARK_FILE: $BENCHMARK_FILE"
echo "DATASET: $DATASET"
echo "DATASET_PATH: $DATASET_PATH"
echo "MODEL_PATH: $MODEL_PATH"
echo "MAX_MODEL_LENGTH: $MAX_MODEL_LENGTH"

NODES=$SLURM_STEP_NUM_NODES

echo "NODES: $NODES"

# If the model is 'gemma-3-12b' -> pp and tp aren't implemented yet! #TODO: check newer version
if [[ "$MODEL_PATH" == *"gemma-3-12b-it"* ]]; then
    # Run without --pp-size and --tp
    $SCRIPTS_DIR/sglang/wrapper_singularity.sh
    # $SINGULARITY_EXEC_COMMAND \
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
      python3 -m sglang.launch_server \
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
    $SCRIPTS_DIR/sglang/wrapper_singularity.sh
    # $SINGULARITY_EXEC_COMMAND \
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
      python3 -m sglang.launch_server \
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
    $SCRIPTS_DIR/sglang/wrapper_singularity.sh
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
      python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp-size "$TENSOR_PARALLEL" \
        --pp-size "$PIPELINE_PARALLEL" \
        --context-length "$MAX_MODEL_LENGTH" \
        --port "$PORT" \
        --grammar-backend "xgrammar" \
        --dist-init-addr "$NCCL_INIT_ADDR" \
        --mem-fraction-static 0.80 \
        --nnodes $NODES \
        --node-rank "$SLURM_NODEID" &

else
    # Run with --pp-size and --tp-size
    $SCRIPTS_DIR/sglang/wrapper_singularity.sh
    # $SINGULARITY_EXEC_COMMAND \
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $SGLANG_IMAGE \
      python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --pp-size "$PIPELINE_PARALLEL" \
        --tp-size "$TENSOR_PARALLEL" \
        --context-length "$MAX_MODEL_LENGTH" \
        --port "$PORT" \
        --grammar-backend "xgrammar" \
        --dist-init-addr "$NCCL_INIT_ADDR" \
        $ADDITIONAL_ARGS \
        --nnodes $NODES \
        --node-rank "$SLURM_NODEID" &
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


# # Activate environment
source $SCRIPTS_DIR/activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM

concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc"
    
    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"
    
    # Run in GPU monitor in background.
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $METRICS_IMAGE \
            python3 $SCRIPTS_DIR/sglang/gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 &
    GPU_MON_PID=$!

    # Run benchmark stressing the sglang server.
    singularity exec -B $BINDINGS_SINGULARITY $ADDITIONAL_SINGULARITY_ARGS $METRICS_IMAGE \
            python3 $BENCHMARK_FILE \
            --backend sglang \
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

    # Stop monitoring
    kill "$GPU_MON_PID"

    # Wait for nvidia-smi to exit cleanly
    sleep 2
done

exit 0

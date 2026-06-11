#!/bin/bash

source activate-env-variables-per-supercomputer.sh

echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"

# Activate sglang environment
source activate-env-per-supercomputer.sh $ENVIRONMENT_SGLANG

export SGLANG_USE_AITER=0

# Build sglang launch args (TP seul : pipeline parallelism non supporté sur ROCm ici)
SGLANG_ARGS=(
    --model-path        "$MODEL_PATH"
    --context-length    "$MAX_MODEL_LENGTH"
    --host              "0.0.0.0"
    --port              "$PORT"
    --attention-backend "triton"
    --dist-init-addr    "$NCCL_INIT_ADDR"
    --nnodes            "$NODES"
    --node-rank         "$SLURM_NODEID"
    --disable-cuda-graph
    --watchdog-timeout   7200
    --log-level         info
    --tensor-parallel-size "$TENSOR_PARALLEL"
)

if [[ "$MODEL_PATH" == *"Llama-3.1-405B"* || "$MODEL_PATH" == *"Llama-3.1-405B-Instruct"* ]]; then
    SGLANG_ARGS+=(
        --mem-fraction-static  0.80
        --chunked-prefill-size 4096
    )
fi

if [[ -n "$ADDITIONAL_ARGS" ]]; then
    SGLANG_ARGS+=($ADDITIONAL_ARGS)
fi

python3 -m sglang.launch_server "${SGLANG_ARGS[@]}" &
SGLANG_PID=$!
echo "Waiting for SGLANG server to be ready..."


echo "Waiting for SGLANG server to be ready..."
until curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "❌ SGLANG serve failed to start or crashed!"
        exit 1
    fi
    sleep 5
done

sleep 10

concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc"
    
    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"
    
    # Run in GPU monitor in background.
    python gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 & #> "$LOG_FILE" 2>&1 &
    GPU_MON_PID=$!

    # Run benchmark stressing the sglang server.
    python $BENCHMARK_FILE --backend 'sglang' \
        --host              'localhost' \
        --port              $PORT \
        --model             $MODEL_PATH \
        --dataset-name      $DATASET \
        --dataset-path      $DATASET_PATH \
        --max-concurrency   $conc \
        --num-prompts       1000 \
        --save-result \
        --result-filename   "$LAUNCH_FOLDER/Concurrency_$conc.json" \
        > "$LAUNCH_FOLDER/logs_benchmarking_$conc-concurrency.log"

    # Stop monitoring
    kill "$GPU_MON_PID"
    sleep 2
done

sleep 10

kill "$SGLANG_PID"

exit 0
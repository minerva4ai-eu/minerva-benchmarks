#!/bin/bash


source activate-env-variables-per-supercomputer.sh

python3 -c "import torch; torch.cuda.empty_cache()"

echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"
echo "serve.sh: MACHINE: ${MACHINE}"
echo "serve.sh: MACHINE_TYPE: ${MACHINE_TYPE}"
echo "serve.sh: Container name: ${SINGULARITY_NAME}"

# vLLM serve
vllm serve "$MODEL_PATH" \
    --port $PORT \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --max-model-len $MAX_MODEL_LENGTH \
    --distributed-executor-backend "ray" \
    $ADDITIONAL_ARGS &

VLLM_PID=$!
echo "Waiting for vLLM server to be ready..."

until curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ vLLM serve failed to start or crashed!"
        exit 1
    fi
    sleep 5
done

sleep 10

concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc"

    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"

    # Run GPU monitor in background
    python gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 &
    GPU_MON_PID=$!

    # Run benchmark
    python $BENCHMARK_FILE --backend 'vllm' \
        --host 'localhost' \
        --port $PORT \
        --model $MODEL_PATH \
        --dataset-name $DATASET \
        --dataset-path $DATASET_PATH \
        --max-concurrency $conc \
        --num-prompts 1000 \
        --save-result \
        --result-filename "$LAUNCH_FOLDER/Concurrency_$conc.json" \
        > "$LAUNCH_FOLDER/logs_benchmarking_$conc-concurrency.log"

    # Stop GPU monitor
    kill "$GPU_MON_PID"
    sleep 2
done

sleep 10

exit 0

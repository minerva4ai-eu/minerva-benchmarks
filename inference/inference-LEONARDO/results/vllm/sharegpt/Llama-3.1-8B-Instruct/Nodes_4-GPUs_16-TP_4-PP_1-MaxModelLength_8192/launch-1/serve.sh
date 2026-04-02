#!/bin/bash

# Activate environment
echo "ENVIRONMENT_VLLM: $ENVIRONMENT_VLLM"
echo ""
source activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM
# ml miniforge
# source activate $ENVIRONMENT_VLLM
# export PATH=$ENVIRONMENT_VLLM/bin:$PATH
# which python

echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"
echo "serve.sh: MACHINE: ${MACHINE}"
echo "serve.sh: MACHINE_TYPE: ${MACHINE_TYPE}"
echo "server.sh: PYTHON PATH:"
which python
echo ""

# vLLM serve
vllm serve "$MODEL_PATH" \
    --port $PORT \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --max-model-len $MAX_MODEL_LENGTH \
    $ADDITIONAL_ARGS &

#    --swap-space 2 --cpu-offload-gb 0.5 --enable-chunked-prefill --enforce-eager \

#    --cpu-offload-gb 0.5 \


VLLM_PID=$!
echo "Waiting for vLLM server to be ready..."


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

concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc"

    # Launch GPU monitor in background
    # METRICS_FILE="$LAUNCH_FOLDER/gpu_metrics_${conc}.csv"
    # nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw \
    #            --format=csv,noheader,nounits -l 1 > "$METRICS_FILE" &
    #         # -l 1 -> loop every 1 second
    
    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"
    
    # Run in GPU monitor in background.
    python gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 & #> "$LOG_FILE" 2>&1 &
    GPU_MON_PID=$!

    # Run benchmark stressing the vllm server.
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

    # Stop monitoring
    kill "$GPU_MON_PID"

    # Wait for nvidia-smi to exit cleanly
    sleep 2
done

sleep 10

exit 0

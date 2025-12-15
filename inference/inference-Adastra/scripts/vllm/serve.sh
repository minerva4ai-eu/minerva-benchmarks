#!/bin/bash

module load $MODULES
source $ENVIRONMENT_VLLM/bin/activate
source config.sh
export PATH=$ENVIRONMENT_VLLM/bin:$PATH
which python

echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"

python3 -c "import torch; torch.cuda.empty_cache()"

MEM=0.60  # Fraction of GPU memory to use

# vLLM serve
vllm serve "$MODEL_PATH" \
    --port $PORT \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --max-model-len $MAX_MODEL_LENGTH \
    --gpu-memory-utilization $MEM \
    $ADDITIONAL_ARGS \
    --swap-space 2 \
    --enable-chunked-prefill \
    --enforce-eager \
    --distributed-executor-backend=ray &

#    --cpu-offload-gb 0.5 \


VLLM_PID=$!
echo "Waiting for vLLM server to be ready..."

until curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; do
  # Check if vLLM process has exited
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "❌ vLLM serve failed to start or crashed!"
    exit 1
  fi
  sleep 5
done


echo "Starting sending requests to the model"

# Requests (controlling the time of each request)
time curl -X POST localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" --data "{\"model\": \"$MODEL_PATH\",\"messages\": [{\"role\": \"user\", \"content\": \"say I have many shapes and I have a 3D cavity. I want to study the fit between possible shapes and the cavity. What is this problem? Has it been studied?\"}]}"

sleep 60
echo "Parallel requests"

# Parallel requests
curl -X POST localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" --data "{\"model\": \"$MODEL_PATH\",\"messages\": [{\"role\": \"user\", \"content\": \"say I have many shapes and I have a 3D cavity. I want to study the fit between possible shapes and the cavity. What is this problem? Has it been studied?\"}]}" &
curl -X POST localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" --data "{\"model\": \"$MODEL_PATH\",\"messages\": [{\"role\": \"user\", \"content\": \"say I have many shapes and I have a 3D cavity. I want to study the fit between possible shapes and the cavity. What is this problem? Has it been studied?\"}]}" &
curl -X POST localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" --data "{\"model\": \"$MODEL_PATH\",\"messages\": [{\"role\": \"user\", \"content\": \"say I have many shapes and I have a 3D cavity. I want to study the fit between possible shapes and the cavity. What is this problem? Has it been studied?\"}]}" &

sleep 30

echo "Requests sent"

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
    # LOG_FILE="$LAUNCH_FOLDER/gpu_monitor_${conc}.log"
    
    # python gpu_summary_monitor.py "$SUMMARY_FILE" 0.5 &
    python gpu_summary_monitor.py "$SUMMARY_FILE" 0.10 $MEM & #> "$LOG_FILE" 2>&1 &

    GPU_MON_PID=$!

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

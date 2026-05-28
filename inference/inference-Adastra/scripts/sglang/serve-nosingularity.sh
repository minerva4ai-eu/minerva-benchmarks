#!/bin/bash

# Activate environment variables (NCCL, ROCm, etc.)
source activate-env-variables-per-supercomputer.sh

echo "serve.sh: LAUNCH_FOLDER: $LAUNCH_FOLDER"
echo "serve.sh: ADDITIONAL_ARGS: ${ADDITIONAL_ARGS[*]}"

# Activate sglang environment
source activate-env-per-supercomputer.sh $ENVIRONMENT_SGLANG

export SGLANG_USE_AITER=0

##################################################
###         Build sglang launch args           ###
##################################################

# Base args common to all models
SGLANG_ARGS=(
    --model-path        "$MODEL_PATH"
    --context-length    "$MAX_MODEL_LENGTH"
    --port              "$PORT"
    --grammar-backend   "xgrammar"
    --dist-init-addr    "$NCCL_INIT_ADDR"
    --nnodes            "$NODES"
    --node-rank         "$SLURM_NODEID"
    --watchdog-timeout   1800
    --mem-fraction-static   0.50
    --disable-cuda-graph
)

# Model-specific args
if [[ "$MODEL_PATH" == *"gemma-3-12b-it"* ]]; then
    # No pp/tp for gemma -- not implemented in SGLang yet
    SGLANG_ARGS+=(--model-impl transformers)

elif [[ "$MODEL_PATH" == *"Llama-3.1-405B"* || "$MODEL_PATH" == *"Llama-3.1-405B-Instruct"* ]]; then
    SGLANG_ARGS+=(
        --pp-size               "$PIPELINE_PARALLEL"
        --tp-size               "$TENSOR_PARALLEL"
        --chunked-prefill-size  4096
    )

elif [[ "$MODEL_PATH" == *"Mistral-7B-Instruct-v0.3"* ]]; then
    # pp and tp are intentionally swapped for Mistral
    SGLANG_ARGS+=(
        --pp-size               "$TENSOR_PARALLEL"
        --tp-size               "$PIPELINE_PARALLEL"
    )

else
    SGLANG_ARGS+=(
        --pp-size   "$PIPELINE_PARALLEL"
        --tp-size   "$TENSOR_PARALLEL"
    )
fi

# Append any extra args passed from the caller
[[ -n "$ADDITIONAL_ARGS" ]] && SGLANG_ARGS+=($ADDITIONAL_ARGS)

##################################################
###            Launch sglang server            ###
##################################################

sglang serve "${SGLANG_ARGS[@]}" &
SGLANG_PID=$!

echo "Waiting for SGLANG server to be ready..."
until curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "❌ SGLANG serve failed to start or crashed!"
        exit 1
    fi
    sleep 5
done

sleep 10

##################################################
###              Run benchmarks                ###
##################################################

concurrencies=(150 250 300 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc"

    SUMMARY_FILE="$LAUNCH_FOLDER/gpu_summary_${conc}.txt"

    python gpu_summary_monitor-$MACHINE_TYPE.py "$SUMMARY_FILE" 0.10 &
    GPU_MON_PID=$!

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

    kill "$GPU_MON_PID"
    sleep 2
done

sleep 10

exit 0
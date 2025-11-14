
echo "Using TENSOR_PARALLEL: $TENSOR_PARALLEL"
# PORT=2951 # PORT was previously defined in .env as 2950 but was not exported. It is also defined in the serve_deepspeed_mii.py to 2951
# Deploy the DeepSpeed-MII server.
python $FRAMEWORK_PATH/source/serve_deepspeed_mii.py $TENSOR_PARALLEL & # Is last & is needed?

sleep 180


concurrencies=(50 100 200 300 400 500 1000) # SHOULD THIS BE FIX OR IN THE MAIN SCRIPT?

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc for MODEL: $MODEL"

    # Launch GPU monitor in background
    METRICS_FILE="$OUTPUT_PATH/gpu_metrics_${conc}.csv"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw \
               --format=csv,noheader,nounits -l 1 > "$METRICS_FILE" &
            # -l 1 -> loop every 1 second
    GPU_MON_PID=$! #CHECK THE PREVIOUS COMMENT, AND IN GENERAL THE nvidia-smi COMMAND

    python $SOURCE_PATH/benchmark_serving.py --backend 'deepspeed-mii' \
        --host 'localhost' \
        --port $PORT \
        --model $MODEL_PATH \
        --dataset-name $DATASET \
        --dataset-path $DATASET_PATH \
        --max-concurrency $conc \
        --num-prompts 1000 \
        --save-result \
        --result-filename $OUTPUT_PATH/"Concurrency_$conc.json" \
        --endpoint "/mii/$MODEL" \ # WHERE IS THE mii DIRECTORY?
        > $OUTPUT_PATH/"logs_benchmarking_$conc-concurrency.log"

    # Stop monitoring
    kill $GPU_MON_PID

    # Wait for nvidia-smi to exit cleanly
    sleep 2

done


sleep 20


exit # exit 0?


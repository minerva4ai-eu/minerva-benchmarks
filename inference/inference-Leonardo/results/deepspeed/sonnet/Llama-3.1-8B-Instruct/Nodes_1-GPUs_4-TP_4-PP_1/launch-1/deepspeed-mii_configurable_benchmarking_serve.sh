#!/bin/bash

#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=MII-DeepSpeed-DYNAMIC
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
##SBATCH --time=02:00:00


# Get Arguments
LAUNCH_FOLDER=$1
BENCHMARK_FILE=$2
DATASET=$3
DATASET_PATH=$4

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}"
echo "BENCHMARK_FILE: {$BENCHMARK_FILE}"
echo "DATASET: {$DATASET}"
echo "DATASET_PATH: {$DATASET_PATH}"
echo "MODEL: {$MODEL}"
echo "MODEL_PATH: {$MODEL_PATH}"

# Activate environment
module load $MODULES
source $ENVIRONMENT_DEEPSPEED/bin/activate
export PATH=$ENVIRONMENT_DEEPSPEED/bin:$PATH
which python

echo "Using TENSOR_PARALLEL: $TENSOR_PARALLEL"
PORT=2951
# Deploy the DeepSpeed-MII server.
python $LAUNCH_FOLDER/serve_deepspeed_mii.py $TENSOR_PARALLEL &

sleep 180


concurrencies=(50 100 200 300 400 500 1000)

for conc in "${concurrencies[@]}"; do
    echo "Running concurrency level $conc for MODEL: $MODEL"

    # Launch GPU monitor in background
    METRICS_FILE="$LAUNCH_FOLDER/gpu_metrics_${conc}.csv"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw \
               --format=csv,noheader,nounits -l 1 > "$METRICS_FILE" &
            # -l 1 -> loop every 1 second
    GPU_MON_PID=$!

    python $BENCHMARK_FILE --backend 'deepspeed-mii' \
        --host 'localhost' \
        --port $PORT \
        --model $MODEL_PATH \
        --dataset-name $DATASET \
        --dataset-path $DATASET_PATH \
        --max-concurrency $conc \
        --num-prompts 1000 \
        --save-result \
        --result-filename $LAUNCH_FOLDER/"Concurrency_$conc.json" \
        --endpoint "/mii/$MODEL" \
        > $LAUNCH_FOLDER/"logs_benchmarking_$conc-concurrency.log"

    # Stop monitoring
    kill $GPU_MON_PID

    # Wait for nvidia-smi to exit cleanly
    sleep 2

done


sleep 20


exit


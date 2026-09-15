#!/bin/bash



##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi

echo yaml_path=$1

source scripts/shared/runtime_environment.sh
training_activate_runtime_environment

echo "EXECUTION_MODE: $EXECUTION_MODE"
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix: $runtime_prefix"

###################################################

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.


gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"

train_command="${runtime_prefix:+$runtime_prefix} python $TRAIN_MODULE --yaml $1"

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $DISABLE_COMPILE == "True" || $DISABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

$prepare_train_command

# Launch Run

# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5

# Run training in foreground (this blocks until done)
$train_command

kill -SIGTERM "$monitor_pid"

# Wait for the monitor to clean up and exit
wait "$monitor_pid"


echo "✅ Single Node single GPU job completed."


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


prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

$prepare_train_command

# Launch Run

# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5

# Each repeat is a fresh process; a failed repeat does not stop the next ones
exit_code=0
for (( repeatid=0; repeatid<${REPEATS:-1}; repeatid++ )); do
    echo "Running repeatid: $repeatid"
    REPEAT_ID=$repeatid $train_command || {
        rc=$?
        echo "repeat $repeatid failed with exit code $rc"
        exit_code=$rc
    }
done

# Terminate the monitor
kill -SIGTERM "$monitor_pid"
# Wait for the monitor to clean up and exit
wait "$monitor_pid"



echo "✅ Single Node single GPU job completed."
exit "$exit_code"

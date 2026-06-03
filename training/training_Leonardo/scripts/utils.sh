#!/bin/bash

# --- Utility to get dataset path from config file ---
# Get dataset as a String
# get_dataset_path() {
#   local dataset="$1"
#   local json_file="$2"
#   grep -oP "\"$dataset\"\\s*:\\s*\"[^\"]+\"" "$json_file" | \
#     sed -E "s/.*: \"([^\"]+)\"/\1/"
# }
# Get dataset as JSON (String or Object)
get_dataset_path() {
  local dataset="$1"
  local json_file="$2"

  if [ ! -f "$json_file" ]; then
    echo "❌ Config file not found: $json_file" >&2
    exit 1
  fi

  # Extract the dataset entry (can be a string or an object)
  local entry
  entry=$(jq -c --arg ds "$dataset" '.[$ds]' "$json_file")

  if [ "$entry" == "null" ] || [ -z "$entry" ]; then
    echo "❌ Dataset '$dataset' not found in $json_file" >&2
    exit 1
  fi

  # Output the entry as-is (compact JSON)
  echo "$entry"
}

# Function to get dataset path and handler from YAML
get_dataset_info() {
  local dataset="$1"
  local yaml_file="$2"

  # Use yq to extract values
  DATASET_PATH=$(yq -r ".datasets.${dataset}.dataset_path" "$yaml_file")
  DATASET_HANDLER=$(yq -r ".datasets.${dataset}.dataset_handler" "$yaml_file")

  # Check if the dataset exists
  if [[ "$DATASET_PATH" == "null" || "$DATASET_HANDLER" == "null" ]]; then
    echo "Error: Dataset '$dataset' not found in $yaml_file" >&2
    return 1
  fi

  export DATASET_PATH
  export DATASET_HANDLER
}

# --- Utility to get model type from model_type_map.json ---
get_model_type() {
  local model="$1"
  local json_file="$2"
  grep -oP "\"$model\"\\s*:\\s*\"[^\"]+\"" "$json_file" | \
    sed -E "s/.*: \"([^\"]+)\"/\1/"
}

# --- Utility to get model directory from model_type_directories.json ---
get_model_directory() {
  local model_type="$1"
  local json_file="$2"
  grep -oP "\"$model_type\"\\s*:\\s*\"[^\"]+\"" "$json_file" | \
    sed -E "s/.*: \"([^\"]+)\"/\1/"
}

# --- Utility to get model parallelism for training from model_parallelism_config.json ---
get_model_parallelism_config() {
  local model="$1"
  local parallelism="$2"
  local config_file="$3"
  local key="$model.$parallelism"
  
  if [ ! -f "$config_file" ]; then
    echo "Config file $config_file not found!"
    exit 1
  fi

  # Extract values with jq
  local json=$(jq -r --arg m "$model" --arg p "$parallelism" '.[$m][$p]' "$config_file")
  echo "$json"
}

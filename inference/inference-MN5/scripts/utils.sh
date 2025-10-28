#!/bin/bash

# --- Utility to get dataset path from config file ---
get_dataset_path() {
  local dataset="$1"
  local json_file="$2"
  grep -oP "\"$dataset\"\\s*:\\s*\"[^\"]+\"" "$json_file" | \
    sed -E "s/.*: \"([^\"]+)\"/\1/"
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

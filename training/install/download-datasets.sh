#!/bin/bash

set -euo pipefail
SCRIPT_ABSPATH=$(realpath "$0")
PARENT_DIR=$(dirname "$SCRIPT_ABSPATH")
SCRIPT_NAME=$(basename "$SCRIPT_ABSPATH")

source "$PARENT_DIR/utils.sh"
echo -e "$DATASETS_LOGO_MAIN"

if  [[ "$(basename "$PWD")" != "training" && "$(basename "$PWD")" != "install" ]];then
    echo -e "$RED>> Error:'$SCRIPT_NAME' must be executed inside minerva-benchmarks/training or minerva-benchmarks/training/install$RESET"
    exit 1
fi

temp_env=".temp_env_datasets"
datasets_path="datasets"
if [ "$(basename $PWD)" == "training" ];then 
    temp_env="envs/$temp_env"
fi
if [ "$(basename $PWD)" == "install" ];then 
    datasets_path="../$datasets_path"
fi

# if uv 
if ! command -v uv &> /dev/null; then
    echo -e "$RED>> Error: 'uv' command could not be found! Make sure that 'uv' is available before executing $SCRIPT_NAME!!$RESET" >&2
    exit 2
fi


echo -e "${YELLOW}>>${GREEN} Creating temp environment on: '$temp_env'${RESET}"
uv venv $temp_env -q --clear
source "$temp_env/bin/activate"
uv pip install huggingface-hub

dataset="yahma/alpaca-cleaned"
download_folder="$datasets_path/alpaca"
echo -e "${YELLOW}>>${GREEN} Downloading dataset '$dataset' in folder '$download_folder'...${RESET}"
hf download $dataset --repo-type dataset --local-dir "$download_folder"

dataset="rajpurkar/squad_v2"
download_folder="$datasets_path"
echo -e "${YELLOW}>>${GREEN} Downloading dataset '$dataset' in folder '$download_folder'...${RESET}"
hf download $dataset --repo-type dataset --local-dir "$download_folder"

rm -rf $temp_env
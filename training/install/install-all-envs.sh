#!/bin/bash

set -euo pipefail
SCRIPT_ABSPATH=$(realpath "$0")
PARENT_DIR=$(dirname "$SCRIPT_ABSPATH")
SCRIPT_NAME=$(basename "$SCRIPT_ABSPATH")

source "$PARENT_DIR/utils.sh"
echo -e "$INSTALLATION_LOGO_MAIN"

if  [[ "$(basename "$PWD")" != "training" && "$(basename "$PWD")" != "install" ]];then
    echo -e "$RED>> Error:'$SCRIPT_NAME' must be executed inside minerva-benchmarks/training or minerva-benchmarks/training/install$RESET"
    exit 1
fi

folder_bench="envs/benchmarks"
folder_cli="envs/cli"
if [ "$(basename $PWD)" == "install" ];then 
    folder_bench="../$folder_bench"
    folder_cli="../$folder_cli"
fi

# if uv 
if ! command -v uv &> /dev/null; then
    echo -e "$RED>> Error: 'uv' command could not be found! Make sure that 'uv' is available before executing $SCRIPT_NAME!!$RESET" >&2
    exit 2
fi

base_dir=$PWD
echo -e "$YELLOW>>$BLUE Installing environment in '$folder_cli'...$RESET"
echo ""
cd $folder_cli && uv sync
cd "$base_dir"
echo -e "$YELLOW>>$BLUE Installing environments in '$folder_bench'...$RESET"
echo ""

for dir in "$folder_bench"/*/; do
    dirname=$(basename "$dir") 
    echo -e "$YELLOW>>$BLUE Installing environment in '$folder_bench/$dirname/'...$RESET"
    echo ""
    
    cd "$dir" && uv sync
    cd $base_dir
done


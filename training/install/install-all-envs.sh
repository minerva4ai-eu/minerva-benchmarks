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

echo -e "$YELLOW>>$BLUE The following environments will be installed:$RESET"
echo -e "$YELLOW  - CLI environment in '$folder_cli'$RESET"
for dir in "$folder_bench"/*/; do
    echo -e "$YELLOW  - Benchmark environment in '$dir'$RESET"
done

if [ ! -f "$folder_cli/pyproject.toml" ]; then
    echo -e "$RED>> Warning: no 'pyproject.toml' found in '$folder_cli'. Folder skipped.$RESET"
elif [ -d "$folder_cli/.venv" ]; then
    echo -e "$GREEN>> Environment already exists in '$folder_cli/.venv'. Skipping to the next.$RESET"
else
    echo -e "$YELLOW>>$BLUE Installing environment in '$folder_cli'...$RESET"
    echo ""
    cd $folder_cli && uv sync
    cd "$base_dir"
fi
echo -e "$YELLOW>>$BLUE Installing environments in '$folder_bench'...$RESET"
echo ""

for dir in "$folder_bench"/*/; do
    dirname=$(basename "$dir") 
    echo ""
    echo -e -n ">> Do you want to proceed with the installation of uv env $YELLOW'$dirname'$RESET? [y/N] "
    read -e -r confirm
    if [[ ! "$confirm" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo -e "$RED>> Skipping $YELLOW'$dirname'$RESET..."
        continue
    fi
    echo ""
    echo -e "$YELLOW>>$BLUE Installing environment in '$folder_bench/$dirname/'...$RESET"
    
    
    if [ ! -f "$dir/pyproject.toml" ]; then
        echo -e "$RED>> Warning: no 'pyproject.toml' found in '$dir'. Folder skipped.$RESET"
        continue
    fi
    if [ -d "$dir/.venv" ]; then
        echo -e "$GREEN>> Environment already exists in '$dir.venv'. Skipping to the next.$RESET"
        continue
    fi
    cd "$dir" && uv sync
    cd $base_dir
done


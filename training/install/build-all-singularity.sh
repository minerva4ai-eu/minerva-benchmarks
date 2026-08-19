#!/bin/bash

set -euo pipefail
SCRIPT_ABSPATH=$(realpath "$0")
PARENT_DIR=$(dirname "$SCRIPT_ABSPATH")
SCRIPT_NAME=$(basename "$SCRIPT_ABSPATH")

source "$PARENT_DIR/utils.sh"
echo -e "$LOGO_MAIN"

if  [[ "$(basename "$PWD")" != "training" && "$(basename "$PWD")" != "install" ]];then
    echo -e "$RED>> Error: '$SCRIPT_NAME' must be executed inside 'minerva-benchmarks/training' or 'minerva-benchmarks/training/install'!$RESET"
    exit 1
fi

folder_bench="envs/benchmarks"
if [ "$(basename $PWD)" == "install" ];then 
    folder_bench="../$folder_bench"
fi

# if singularity 
if ! command -v singularity &> /dev/null && ! command -v apptainer &> /dev/null; then
    echo -e "$RED>> Error: 'singularity' or 'apptainer' command could not be found! Make sure that either is available before executing $SCRIPT_NAME!!$RESET" >&2
    exit 2
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    CONTAINER_CMD="apptainer"
fi

echo -e "${GREEN}>> Using '$CONTAINER_CMD' as the container runtime.${RESET}"

base_dir=$PWD
echo -e "$YELLOW>>$BLUE Building $CONTAINER_CMD environments in '$folder_bench'...$RESET"
echo ""

for dir in "$folder_bench"/*/; do
    dirname=$(basename "$dir") 
    echo -e "$YELLOW>>$BLUE Building $CONTAINER_CMD in '$folder_bench/$dirname/'...$RESET"
    def_file=$(find "$dir" -maxdepth 1 -name "*.def" -print -quit)
    def_file=$(basename "$def_file")
    def_name="$(basename "$def_file" .def)"
    cd "$dir" && $CONTAINER_CMD build "$def_name.sif" "$def_file"
    cd "$base_dir"
    echo ""
done


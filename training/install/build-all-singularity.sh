#!/bin/bash

set -euo pipefail
SCRIPT_ABSPATH=$(realpath "$0")
PARENT_DIR=$(dirname "$SCRIPT_ABSPATH")
SCRIPT_NAME=$(basename "$SCRIPT_ABSPATH")

source "$PARENT_DIR/utils.sh"
echo -e "$INSTALLATION_LOGO_MAIN"

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

echo -e "$YELLOW>>$BLUE The following $CONTAINER_CMD containers will be built:$RESET"
for dir in "$folder_bench"/*/; do
    def_file=$(find "$dir" -maxdepth 1 -name "*.def" -print -quit)
    if [ -n "$def_file" ]; then
        def_name="$(basename "$def_file" .def)"
        echo -e "$YELLOW  - '$dir$def_name.sif'$RESET"
    fi
done

echo -e "$YELLOW>>$BLUE Building $CONTAINER_CMD environments in '$folder_bench'...$RESET"
echo ""

for dir in "$folder_bench"/*/; do
    dirname=$(basename "$dir")
    def_file=$(find "$dir" -maxdepth 1 -name "*.def" -print -quit)
    def_name="$(basename "$def_file" .def)"
    echo ""
    echo -e -n ">> Do you want to proceed with the build of $YELLOW'$def_file'$RESET? [y/N] " 
    read -e -r confirm
    if [[ ! "$confirm" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo -e "$RED>> Skipping '$def_file'...$RESET"
        continue
    fi
    
    echo -e "$YELLOW>>$BLUE Building $CONTAINER_CMD in '$folder_bench/$dirname/'...$RESET"
    if [ -z "$def_file" ]; then
        echo -e "$RED>> Warning: no '.def' file found in '$dir'. Folder skipped.$RESET"
        echo ""
        continue
    fi
    def_file=$(basename "$def_file")
    def_name="$(basename "$def_file" .def)"

    if [ -f "$dir$def_name.sif" ]; then
        echo -e "$GREEN>> Container '$dir$def_name.sif' already exists. Skipping to the next.$RESET"
        echo ""
        continue
    fi
    cd "$dir" && $CONTAINER_CMD build "$def_name.sif" "$def_file"
    cd "$base_dir"
    echo ""
done


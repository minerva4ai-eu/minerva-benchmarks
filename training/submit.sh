#!/bin/bash

# Source common configuration
source config.sh

# Function to display run help
show_run_help() {
    echo "Usage: ./submit.sh [OPTIONS]"
    echo ""
    echo "Generate all valid configs and submit all pending jobs."
    echo ""
    echo "Options:"
    echo "  --dry-run              Perform a dry run without submitting jobs"
    echo "  --configs-path PATH    Path to configs directory [default: $DEFAULT_CONFIGS_PATH]"
    echo "  --config-name NAME     Config name [default: $DEFAULT_CONFIG_NAME]"
    echo "  --config-env ENV       Config environment [default: container]"
    echo "  --out-dir DIR          Output directory [default: $OUT_DIR]"
    echo "  --mini-mode            Enable mini mode with reduced combinations"
    echo "  --help                 Show this message and exit"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --configs-path)
            CONFIGS_PATH="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --config-env)
            CONFIG_ENV="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --mini-mode)
            MINI_MODE=true
            shift
            ;;
        --help)
            show_run_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${RESET}" >&2
            show_run_help
            exit 1
            ;;
    esac
done

echo ""
echo -e "${POINT_DIAMOND} ${CYAN} Running ${MAGENTA} MINERVA Benchmarks ${CYAN} for LLMs training and fine-tuning ${POINT_DIAMOND} ${RESET}"

# MINERVA_DIR=$(dirname "$0")

if $DRY_RUN; then

    module load jq parallel

    # Read benchmark config data
    get_config

    CONFIGS=()
    # Rule Check: Model-Framework-Parallelism
    for model_name in "${model_names[@]}"; do
        mapfile -t frameworks < <(echo $CONFIG_DATA | jq -r ".models.\"$model_name\".frameworks_supported[]")
        mapfile -t parallelisms < <(echo $CONFIG_DATA | jq -r ".models.\"$model_name\".parallelism_supported[]")
        
        # Launch with Rule Check
        mapfile -t configs < <(parallel -k "if check_config $model_name {}; then echo $model_name {}; fi" ::: ${frameworks[@]} ::: ${parallelisms[@]} ::: ${node_configs[@]} ::: ${dataset_names[@]} ::: ${batch_sizes[@]} ::: ${grad_accums[@]} ::: ${enable_compile[@]} ::: ${enable_bf16[@]} ::: ${lrs[@]} ::: $(seq $trials))
        
        # Pretty Print
        echo "Generated ${#configs[@]} total configurations for model: $model_name"
        for config in "${configs[@]}"; do
            config="${config//Checking configs}"
            if [[ $config == *"Skipping"* ]]; then
                echo "Invalid configuration: $config"
            elif [[ -z $config ]]; then
                continue
            else
                # echo here
                # printf '%s\n' "$config"
                # config="${config//\n/""}"
                CONFIGS+=("$config")
            fi
        done
    done
    
    # Pretty Print
    echo Valid Configs:
    echo -e "Model\t\tFramework\t\tParallelism\tNodes\tDataset\tBatch Size\tGrad Accum\tCompile\tBF16\tLR\tTrial"
    for config in "${CONFIGS[@]}"; do
        shmonfig=($config)
        display_config="${shmonfig[0]}\t${shmonfig[1]}\t"
        if [[ "${#display_config}" -lt 31 ]]; then
            display_config+="\t"
        fi
        display_config+="${shmonfig[2]}\t\t${shmonfig[3]}\t${shmonfig[4]}\t${shmonfig[5]}\t\t${shmonfig[6]}\t\t${shmonfig[7]}\t${shmonfig[8]}\t${shmonfig[9]}\t${shmonfig[10]}\t${shmonfig[11]}"
        echo -e $display_config
    done

    echo -e "\nPrepared ${#CONFIGS[@]} valid configurations."
    
else

    # echo $MINI_MODE
    # export MINI_MODE
    JOB_ID=$(sbatch $CONFIG_NAME.job)
    # JOB_ID=$(sbatch --export=MINI_MODE $CONFIG_NAME.job)

    echo -e "\nSlurm monitor ID: $JOB_ID"
fi

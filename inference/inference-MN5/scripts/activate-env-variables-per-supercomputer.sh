#!/bin/bash
#
# Usage:
#   source activate-env-variables-per-supercomputer.sh
#
# Example:
#   source activate-env-variables-per-supercomputer.sh
#


case "$MACHINE" in
    bsc-mn5-acc)
        # NCCL variables
        export NCCL_NET=IB
        export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
        export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
        export NCCL_DEBUG=TRACE
        export NCCL_NVLS_ENABLE=0
        export NCCL_IB_DISABLE=0

        # Singularity variables
        # export SINGULARITY_EXEC_COMMAND="singularity exec -B /gpfs:/gpfs --nv -C $SGLANG_IMAGE"
        # export SINGULARITY_EXEC_COMMAND="singularity exec -B /gpfs:/gpfs --no-home --nv -C $SGLANG_IMAGE"
        export BINDINGS_SINGULARITY="/gpfs:/gpfs,$CUR_DIR/tmp:/tmp,$CUR_DIR/tmp:/home/bsc"
        export ADDITIONAL_SINGULARITY_ARGS="--no-home --nv -C"
        
        # CUDA DEVICES
        export CUDA_VISIBLE_DEVICES="0,1,2,3"

        # PYTORCH
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export CUDA_LAUNCH_BLOCKING=1
        ;;

    leonardo)
        export COMPILER=nvhpc
        export CUDA_HOME=/cineca/prod/CUDA/12.1
        ;;

    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

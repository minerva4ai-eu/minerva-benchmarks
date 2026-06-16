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
        echo "NCCL_NET: $NCCL_NET"
        export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
        echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
        export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
        echo "NCCL_IB_HCA: $NCCL_IB_HCA"
        export NCCL_DEBUG=INFO
        echo "NCCL_DEBUG: $NCCL_DEBUG"
        export NCCL_NVLS_ENABLE=0
        echo "NCCL_NVLS_ENABLE: $NCCL_NVLS_ENABLE"
        export NCCL_IB_DISABLE=0
        echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"

        # CUDA DEVICES
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


        # PYTORCH
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
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

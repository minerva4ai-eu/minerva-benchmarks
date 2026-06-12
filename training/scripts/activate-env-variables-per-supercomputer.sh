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
        #export NCCL_NET=IB
        echo "NCCL_NET: $NCCL_NET"
        export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
        echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
        export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
        echo "NCCL_IB_HCA: $NCCL_IB_HCA"
        export NCCL_DEBUG=INFO
        echo "NCCL_DEBUG: $NCCL_DEBUG"
        export NCCL_NVLS_ENABLE=0
        echo "NCCL_NVLS_ENABLE: $NCCL_NVLS_ENABLE"
        export NCCL_IB_DISABLE=1
        echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"

        # CUDA DEVICES
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


        # PYTORCH
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
        export CUDA_LAUNCH_BLOCKING=1
        ;;

    leonardo)
        # NOTE: do NOT set NCCL_NET=IB on Leonardo: the NCCL build inside the
        # container does not ship the IB plugin, so forcing the IB transport
        # fails with "Error: network IB not found.". Letting NCCL auto-detect
        # the transport works correctly on Leonardo.
        # Likewise NCCL_IB_HCA / NCCL_SOCKET_IFNAME / NCCL_IB_DISABLE / NCCL_NET
        # are intentionally left unset.

        # CUDA devices: 4 GPUs per Leonardo Booster node.
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        export NCCL_DEBUG=INFO
        
        # PYTORCH allocator: expandable_segments is unsupported on Leonardo's
        # GPUs (A100), so do not enable it — it triggers a runtime warning and
        # has no effect.
        # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # export CUDA_LAUNCH_BLOCKING=1   # debugging only — disables async kernels
        ;;
 
 

    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

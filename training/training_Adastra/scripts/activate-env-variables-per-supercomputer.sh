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
        
    cines-adastra-mi250)
        # RCCL
        export NCCL_SOCKET_IFNAME=hsn
        export NCCL_NET_GDR_LEVEL=PHB

        export TORCH_BLAS_PREFER_HIPBLASLT=1
        export HIP_FORCE_DEV_KERNARG=1
        export HSA_ENABLE_SDMA=0
        export HSA_FORCE_FINE_GRAIN_PCIE=1
        export FI_CXI_ATS=0
        export FI_CXI_RDZV_THRESHOLD=0

        export OMP_NUM_THREADS=32
        export MIOPEN_DISABLE_CACHE=1

        export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
        # GPU visibility
        export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        ;;

    cines-adastra-mi300)
        # RCCL
        export NCCL_SOCKET_IFNAME=hsn
        export NCCL_NET_GDR_LEVEL=PHB

        export OMP_NUM_THREADS=32

        # export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

        # GPU visibility (4 APU)
        export HIP_VISIBLE_DEVICES="0,1,2,3"
        export ROCR_VISIBLE_DEVICES="0,1,2,3"
        ;;
        
    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

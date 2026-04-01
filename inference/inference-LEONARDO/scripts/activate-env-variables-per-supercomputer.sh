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
        export NCCL_NET=IB
        export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
        export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
        export NCCL_DEBUG=TRACE
        export NCCL_NVLS_ENABLE=0
        export NCCL_IB_DISABLE=0
        #export COMPILER=nvhpc
        #export CUDA_HOME=/cineca/prod/CUDA/12.1
        #export CUDA_HOME=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos
        module load cuda/12.3
        module load gcc/12.2.0
        #export CUDA_VISIBLE_DEVICES="0,1,2,3"
        # Prepend pip-installed nvidia libs so they take priority over system CUDA 12.1
        #export LD_LIBRARY_PATH=$EBROOTGCC/lib64:$LD_LIBRARY_PATH
        #export LD_LIBRARY_PATH=$ENVIRONMENT/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
        ;;

    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

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
        export FLASHINFER_WORKSPACE_BASE=/leonardo_work/MNRVA_bench/minerva-benchmarks/inference/inference-LEONARDO
        #export COMPILER=nvhpc
        #export CUDA_HOME=/cineca/prod/CUDA/12.1
        #export CUDA_HOME=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos
        module load cuda/12.6
        module load gcc/12.2.0
        #export CUDA_VISIBLE_DEVICES="0,1,2,3"
        # Prepend gcc/12.2.0 libstdc++ so it takes priority over system gcc-8.5.0 at runtime
        # ($EBROOTGCC is empty on Leonardo, so find the path dynamically via gcc itself)
        _gcc_libstdcxx=$(gcc -print-file-name=libstdc++.so.6 2>/dev/null)
        if [ -n "$_gcc_libstdcxx" ] && [ "$_gcc_libstdcxx" != "libstdc++.so.6" ]; then
            export LD_LIBRARY_PATH=$(dirname "$_gcc_libstdcxx"):$LD_LIBRARY_PATH
        fi
        unset _gcc_libstdcxx
        #export LD_LIBRARY_PATH=$ENVIRONMENT/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
        ;;

    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

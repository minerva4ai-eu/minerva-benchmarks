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
        export ADDITIONAL_SINGULARITY_ARGS="--no-home --nv"
        # -C only works with SGLang.
        
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
    csc-lumi-gpu)
	export NCCL_SOCKET_IFNAME=hsn
        export NCCL_NET_GDR_LEVEL=PHB
	export BINDINGS_SINGULARITY="/var/spool/slurmd,/pfs,/scratch,/projappl,/project,/flash,/appl,/boot"
	export ADDITIONAL_SINGULARITY_ARGS="--no-home --nv"

	export VLLM_RPC_BASE_PATH="$FLASH/tmp/.cache"
	export VLLM_CACHE_ROOT="$FLASH/tmp/.vllm/cache"
	export XDG_CACHE_HOME="$FLASH/tmp/.xdg/cache"
	export TRITON_CACHE_DIR="$FLASH/tmp/triton"
	export TORCHINDUCTOR_CACHE_DIR="$FLASH/tmp/inductor"

	# GPU visibility
        export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_NODE - 1)))
        ;;
    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

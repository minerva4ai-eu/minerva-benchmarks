#!/bin/bash
#
# Usage:
#   source activate-env-per-supercomputer.sh <environment>
#
# Example:
#   source activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM
#

# -- Arguments ---
if [ $# -ne 1 ]; then
  echo "Usage: source $0 <environment>"
  return 1 2>/dev/null || exit 1
fi

ENVIRONMENT="$1"

case "$MACHINE" in
    bsc-mn5-acc)
        # How to activate miniforge environment in mn5-acc.
        module load $MODULES
        source activate $ENVIRONMENT
        export PATH=$ENVIRONMENT/bin:$PATH
        which python
        ;;

    leonardo)
        # NCCL variables for Leonardo InfiniBand
        export NCCL_NET=IB
        export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
        export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
        export NCCL_DEBUG=TRACE
        export NCCL_NVLS_ENABLE=0
        export NCCL_IB_DISABLE=0
        module load $MODULES
        source $ENVIRONMENT/bin/activate
        # CUDA DEVICES
        export CUDA_VISIBLE_DEVICES="0,1,2,3"

        # PYTORCH
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export CUDA_LAUNCH_BLOCKING=1
        ;;

    idris-jeanzay-h100)
        module load $MODULES
        source activate $ENVIRONMENT
        export PATH=$ENVIRONMENT/bin:$PATH
        which python
        ;;


    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

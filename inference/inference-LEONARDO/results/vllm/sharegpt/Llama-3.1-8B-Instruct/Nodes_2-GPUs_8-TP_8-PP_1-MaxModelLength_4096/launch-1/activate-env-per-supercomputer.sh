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

        export NCCL_NET=IB
        export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
        export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
        export NCCL_DEBUG=TRACE
        export NCCL_NVLS_ENABLE=0
        export NCCL_IB_DISABLE=0
        
        #export COMPILER=nvhpc
        #export CUDA_HOME=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos
        module load cuda/12.3
        module load gcc/12.2.0
        
        source $ENVIRONMENT/bin/activate
        # Prepend pip-installed nvidia libs so they take priority over the system CUDA
        # (system libnvJitLink.so.12 may lack symbols required by pip-installed cusparse/NCCL)
        #export LD_LIBRARY_PATH=$EBROOTGCC/lib64:$LD_LIBRARY_PATH
        #export LD_LIBRARY_PATH=$ENVIRONMENT/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
        which python
        ;;

    *)
        echo "Unknown machine: $MACHINE"
        exit 1
        ;;
esac

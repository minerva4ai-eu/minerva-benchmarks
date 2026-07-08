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

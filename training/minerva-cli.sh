#!/bin/bash

CLI_CONTAINER_PATH="./envs/cli/singularity-uv.sif"

module -s load singularity
cli_args="$@"

# Leonardo runs Slurm in "configless" mode: no slurm.conf on disk.
# The local slurmd daemon fetches the cluster config from the controller and
# caches it under /var/spool/slurmd/conf-cache (exposed via /run/slurm/conf).
# Bind that cache into the container so sbatch/sacct can read the config
# instead of attempting a DNS SRV lookup (which would fail inside the container).
SLURM_CONF_CACHE="/var/spool/slurmd/conf-cache"
if [[ ! -f "$SLURM_CONF_CACHE/slurm.conf" ]]; then
    echo "ERROR: Slurm configless cache not found at $SLURM_CONF_CACHE/slurm.conf" >&2
    echo "       Cannot run sbatch from inside the container without it." >&2
    exit 1
fi

singularity exec --env CWD="$PWD" \
    --bind "$HOME":"$HOME" \
    --bind "$PWD":"$PWD" \
    --bind /etc/passwd:/etc/passwd \
    --bind /etc/group:/etc/group \
    --bind $(which sbatch):/usr/local/bin/sbatch \
    --bind $(which sacct):/usr/local/bin/sacct \
    --bind /var/run/munge:/var/run/munge \
    --bind /etc/munge:/etc/munge \
    --bind /usr/lib64/slurm:/usr/lib64/slurm \
    --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 \
    --bind "$SLURM_CONF_CACHE":/run/slurm/conf \
    "$CLI_CONTAINER_PATH" bash -c "cd $PWD && python -m scripts.slurm.cli $cli_args"

exit 0
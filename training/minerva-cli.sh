#!/bin/bash

CLI_CONTAINER_PATH="./envs/cli/singularity-uv.sif"

module load singularity 2> /dev/null
cli_args="$@"

singularity exec --env CWD="$PWD" \
    --bind "$HOME":"$HOME" \
    --bind "$PWD":"$PWD" \
    --bind /etc/passwd:/etc/passwd \
    --bind /etc/group:/etc/group \
    --bind $(which sbatch):/usr/local/bin/sbatch \
    --bind $(which sacct):/usr/local/bin/sacct \
    --bind /var/run/munge:/var/run/munge \
    --bind /etc/munge:/etc/munge \
    --bind /etc/slurm:/etc/slurm \
    --bind /usr/lib64/slurm:/usr/lib64/slurm \
    --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 \
    --bind /lib64/libc.so.6:/lib64/libc.so.6 \
    --bind /lib64/libm.so.6:/lib64/libm.so.6 \
    --bind /lib64/libresolv.so.2:/lib64/libresolv.so.2 \
    "$CLI_CONTAINER_PATH" bash -c "cd $PWD && python -m scripts.slurm.cli $cli_args"

exit 0
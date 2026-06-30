#!/bin/bash

USE_SINGULARITY=0
cli_args="$@"
if [ "$USE_SINGULARITY" -eq 1 ]; then
    CLI_CONTAINER_PATH="./envs/cli/singularity-uv.sif"

    # Load singularity module only if not already available
    if ! singularity --version &> /dev/null; then
        module load singularity 2> /dev/null
    fi

    # If using singularity container from envs/cli/singularity-uv.def
    # must mount usr home, accounts info and slurm paths
    # Mounts below are specific to MN5.
    # !Must be adapted to SLURM path of runninc HPC"
    singularity exec --env CWD="$PWD" \
        --bind "$HOME":"/tmp_home" \
        --bind "$PWD":"$PWD" \
        --bind /etc/passwd:/etc/passwd \
        --bind /etc/group:/etc/group \
        --bind $(which sbatch):/usr/local/bin/sbatch \
        --bind $(which sacct):/usr/local/bin/sacct \
        --bind $(which scancel):/usr/local/bin/scancel \
        --bind /var/run/munge:/var/run/munge \
        --bind /etc/munge:/etc/munge \
        --bind /etc/slurm:/etc/slurm \
        --bind /usr/lib64/slurm:/usr/lib64/slurm \
        --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 \
        --bind /lib64/libc.so.6:/lib64/libc.so.6 \
        --bind /lib64/libm.so.6:/lib64/libm.so.6 \
        --bind /lib64/libresolv.so.2:/lib64/libresolv.so.2 \
        "$CLI_CONTAINER_PATH" bash -c "cd $PWD && python -m scripts.slurm.cli $cli_args"
else
    VENV_PATH="envs/cli/.venv/bin/activate"
    source "$VENV_PATH"
    python -m scripts.slurm.cli $cli_args
fi

exit 0
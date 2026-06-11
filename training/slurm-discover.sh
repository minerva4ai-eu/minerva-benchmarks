#!/bin/bash

# =============================================================================
# HPC-agnostic Singularity launcher for submitting SLURM jobs from inside
# a container. Auto-detects all paths so it works across different HPC systems
# (MareNostrum5, Leonardo, etc.)
# =============================================================================

CLI_CONTAINER_PATH="./envs/cli/singularity-uv.sif"
cli_args="$@"

# =============================================================================
# 1. Load Singularity/Apptainer module silently
#    Different HPC systems use different module names, so we try common ones.
#    -q suppresses module system output; 2>/dev/null catches any remaining
#    stderr; || true prevents script from exiting if none match.
# =============================================================================
_load_singularity() {
    for mod in \
        "singularity" \
        "Singularity" \
        "singularity/latest" \
        "SINGULARITY/3.11.5" \
        "apptainer" \
        "Apptainer"
    do
        if module -q load "$mod" 2>/dev/null; then
            return 0
        fi
    done

    # Module load failed — check if binary is already in PATH (some systems
    # don't use modules at all)
    if command -v singularity &>/dev/null || command -v apptainer &>/dev/null; then
        return 0
    fi

    echo "WARNING: could not load singularity/apptainer module" >&2
    return 1
}
_load_singularity 2>/dev/null

# Normalize command name — newer HPC systems renamed Singularity to Apptainer
if command -v singularity &>/dev/null; then
    SINGULARITY_CMD=singularity
elif command -v apptainer &>/dev/null; then
    SINGULARITY_CMD=apptainer
else
    echo "ERROR: neither singularity nor apptainer found" >&2
    exit 1
fi

# =============================================================================
# 2. Auto-detect SLURM binary paths
#    `which` resolves the real path on the host. We bind these into fixed
#    locations inside the container (/usr/local/bin/) so the Python script
#    can call sbatch/sacct without knowing the host layout.
# =============================================================================
SBATCH_PATH=$(which sbatch)
SACCT_PATH=$(which sacct)

# =============================================================================
# 3. Auto-detect munge socket directory
#    Munge is the authentication daemon SLURM uses to sign job submissions.
#    Without it, sbatch calls inside the container fail with auth errors.
#    The socket path varies by distro:
#      RHEL/CentOS (MN5):   /var/run/munge
#      Ubuntu (Leonardo):   /run/munge
#    We always bind to /var/run/munge inside the container for consistency.
# =============================================================================
if [ -d /var/run/munge ]; then
    MUNGE_RUN=/var/run/munge
elif [ -d /run/munge ]; then
    MUNGE_RUN=/run/munge
else
    echo "WARNING: munge socket directory not found" >&2
    MUNGE_RUN=""
fi

# =============================================================================
# 4. Auto-detect libmunge shared library
#    The munge client library is needed by SLURM binaries at runtime.
#    Path varies by distro:
#      RHEL/CentOS: /usr/lib64/libmunge.so.2
#      Ubuntu:      /usr/lib/x86_64-linux-gnu/libmunge.so.2
# =============================================================================
if [ -f /usr/lib64/libmunge.so.2 ]; then
    MUNGE_LIB=/usr/lib64/libmunge.so.2
elif [ -f /usr/lib/x86_64-linux-gnu/libmunge.so.2 ]; then
    MUNGE_LIB=/usr/lib/x86_64-linux-gnu/libmunge.so.2
else
    MUNGE_LIB=$(find /usr /lib -name "libmunge.so.2" 2>/dev/null | head -1)
fi

# =============================================================================
# 5. Auto-detect SLURM plugin directory
#    SLURM loads plugins (MPI, task, network) at runtime from this directory.
#    Without it, sbatch may fail to initialize or submit correctly.
#    Path varies by distro:
#      RHEL/CentOS: /usr/lib64/slurm
#      Ubuntu:      /usr/lib/x86_64-linux-gnu/slurm
# =============================================================================
if [ -d /usr/lib64/slurm ]; then
    SLURM_LIB=/usr/lib64/slurm
elif [ -d /usr/lib/x86_64-linux-gnu/slurm ]; then
    SLURM_LIB=/usr/lib/x86_64-linux-gnu/slurm
else
    SLURM_LIB=$(find /usr/lib -type d -name "slurm" 2>/dev/null | head -1)
fi

# =============================================================================
# 6. Auto-detect core C runtime libraries
#    SLURM binaries on the host are compiled against the host's libc/libm.
#    The container may have a different (often older) version. We bind the
#    host versions so the host SLURM binaries run correctly inside.
#    `ldconfig -p` queries the host's library cache — most reliable method.
# =============================================================================
LIBC=$(ldconfig -p | grep "libc.so.6" | awk '{print $NF}' | head -1)
LIBM=$(ldconfig -p | grep "libm.so.6" | awk '{print $NF}' | head -1)
LIBRESOLV=$(ldconfig -p | grep "libresolv.so.2" | awk '{print $NF}' | head -1)

# =============================================================================
# 7. Build bind mount arguments
#    Each --bind src:dst maps a host path into the container.
#    We build the string conditionally so missing paths don't cause errors.
# =============================================================================
BINDS=""

# User home — needed so Python scripts can resolve relative paths, write
# logs, and access config files stored under $HOME
BINDS="$BINDS --bind $HOME:$HOME"

# Current working directory — ensures the container sees the same CWD as
# the host, so relative paths in the Python CLI resolve correctly
BINDS="$BINDS --bind $PWD:$PWD"

# User/group databases — without these the container may not resolve the
# current username, causing permission errors in SLURM job accounting
BINDS="$BINDS --bind /etc/passwd:/etc/passwd"
BINDS="$BINDS --bind /etc/group:/etc/group"

# SLURM client config — tells sbatch where the SLURM controller is,
# what partitions exist, and other cluster-specific settings
BINDS="$BINDS --bind /etc/slurm:/etc/slurm"

# SLURM binaries — bound to /usr/local/bin so they're in PATH inside
# the container without needing to know the host's exact bin location
BINDS="$BINDS --bind $SBATCH_PATH:/usr/local/bin/sbatch"
BINDS="$BINDS --bind $SACCT_PATH:/usr/local/bin/sacct"

# Munge config — contains the munge key used to sign/verify auth tokens.
# Required so the munge client inside the container can authenticate
BINDS="$BINDS --bind /etc/munge:/etc/munge"

# Munge socket — the running munge daemon listens on this socket.
# sbatch connects to it to get an auth credential before submitting
[ -n "$MUNGE_RUN" ]  && BINDS="$BINDS --bind $MUNGE_RUN:/var/run/munge"

# Munge shared library — linked by SLURM binaries at runtime for auth
[ -n "$MUNGE_LIB" ]  && BINDS="$BINDS --bind $MUNGE_LIB:/usr/lib64/libmunge.so.2"

# SLURM plugin directory — contains MPI/PMI/task plugins loaded at runtime
[ -n "$SLURM_LIB" ]  && BINDS="$BINDS --bind $SLURM_LIB:/usr/lib64/slurm"

# Host C runtime libraries — ensures host SLURM binaries find the correct
# libc/libm/libresolv they were compiled against, not the container's version
[ -n "$LIBC" ]       && BINDS="$BINDS --bind $LIBC:/lib64/libc.so.6"
[ -n "$LIBM" ]       && BINDS="$BINDS --bind $LIBM:/lib64/libm.so.6"
[ -n "$LIBRESOLV" ]  && BINDS="$BINDS --bind $LIBRESOLV:/lib64/libresolv.so.2"

# =============================================================================
# 8. Debug output — print resolved paths before launching
# =============================================================================
echo "==> Singularity launcher diagnostics"
echo "    cmd:       $SINGULARITY_CMD"
echo "    container: $CLI_CONTAINER_PATH"
echo "    sbatch:    $SBATCH_PATH"
echo "    sacct:     $SACCT_PATH"
echo "    munge run: $MUNGE_RUN"
echo "    munge lib: $MUNGE_LIB"
echo "    slurm lib: $SLURM_LIB"
echo "    libc:      $LIBC"
echo "    libm:      $LIBM"
echo "    libresolv: $LIBRESOLV"

# =============================================================================
# 9. Launch the container
#    --env CWD passes the host CWD as an env var so Python scripts can
#    reference it even if the working directory bind somehow doesn't apply
# =============================================================================
#$SINGULARITY_CMD exec \
#    --env CWD="$PWD" \
#    $BINDS \
#    "$CLI_CONTAINER_PATH" \
#    bash -c "cd $PWD && python -m scripts.slurm.cli $cli_args"
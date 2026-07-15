# Step-Based Job Submission Optimization

This document explains the new step-based job submission approach implemented in `minerva-benchmarks` to optimize benchmark execution.

## Overview

The previous approach in `minerva-benchmarks` submits each benchmark configuration as a separate SLURM job, which can lead to:

1. Increased queue waiting time
2. Higher overhead from repeated environment setup
    ```
    [bsc079516@alogin2 training]$ squeue -s
            STEPID     NAME PARTITION     USER      TIME NODELIST
    43295168.batch    batch       acc bsc07951      0:04 as06r1b29
    43295168.extern   extern       acc bsc07951      0:04 as06r1b29,as06r4b21
    [bsc079516@alogin2 training]$ squeue -s
            STEPID     NAME PARTITION     USER      TIME NODELIST
        43295168.0     bash       acc bsc07951      0:02 as06r1b29
        43295168.1     bash       acc bsc07951      0:02 as06r4b21
    43295168.batch    batch       acc bsc07951      0:09 as06r1b29
    43295168.extern   extern       acc bsc07951      0:09 as06r1b29,as06r4b21
    ```
3. More complex job management and monitoring
4. Limited number of jobs can be submitted
    ```
    [bsc079516@alogin4 ~]$ sacctmgr show qos where name=acc_bscaii format=Name%20,MaxJobsPU,MaxSubmitPU
                    Name MaxJobsPU MaxSubmitPU 
    -------------------- --------- ----------- 
            acc_bscaii                   366
    ```

The new step-based approach addresses these issues by:

1. Submitting a single master job with configuration-wise steps
2. Shared and isolated environment setup
3. Simplifying resource management/monitoring

    ```
    [bsc079516@alogin2 training]$ squeue -s
            STEPID     NAME PARTITION     USER      TIME NODELIST
        43300846.0     bash       acc bsc07951      1:14 as04r3b18
        43300846.1     bash       acc bsc07951      1:14 as04r3b21
    43300846.batch    batch       acc bsc07951      1:18 as04r3b18
    43300846.extern   extern       acc bsc07951      1:18 as04r3b[18,21]
    [bsc079516@alogin2 training]$ squeue -s
            STEPID     NAME PARTITION     USER      TIME NODELIST
        43300846.5     bash       acc bsc07951      2:31 as04r3b21
        43300846.11     bash       acc bsc07951      0:04 as04r3b18
    43300846.batch    batch       acc bsc07951      7:30 as04r3b18
    43300846.extern   extern       acc bsc07951      7:30 as04r3b[18,21]
    [bsc079516@alogin2 training]$ sacct -j 43300846 -o jobid,alloctres%50,state,elapsed
    JobID                                                 AllocTRES      State    Elapsed 
    ------------ -------------------------------------------------- ---------- ---------- 
    43300846     billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2    RUNNING   00:08:33 
    43300846.ba+              cpu=160,gres/gpu=4,mem=500000M,node=1    RUNNING   00:08:33 
    43300846.ex+ billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2    RUNNING   00:08:33 
    43300846.0                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:01:26 
    43300846.1                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:01:26 
    43300846.2                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:03:26 
    43300846.3                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:03:50 
    43300846.4                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:00:02 
    43300846.5                cpu=160,gres/gpu=4,mem=500000M,node=1    RUNNING   00:03:34 
    43300846.6                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:00:02 
    43300846.7                 cpu=40,gres/gpu=1,mem=125000M,node=1  COMPLETED   00:00:35 
    43300846.8                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:00:03 
    43300846.9                cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:01:23 
    43300846.10               cpu=160,gres/gpu=4,mem=500000M,node=1  COMPLETED   00:00:03 
    43300846.11               cpu=160,gres/gpu=4,mem=500000M,node=1    RUNNING   00:01:07
    ```

4. More allowance (e.g. MaxStepCount=40000)

    ```
    [bsc079516@alogin4 ~]$ scontrol show config | grep -i step
    InteractiveStepOptions  = --interactive --preserve-env --pty $SHELL
    LaunchParameters        = use_interactive_step
    MaxStepCount            = 40000
    UnkillableStepProgram   = (null)
    UnkillableStepTimeout   = 360 sec
    ```

## Implementation Details

### Training Benchmarks

1. **General entrypoint**: The CLI (`main.sh`) has the same subcommands `run`, `status`, and `cancel`
2. **Workers**: Each configuration gets a dedicated `work.sh` launch that encapsulates its environment and execution logic
3. **Main Job**: GNU `parallel` and `srun` to execute all configurations as steps in machine-specific job scripts

### Inference Benchmarks

1. **Worker**: #TODO
2. **Main Job**: #TODO

## Usage

### Training Benchmarks

Examine configurations:
```bash
./main.sh run --dry-run --config-name MN5
```

Submit benchmark:
```bash
./main.sh run --config-name MN5 --config-env singularity
```

Submit subset:
```bash
./main.sh run --config-name MN5 --mini-mode
```

### Inference Benchmarks

#TODO

## Benefits

1. **Reduced Queue Time**: Only one job submission instead of many
2. **Lower Overhead**: Shared environment setup and resource allocation
3. **More Configurations**: Steps can be scheduled more efficiently
4. **Simplified Management**: Single job to monitor instead of many

## Compatibility

The step-based approach does not change the underlying benchmark execution logic. However, configuration is heavily related to task launching, so these definitions along with the CLI are modified (python dependencies for the CLI and Configuration are removed). Monitoring is also modified here and being discussed in terms of overhead affecting performance.

> This is not an official proprosal to change the exiting functionalities for CLI and Configuration; given the objective of optimizing the task submission, the developer thought it easier to focus on this in isolation, and planned for subsequent steps for integrating with existing functionalities with python and it's dependencies.

### Major changes
1. Configurations: minimize config file reading/writing 
    - machine-specific configurations in job script (`$CONFIG_NAME.job`)
    - cross-machine configurations in a single JSON (`benchmark.json`)
2. CLI: flexibility and control
    - orchestrate configuration and submission
    - Global and local environments are set at the appropriate levels
3. Monitoring: "separation of concerns"
    - eliminate errors with CPU
    - plotting is done outside of main run


## References

- [PEARC 24 Tutorial](https://github.com/ketancmaheshwari/pearc24tut)
- [U Chicago User Guide - Tutorial](https://docs.rcc.uchicago.edu/tutorials/kicp/#gnu-parallel)
- [U Chicago User Guide - Slurm](https://docs.rcc.uchicago.edu/slurm/sbatch/#gnu-parallel)
- [U Luxembourg Tutorial - Sequential](https://ulhpc-tutorials.readthedocs.io/en/latest/sequential/basics/)
- [U Luxembourg Tutorial - Distributed](https://ulhpc-tutorials.readthedocs.io/en/latest/sequential/basics/)
- [U Berkeley User Guide](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/gnu-parallel/)
- [U Colorado Boulder Docs](https://curc.readthedocs.io/en/latest/software/GNUParallel.html)

## TODO

- [ ] Clarify SBATCH parameters
- [ ] Implement `status` and `cancel`
- [ ] Implement inference
    - [ ] main job
    - [ ] CLI
- [ ] Refactor monitoring (and plotting)
- [ ] Implement DeepSpeed

# Questions
- Do we want dry-run in the job?
- Do we want `plot` in the CLI?
- Are there plans to benchmark different environments (eg torch 2.8 vs 2.13)?
- Do we want to define levels of logging?
- Do we want `setup` in the CLI (build environment)?

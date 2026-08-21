# Job-Step Submission

This document explains a task launch approach with a job-step submission to address bottlenecks in benchmark execution.

## Motivation

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

1. Submitting a single main job with configuration-wise steps
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

## Benefits

1. **Reduced Queue Time**: Only one job submission instead of many
2. **Lower Overhead**: Shared environment setup and resource allocation
3. **More Configurations**: Steps can be scheduled more efficiently
4. **Simplified Management**: Single job to monitor instead of many

## Usage

### Training Benchmarks

Write configuration files:
```bash
./minerva-cli.sh run --config-name MN5-uv-venv-cuda130 --dry-run
```

Run benchmark:
```bash
./minerva-cli.sh run --config-name MN5-uv-venv-cuda130
```

Run configuration:
```bash
./minerva-cli.sh run --yaml /home/bsc/bsc079516/minerva_backup/minerva-benchmarks/training/benchmark-runs-MN5-uv-venv-cuda130/bsc-mn5-acc/19-08-2026/gemma_1b/torchrun/none/alpaca/nodes-1/yaml-configs/none--bs8-grad_accum8-compileFalse-precbf16-steps50.yaml
```

### Inference Benchmarks

#TODO

## Compatibility

The step-based approach does not change the underlying benchmark execution logic. Monitoring is also modified here and being discussed in terms of overhead affecting performance.

> Consider changing the exiting functionalities for CLI and Configuration

### Separation of Concerns
1. Configurations: minimize config file reading/writing 
    - machine-specific configurations in job script (`$CONFIG_NAME.job`)
    - cross-machine configurations in a single JSON (`benchmark.json`)
2. CLI: flexibility and control
    - orchestrate configuration and submission
    - Global and local environments are set at the appropriate levels
3. Monitoring:
    - eliminate errors with CPU
    - do plotting outside of main run


## TODO

- [ ] Clarify SBATCH parameters
- [ ] Implement `status` and `cancel`
- [ ] Implement inference
- [ ] Review monitoring (and plotting)
- [ ] Implement DeepSpeed
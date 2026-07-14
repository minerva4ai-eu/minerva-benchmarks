# Step-Based Job Submission Optimization

This document explains the new step-based job submission approach implemented in `minerva-benchmarks` to optimize benchmark execution.

## Overview

The previous approach in `minerva-benchmarks` submits each benchmark configuration as a separate SLURM job, which can lead to:

1. Increased queue waiting time
2. Higher overhead from repeated environment setup
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
3. Shared and isolated environment setup
4. Simplifying job management
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

The step-based approach does not change the underlying benchmark execution logic.


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
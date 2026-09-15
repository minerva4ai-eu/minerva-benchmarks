from typing import Any
import os

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
import torch

from .chrono import TrainingChronometer
from .comm_measurements import comm_profiler, get_comm_results
from .cpu_mem_usage import memory_usage
from .nccl_tagger import NCCLTagger


class MegatronBenchmarkCallback(pl.Callback):
    """Benchmark callback for `nl.MegatronStrategy`.

    Under MegatronStrategy, the full forward-backward over all micro-batches + the
    optimizer step run INSIDE a single training_step. Consequences vs the FSDP2 version:
        * on_train_batch_start/end fire once per GLOBAL step (not per micro-batch).
        * on_before_backward / on_before_optimizer_step do not fire reliably, so
          per-phase (HtoD/Forward/Backward) timing and per-phase NCCL tagging are unavailable.
          We therefore measure one combined time per global step.
        * Every count below is in GLOBAL STEPS, including the warmup.
    """

    def __init__(self, rank: int, n_steps: int, global_batch_size: int, seq_len: int,
                 steps_per_epoch: int, n_warmup_steps: int = 1) -> None:
        self.rank = rank
        self.n_steps = n_steps                      # total global steps (weight updates)
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch      # total global steps per epoch
        self.n_warmup_steps = n_warmup_steps        # global steps to ignore at the start
        self.chronometer = TrainingChronometer()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            print(f'Pre-loop GPU memory usage (allocated): {torch.cuda.max_memory_allocated()/1e9} GB')
            print(f'Pre-loop GPU memory usage (reserved):  {torch.cuda.max_memory_reserved()/1e9} GB')

        self.tagger = NCCLTagger()
        self.global_step_idx = 0
        self.tagger.tag(self.global_step_idx, "TRAINING START")

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        # One call = One global step (one weight update over grad_acc * micro-batches).
        self.global_step_idx += 1
        self.tagger.tag(self.global_step_idx, "GLOBAL")  # only step-level tagging is possible
        if self.rank == 0:
            if self.global_step_idx == self.n_warmup_steps + 1:  # start timing after warmup
                self.chronometer.track_cpu_training_time(start=True)

            # Single combined timer: data transfer + fwd + bwd + optimizer all happen inside the global step.
            self.chronometer.track_gpu_step_time(start=True)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.rank == 0:
            self.chronometer.track_gpu_step_time(start=False)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            ## Training results
            self.chronometer.track_cpu_training_time(start=False)

            nccl_log_path = os.environ["NCCL_DEBUG_FILE"]
            nccl_log_path = nccl_log_path.replace("%p", str(os.getpid()))
            comms_profile = comm_profiler([nccl_log_path], [os.getpid()])
            get_comm_results(comms_profile, skip_steps=self.n_warmup_steps)

            # We have the first weight update as warmup
            training_duration = self.chronometer.display_training_results(self.steps_per_epoch, grad_acc=1, skip_steps=self.n_warmup_steps)    
            measured_steps = self.n_steps - self.n_warmup_steps
            print(f'Throughput: {measured_steps*self.global_batch_size*self.seq_len/training_duration:.1f} tokens/s')

            ## Memory Usage
            memory_usage()
            print(f'Post-loop GPU memory usage (allocated): {torch.cuda.max_memory_allocated()/1e9} GB')
            print(f'Post-loop GPU memory usage (reserved):  {torch.cuda.max_memory_reserved()/1e9} GB')
from typing import Any
import os

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
import lightning.pytorch as pl
import torch

from .chrono import TrainingChronometer
from .comm_measurements import comm_profiler, get_comm_results
from .cpu_mem_usage import memory_usage
from .nccl_tagger import NCCLTagger


class BenchmarkCallback(pl.Callback):
    """Custom pl callback.

    Source: https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/callback.html
    Source: https://lightning.ai/docs/pytorch/stable/common/hooks.html
    """

    def __init__(self, rank: int, n_steps: int, global_batch_size: int, seq_len: int, total_batches_per_epoch: int, grad_acc: int = 1) -> None:
        self.rank = rank
        self.n_steps = n_steps
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.total_batches_per_epoch = total_batches_per_epoch
        self.grad_acc = grad_acc
        self.chronometer = TrainingChronometer()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            print(f'Pre-loop GPU memory usage (allocated): {torch.cuda.max_memory_allocated()/1e9} GB')
            print(f'Pre-loop GPU memory usage (reserved):  {torch.cuda.max_memory_reserved()/1e9} GB')

            # Check parameter sharding across devices
            for name, param in pl_module.named_parameters():
                if hasattr(param, 'device_mesh'):
                    print(f"{name}:")
                    print(f"  mesh: {param.device_mesh}")
                    print(f"  shape: {param.device_mesh.shape}")
                    print(f"  placements: {param.placements}")

        # To add user-defined tag in NCCL log
        self.tagger = NCCLTagger()
        self.global_batch_idx = 0
        self.tagger.tag(self.global_batch_idx, "TRAINING START")

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        self.global_batch_idx += 1
        self.tagger.tag(self.global_batch_idx, "Forward")  # NCCL can be called during Forward
        if self.rank == 0:
            if self.global_batch_idx == self.grad_acc + 1:  # ignore the first weight update (considered as warmup)
                self.chronometer.track_cpu_training_time(start=True)

            self.chronometer.track_gpu_HtoD_step_time(start=True)  # no hook before forward to observe data transfer
            self.chronometer.track_gpu_HtoD_step_time(start=False)

            self.chronometer.track_gpu_fwd_step_time(start=True)

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        self.tagger.tag(self.global_batch_idx, "Backward")  # NCCL can be called during Backward
        if self.rank == 0:
            self.chronometer.track_gpu_fwd_step_time(start=False)
            self.chronometer.track_gpu_bwd_step_time(start=True)

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer) -> None:
        self.tagger.tag(self.global_batch_idx, "Optimizer")  # NCCL can be called for gradient clipping

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.rank == 0:
            self.chronometer.track_gpu_bwd_step_time(start=False)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            ## Training results
            self.chronometer.track_cpu_training_time(start=False)

            nccl_log_path = os.environ["NCCL_DEBUG_FILE"]
            nccl_log_path = nccl_log_path.replace("%p", str(os.getpid()))
            comms_profile = comm_profiler([nccl_log_path], [os.getpid()])
            get_comm_results(comms_profile, skip_steps=self.grad_acc)

            # We have the first weight update as warmup
            training_duration = self.chronometer.display_training_results(self.total_batches_per_epoch, self.grad_acc, skip_steps=self.grad_acc)
            print(f'Throughput: {(self.n_steps-1)*self.global_batch_size*self.seq_len/training_duration:.1f} tokens/s')

            ## Memory Usage
            memory_usage()
            print(f'Post-loop GPU memory usage (allocated): {torch.cuda.max_memory_allocated()/1e9} GB')
            print(f'Post-loop GPU memory usage (reserved):  {torch.cuda.max_memory_reserved()/1e9} GB')

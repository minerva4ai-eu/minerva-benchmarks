import logging
import os
import time
from typing import Dict, Optional

import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist
from transformers import (
    Trainer,
    TrainerCallback,
)

# Save tuning results to disk so next run skips benchmarking
inductor_config.autotune_local_cache = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)


def print_rank(rank_or_msg: int | str | None, msg: str | None = None):
    """Prints the message with the rank number.
    Usage:
        print_rank("msg")       -> all ranks
        print_rank(0, "msg")    -> rank 0 only
    """
    if isinstance(rank_or_msg, str):
        rank = None
        msg = rank_or_msg
    else:
        rank = rank_or_msg

    local_rank = dist.get_rank()
    if rank is None or local_rank == rank:
        print(f"[ RANK {local_rank} ]: {msg}")





class PerformanceTrackingTrainer(Trainer):
    """
    Trainer subclass that:
      - Tracks tokens per GPU during training
      - Reduces tokens globally across FSDP/DDP ranks
      - Computes throughput at end of training
    """

    def __init__(
        self,
        train_dataloader=None,
        eval_dataloader=None,
        peak_gpu_tflops=None,
        *args,
        **kwargs,
    ):

        self._num_params = sum(p.numel() for p in kwargs["model"].parameters())
        super().__init__(*args, **kwargs)
        self._num_params_this_gpu = sum(p.numel() for p in kwargs["model"].parameters())
        self.custom_train_dataloader = train_dataloader
        self.custom_eval_dataloader = eval_dataloader

        self.total_tokens_this_gpu = 0
        self.total_tokens_global = None

        self.last_logged_flops_this_gpu = 0
        self.flops_accumulated = 0
        self.logged_flops_this_gpus = []
        self.global_average_flops = 0
        self.avg_flops_this_gpu = 0
        self.logged_mfu_this_gpus = []
        self.global_average_mfu = 0
        self.avg_mfu_this_gpu = 0

        self.training_step_times = []

        # Peak GPU TFLOPs for MFU calculation (bf16/fp16 tensor core peak).
        # e.g. A100 SXM = 312, H100 SXM = 989, MI250X = 383
        # Pass None to skip MFU logging.
        self.peak_gpu_tflops = peak_gpu_tflops

        print_rank(
            0,
            f"Initialized PerformanceTrackingTrainer on rank {dist.get_rank() if dist.is_available() and dist.is_initialized() else 0}",
        )
        print_rank(0, f"Trainer args: \n{self.args}")
        print_rank(0, f"Model architecure: \n{kwargs['model']}")
        print_rank(0, f"Number of model parameters: {self._num_params / 1e9:.2f}B")
        print_rank(
            int(os.environ["RANK"]),
            f"Number of model parameters on this GPU: {self._num_params_this_gpu / 1e9:.2f}B",
        )

        self.step_start_time = None  # ← Add this
        self.step_interval_time = 0.00001  # avoid division by zero in first step
        self.last_log_timestamp = None

        print_rank(0, ":::::::::")
        for i in kwargs["model"].named_parameters():
            print_rank(0, f"{i[0]} -> {i[1].device}")
        print_rank(0, ":::::::::")

    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.custom_eval_dataloader is not None:
            return self.custom_eval_dataloader
        return super().get_eval_dataloader(eval_dataset)

    def calculate_flops_per_sec(self):
        return self.last_logged_flops_this_gpu / self.step_interval_time

    def calculate_mfu(self):
        achieved_tflops_per_sec = (
            self.last_logged_flops_this_gpu / 1e12
        ) / self.step_interval_time
        mfu = achieved_tflops_per_sec / self.peak_gpu_tflops * 100
        return mfu

    # Add this new method:
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log to add custom metrics before logging.
        """
        # Add custom metrics to logs
        if self.state.global_step >= 0:
            if "train_runtime" in logs:
                super().log(logs)  # Log the original metrics first
                return
            # Tokens per second (instantaneous)
            if hasattr(self, "step_start_time") and self.step_start_time:
                step_time = time.time() - self.step_start_time
                tokens_this_log_interval = self.total_tokens_this_gpu - getattr(
                    self, "_last_logged_tokens", 0
                )

                # Calculate token throughput for this step and reduce across GPUs
                # Throughput might be higher than actual due to overlapping steps, but gives a sense of instantaneous performance
                tokens_per_sec_step_device = (
                    tokens_this_log_interval / step_time if step_time > 0 else 0
                )
                self._last_logged_tokens = self.total_tokens_this_gpu

                # device = torch.device("cuda")
                # local = torch.tensor(tokens_per_sec_step_device, device=device)

                # if dist.is_available() and dist.is_initialized():
                #    dist.all_reduce(local, op=dist.ReduceOp.AVG)

                # logs["_tokens_per_sec_step"] = local.item()
            # Cumulative tokens processed
            rank = dist.get_rank()
            if "loss" in logs:
                logs["loss"] = f"{logs['loss']:.4f}"
            if "grad_norm" in logs:
                logs["grad_norm"] = f"{logs['grad_norm']:.4f}"
            if "learning_rate" in logs:
                logs["learning_rate"] = f"{logs['learning_rate']:.4f}"
            # in millions for readability
            logs[f"_total_1M_tokens_gpu_{rank}"] = (
                f"{(self.total_tokens_this_gpu / 1e6):.3f}"
            )

            # GPU memory usage (if available)
            if torch.cuda.is_available():
                logs["_gpu_mem_allocated_gb"] = (
                    f"{torch.cuda.memory_allocated() / 1e9:.2f}"
                )
                logs["_gpu_mem_reserved_gb"] = (
                    f"{torch.cuda.memory_reserved() / 1e9:.2f}"
                )

            # Batch size info
            # logs["effective_batch_size"] = (
            #    self.args.per_device_train_batch_size
            #    * self.args.gradient_accumulation_steps
            #    * self.args.world_size
            # )
        logs["TFLOPs/sec/GPU"] = f"{self.logged_flops_this_gpus[-1] / 1e12:.2f}"
        logs["mfu/GPU"] = f"{self.logged_mfu_this_gpus[-1]:.2f}%"

        # Call parent log method
        super().log(logs)

        # Optional: Print to console with custom format
        # if self.is_world_process_zero():
        #    print(f"\n[Step {self.state.global_step}] Custom Metrics:")
        #    for key, value in logs.items():
        #        if key.startswith("tokens_") or key.startswith("gpu_"):
        #            print(f"  {key}: {value:.4f}")

    # ************************************
    # CORRECT SIGNATURE FOR HF Trainer
    # ************************************
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        HuggingFace Trainer calls this as:
            training_step(model, inputs, num_items_in_batch)
        """

        # ------------------------------
        # Step custom metrics
        # ------------------------------
        if "input_ids" in inputs and inputs["input_ids"] is not None:
            tokens_in_batch = int(inputs["input_ids"].numel())
        elif "labels" in inputs and inputs["labels"] is not None:
            tokens_in_batch = int(inputs["labels"].numel())
        else:
            tokens_in_batch = 0

        # add to running counter
        self.total_tokens_this_gpu += tokens_in_batch

        step_start = time.time()
        torch.cuda.synchronize()  #
        result = super().training_step(model, inputs)
        torch.cuda.synchronize()  #
        self.step_interval_time = time.time() - step_start

        # 6N*T: 2 (matmul FMA) * 3 (fwd + 2x bwd) * params * tokens
        forward_flops = 6 * self._num_params * tokens_in_batch
        self.last_logged_flops_this_gpu = forward_flops

        if self.step_interval_time > 0:
            flops_per_sec = self.calculate_flops_per_sec()
            self.logged_flops_this_gpus.append(flops_per_sec)
            mfu = self.calculate_mfu()
            self.logged_mfu_this_gpus.append(mfu)

        return result

    # ************************************
    # GLOBAL TOKEN ALL-REDUCE AFTER TRAIN
    # ************************************
    def _finalize_token_counts(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local = torch.tensor(self.total_tokens_this_gpu, device=device)

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local, op=dist.ReduceOp.SUM)

            self.total_tokens_global = int(local.item())

        except Exception:
            print_rank(
                0,
                "Warning: Failed to reduce token counts across GPUs! Setting global tokens equal to local tokens on each GPU.",
            )
            self.total_tokens_global = self.total_tokens_this_gpu

    def _finalize_flop_counts(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_average_flops = 0
        local_avg_flops = sum(self.logged_flops_this_gpus) / len(
            self.logged_flops_this_gpus if self.logged_flops_this_gpus else 0
        )
        local_flops = torch.tensor(local_avg_flops, device=device)
        try:
            dist.all_reduce(local_flops, op=dist.ReduceOp.SUM)
            global_average_flops = local_flops.item()
        except Exception:
            print_rank(
                0,
                "Warning: Failed to reduce FLOP counts across GPUs! Setting global FLOPs equal to local FLOPs on each GPU.",
            )
        finally:
            self.global_average_flops = float(global_average_flops)
            self.avg_flops_this_gpu = local_avg_flops
            # self.global_average_flops_per_gpu = (
            #    float(global_average_flops) / dist.get_world_size()
            # )

    def _finalize_mfu_counts(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_average_mfu = 0
        local_avg_mfu = (
            sum(self.logged_mfu_this_gpus) / len(self.logged_mfu_this_gpus)
            if self.logged_mfu_this_gpus
            else 0
        )
        local_mfu = torch.tensor(local_avg_mfu, device=device)
        try:
            dist.all_reduce(local_mfu, op=dist.ReduceOp.SUM)
            global_average_mfu = local_mfu.item() / dist.get_world_size()
        except Exception:
            print_rank(
                0,
                "Warning: Failed to reduce MFU counts across GPUs! Setting global MFUs equal to local MFUs on each GPU.",
            )
        finally:
            self.global_average_mfu = float(global_average_mfu)
            self.avg_mfu_this_gpu = local_avg_mfu
            # self.global_average_mfu_per_gpu = (
            #    float(global_average_mfu) / dist.get_world_size()
            # )

    # ************************************
    # WRAP TRAINING WITH TIMING + FINAL REDUCTION
    # ************************************
    def train(self, *args, **kwargs):
        # reset counters in case reused
        self.total_tokens_this_gpu = 0
        self.total_tokens_global = None

        start_time = time.time()
        output = super().train(*args, **kwargs)
        end_time = time.time()

        # all-reduce token counters
        self._finalize_token_counts()
        self._finalize_flop_counts()
        self._finalize_mfu_counts()
        elapsed = end_time - start_time

        # store into trainer.state for later consumption
        try:
            setattr(self.state, "total_training_seconds_custom", float(elapsed))
            setattr(
                self.state,
                "total_tokens_per_gpu_custom",
                int(self.total_tokens_this_gpu),
            )
            setattr(
                self.state, "total_tokens_global_custom", int(self.total_tokens_global)
            )
            setattr(
                self.state,
                "average_flops_custom",
                float(self.avg_flops_this_gpu),
            )
            setattr(
                self.state,
                "average_mfu_custom",
                float(self.avg_mfu_this_gpu),
            )
        except Exception:
            pass

        # print summary on rank 0
        if self.is_world_process_zero():
            print("\n=== TOKEN / THROUGHPUT SUMMARY (PerformanceTrackingTrainer) ===")
            print(f"Total tokens per GPU (ALL epochs): {self.total_tokens_this_gpu:,}")
            print(f"Total tokens GLOBAL (ALL epochs): {self.total_tokens_global:,}")

            if elapsed > 0:
                print(f"Total training time (s): {elapsed:.2f}")
                print(
                    f"Tokens/sec per GPU: {self.total_tokens_this_gpu / elapsed:,.2f}"
                )
                print(f"Tokens/sec GLOBAL: {self.total_tokens_global / elapsed:,.2f}")
                print(
                    f"Average FLOPs over all GPUs: {self.global_average_flops / 1e12:.2f} TFLOPs/sec"
                )
                print(
                    f"Average FLOPs per GPU: {self.global_average_flops / dist.get_world_size() / 1e12:.2f} TFLOPs/sec/GPU"
                )
                # print(f"Average MFU across all GPUs: {self.global_average_mfu:.2f}%")
                print(f"Average MFU per GPU: {self.global_average_mfu:.2f}%")
            print("==========================================================\n")

        return output

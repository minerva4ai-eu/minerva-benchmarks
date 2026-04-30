import functools
import logging
import os
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers import (
    Trainer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model

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


def timed(attr: str):
    """
    Method decorator that appends the execution time (seconds) of the
    decorated method to the list `self.<attr>` after each call.

    Usage:
        @timed("step_times")
        def training_step(self, ...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            self.__dict__.setdefault(attr, []).append(time.time() - start)
            return result

        return wrapper

    return decorator


def perf_timed(attr: str):
    """
    Method decorator that appends the execution time (seconds) of the
    decorated method to the list `self.<attr>` after each call.

    Usage:
        @perf_timed("step_times")
        def training_step(self, ...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            self.__dict__.setdefault(attr, []).append(time.perf_counter() - start)
            return result

        return wrapper

    return decorator


class TokenTrackingTrainer(Trainer):
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
            f"Initialized TokenTrackingTrainer on rank {dist.get_rank() if dist.is_available() and dist.is_initialized() else 0}",
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

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.
        """
        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        num_tokens = inputs["input_ids"].numel()

        outputs = model(**inputs)  # no FlopCounterMode wrapper

        # 6N*T: 2 (matmul FMA) * 3 (fwd + 2x bwd) * params * tokens
        forward_flops = 6 * self._num_params * num_tokens
        self.last_logged_flops_this_gpu = forward_flops
        self.flops_accumulated += forward_flops

        # OLDER OPTION WITH FlopCounterMode (instrusive, hooking on pytorch dispatcher)
        # next_step = self.state.global_step + 1
        # will_log = next_step % self.args.logging_steps == 0
        # if will_log:
        # Tokens in this micro-batch on this GPU
        # flop_counter = FlopCounterMode(display=False)
        # with flop_counter:
        #    outputs = model(**inputs)
        # forward_flops = float(flop_counter.get_total_flops())
        # self.last_logged_flops_this_gpu = forward_flops * 3  # backward ≈ 2× forward
        # self.flops_accumulated += self.last_logged_flops_this_gpu

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        # Default HF loss handling (label smoothing) if no custom loss function
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= (
                self.accelerator.num_processes
                if self.args.n_gpu <= 1
                else self.args.n_gpu
            )
        # MFU: achieved TFLOPs/s divided by peak hardware TFLOPs/s
        if (
            self.peak_gpu_tflops
            and self.last_logged_flops_this_gpu > 0
            and hasattr(self, "step_start_time")
            and self.step_start_time
        ):
            self.step_interval_time = time.time() - self.step_start_time
            if self.step_interval_time > 0:
                flops_per_sec = self.calculate_flops_per_sec()
                self.logged_flops_this_gpus.append(flops_per_sec)
                mfu = self.calculate_mfu()
                self.logged_mfu_this_gpus.append(mfu)

        # Reset step timer
        self.step_start_time = time.time()
        self.flops_accumulated = 0

        return (loss, outputs) if return_outputs else loss

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

        # account for gradient accumulation (HF normalizes loss for that)
        tokens_in_batch *= max(1, self.args.gradient_accumulation_steps)

        # add to running counter
        self.total_tokens_this_gpu += tokens_in_batch

        return super().training_step(model, inputs)

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
            self.logged_flops_this_gpus
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
        local_avg_mfu = sum(self.logged_mfu_this_gpus) / len(self.logged_mfu_this_gpus)
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
            print("\n=== TOKEN / THROUGHPUT SUMMARY (TokenTrackingTrainer) ===")
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

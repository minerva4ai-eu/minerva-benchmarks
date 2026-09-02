import logging
import os
import time

import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist
from shared.utils import save_summary_stats_json
from trl import SFTTrainer

# Save tuning results to disk so next run skips benchmarking
inductor_config.autotune_local_cache = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch.compiler.config.assume_static_by_default = False
# torch._dynamo.config.capture_dynamic_shapes = True
# torch._dynamo.config.recompile_limit = 16

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

    if dist.is_initialized():
        device_rank = dist.get_rank()
    else:
        device_rank = int(os.environ["RANK"])
    if rank is None or device_rank == rank:
        print(f"[ RANK {device_rank} ]: {msg}")


class PerformanceTrackingSFTTrainer(SFTTrainer):
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
        # model_config = AutoConfig.from_pretrained(kwargs["model"])
        # self._num_params = sum(p.numel() for p in model_config.parameters())
        super().__init__(*args, **kwargs)
        # self._num_params_this_gpu = sum(p.numel() for p in kwargs["model"].parameters())
        self.custom_train_dataloader = train_dataloader
        self.custom_eval_dataloader = eval_dataloader

        self.total_tokens_this_gpu = 0
        self.total_tokens_global = 0
        self.batches_seen_this_gpu = 0

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
        # print_rank(0, f"Model architecure: \n{model_config.architectures}")
        # print_rank(0, f"Number of model parameters: {self._num_params / 1e9:.2f}B")
        # print_rank(
        #    int(os.environ["RANK"]),
        #    f"Number of model parameters on this GPU: {self._num_params_this_gpu / 1e9:.2f}B",
        # )

        self.step_start_time = None  # ← Add this
        self.step_interval_time = 0.00001  # avoid division by zero in first step
        self.last_log_timestamp = None

        # print_rank(":::::::::")
        # layer_limit = 10
        # for idx, i in enumerate(self.model.named_parameters()):
        #    print_rank(f"{i[0]} -> {i[1].device}")
        #    if idx == layer_limit:
        #        break
        # print_rank(":::::::::")

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
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Override log to add custom metrics before logging.
        """
        # Add custom metrics to logs
        if self.state.global_step >= 0:
            if "train_runtime" in logs:
                super().log(logs)  # Log the original metrics first
                return

            # Cumulative tokens processed
            rank = dist.get_rank() if dist.is_initialized() else 0
            if "loss" in logs:
                logs["loss"] = f"{logs['loss']:.4f}"
            if "grad_norm" in logs:
                logs["grad_norm"] = f"{logs['grad_norm']:.4f}"
            if "learning_rate" in logs:
                logs["learning_rate"] = f"{logs['learning_rate']:.5e}"
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
            logs[f"D{os.environ['RANK']}:seen_batches"] = self.batches_seen_this_gpu
            logs[f"D{os.environ['RANK']}:seen_tokens"] = self.total_tokens_this_gpu
            logs["effective_batch_size"] = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * self.args.world_size
            )
        if self.logged_flops_this_gpus:
            logs["TFLOPs/sec/GPU"] = f"{self.logged_flops_this_gpus[-1] / 1e12:.2f}"
        if self.logged_mfu_this_gpus:
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
        self.batches_seen_this_gpu += 1
        self.total_tokens_this_gpu += tokens_in_batch

        step_start = time.time()
        result = super().training_step(model, inputs)
        self.step_interval_time = time.time() - step_start

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
        local_avg_flops = (
            sum(self.logged_flops_this_gpus) / len(self.logged_flops_this_gpus)
            if self.logged_flops_this_gpus
            else 0
        )

        local_flops = torch.tensor(local_avg_flops, device=device)
        if dist.is_available() and dist.is_initialized():
            try:
                dist.all_reduce(local_flops, op=dist.ReduceOp.SUM)
                global_average_flops = local_flops.item()
            except Exception:
                print_rank(
                    0,
                    "Warning: Failed to reduce FLOP counts across GPUs! Setting global FLOPs equal to local FLOPs on each GPU.",
                )
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
        if dist.is_available() and dist.is_initialized():
            try:
                dist.all_reduce(local_mfu, op=dist.ReduceOp.SUM)
                global_average_mfu = local_mfu.item() / dist.get_world_size()
            except Exception:
                print_rank(
                    0,
                    "Warning: Failed to reduce MFU counts across GPUs! Setting global MFUs equal to local MFUs on each GPU.",
                )
        self.global_average_mfu = float(global_average_mfu)
        self.avg_mfu_this_gpu = local_avg_mfu
        # self.global_average_mfu_per_gpu = (
        #    float(global_average_mfu) / dist.get_world_size()
        # )

    # ************************************
    # WRAP TRAINING WITH TIMING + FINAL REDUCTION
    # ************************************
    def train(self, *args, **kwargs):

        print_rank(
            f"ACCELERATE_USE_FSDP={os.environ.get('ACCELERATE_USE_FSDP')}, "
            f"FSDP_CPU_RAM_EFFICIENT_LOADING={os.environ.get('FSDP_CPU_RAM_EFFICIENT_LOADING')}",
        )

        from transformers.distributed.fsdp import is_fsdp_enabled
        from transformers.modeling_utils import is_local_dist_rank_0

        print_rank(
            f"is_fsdp_enabled={is_fsdp_enabled()}, is_local_dist_rank_0={is_local_dist_rank_0()}",
        )

        # reset counters in case reused
        self.total_tokens_this_gpu = 0
        self.total_tokens_global = 0

        start_time = time.time()
        output = super().train(*args, **kwargs)
        end_time = time.time()

        # all-reduce token counters
        self._finalize_token_counts()
        self._finalize_flop_counts()
        self._finalize_mfu_counts()
        elapsed = end_time - start_time
        try:
            self.total_training_seconds = float(elapsed)

            if "__get_item__" in self.train_dataset.__dict__:
                if self.train_dataset.__dict__["__get_item__"]:
                    self.average_get_item_time = sum(
                        self.train_dataset.__dict__["__get_item__"]
                    ) / len(self.train_dataset.__dict__["__get_item__"])
            if "collate_fn" in self.train_dataset.__dict__:
                if self.train_dataset.__dict__["collate_fn"]:
                    self.average_collate_fn_time = sum(
                        self.train_dataset.__dict__["collate_fn"]
                    ) / len(self.train_dataset.__dict__["collate_fn"])
        except Exception:
            print_rank(
                "WARNING! Could not calculate 'average_get_item_time' or 'average_collate_fn_time'..."
            )

        # print summary on rank 0
        print_rank("\n=== TOKEN / THROUGHPUT SUMMARY (PerformanceTrackingTrainer) ===")
        print_rank(f"Total tokens per GPU (ALL epochs): {self.total_tokens_this_gpu:,}")
        print(f"Total tokens GLOBAL (ALL epochs): {self.total_tokens_global:,}")

        if elapsed > 0:
            print_rank(0, f"Total training time (s): {elapsed:.2f}")
            print_rank(
                0, f"Tokens/sec per GPU: {self.total_tokens_this_gpu / elapsed:.2f}"
            )
            print_rank(
                0, f"Tokens/sec GLOBAL: {self.total_tokens_global / elapsed:.2f}"
            )
            print_rank(
                0,
                f"Average FLOPs over all GPUs: {self.global_average_flops / 1e12:.2f} TFLOPs/sec",
            )
            print_rank(
                0,
                f"Average FLOPs per GPU: {self.global_average_flops / dist.get_world_size() / 1e12:.2f} TFLOPs/sec/GPU",
            )
            print_rank(0, f"Average MFU per GPU: {self.global_average_mfu:.2f}%")
            print_rank(
                0, "==========================================================\n"
            )

            print_rank(f"Average TFLOPs: {self.avg_flops_this_gpu:.2f}")
            print_rank(f"Average MFU: {self.avg_mfu_this_gpu:.2f}")
        return output

    def write_summary(self, output_dir: str, gpu_stats: dict[str, list[float]]):
        try:
            # trainable_params, total_params, trainable_pct = count_parameters(model)
            trainable_params, total_params, trainable_pct = 0, 0, 0

            log_history = self.state.log_history
            print(f"log_history: {log_history}")
            avg_training_loss = avg_validation_loss = None
            avg_epoch_time_sec = avg_epoch_time_hours = None
            avg_step_time_sec = avg_step_time_hours = None

            total_training_time_secs = self.total_training_seconds
            max_elapsed_training_time_secs = total_training_time_secs
            if dist.is_initialized():
                elapsed_tensor = torch.tensor(total_training_time_secs, device="cuda")
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                max_elapsed_training_time_secs = elapsed_tensor.item()
            tokens_per_gpu_all_epochs = self.total_tokens_this_gpu

            tokens_global_all_epochs = self.total_tokens_global
            avg_gpu_flops = self.global_average_flops
            avg_gpu_mfu = self.global_average_mfu

            if self.args.max_steps:
                avg_step_time_sec = total_training_time_secs / self.args.max_steps
                avg_step_time_hours = avg_step_time_sec / 3600
            else:
                avg_epoch_time_sec = (
                    total_training_time_secs / self.args.num_train_epochs
                )
                avg_epoch_time_hours = avg_epoch_time_sec / 3600
                avg_step_time_sec = avg_epoch_time_sec / len(self.args.train_dataset)
                avg_step_time_hours = avg_step_time_sec / 3600

            effective_batch_size = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * dist.get_world_size()
            )

            samples_per_sec = (
                effective_batch_size / avg_step_time_sec if avg_step_time_sec else None
            )
            training_throughput_tokens_per_sec_per_gpu = (
                tokens_per_gpu_all_epochs / total_training_time_secs
                if total_training_time_secs
                else None
            )
            training_throughput_tokens_per_sec_global = (
                tokens_global_all_epochs / total_training_time_secs
                if max_elapsed_training_time_secs
                else None
            )
            avg_gpu_power_watts = (
                sum(gpu_stats["power"]) / len(gpu_stats["power"])
                if gpu_stats["power"]
                else None
            )
            tokens_per_sec_per_watt_global = (
                training_throughput_tokens_per_sec_global / avg_gpu_power_watts
                if training_throughput_tokens_per_sec_global and avg_gpu_power_watts
                else None
            )

            summary = {
                "nodes": int(os.environ.get("SLURM_NNODES", "1")),
                "num_gpus_per_node": int(os.environ.get("GPU_NODE", "1")),
                "total_gpus": dist.get_world_size(),
                "model": self.model.config.name_or_path,
                "dataset": os.environ.get("DATASET_PATH", "Unknown"),
                "framework": "accelerate",
                "parallelism_type": os.environ.get("PARALLELISM", "Unknown"),
                "batch_size": self.args.per_device_train_batch_size,
                "gradient_accumulation": self.args.gradient_accumulation_steps,
                "learning_rate": self.args.learning_rate,
            }

            metrics_summary = {
                "avg_gpu_memory_gb": sum(gpu_stats["mem"]) / len(gpu_stats["mem"])
                if gpu_stats["mem"]
                else None,
                "peak_gpu_memory_gb": max(gpu_stats["mem"])
                if gpu_stats["mem"]
                else None,
                "avg_gpu_utilization_percent": sum(gpu_stats["util"])
                / len(gpu_stats["util"])
                if gpu_stats["util"]
                else None,
                "peak_gpu_utilization_percent": max(gpu_stats["util"])
                if gpu_stats["util"]
                else None,
                "avg_gpu_power_watts": avg_gpu_power_watts,
                "peak_gpu_power_watts": max(gpu_stats["power"])
                if gpu_stats["power"]
                else None,
                "total_execution_time_hours": total_training_time_secs / 3600,
                "training_throughput_tokens_per_sec_global": training_throughput_tokens_per_sec_global,
                "training_throughput_tokens_per_sec_per_gpu": training_throughput_tokens_per_sec_per_gpu,
                "tokens_per_sec_per_watt_global": tokens_per_sec_per_watt_global,
                "samples_per_sec": samples_per_sec,
                "total_tokens_per_gpu_all_epochs": tokens_per_gpu_all_epochs,
                "total_tokens_global_all_epochs": tokens_global_all_epochs,
                "total_training_time_hours": total_training_time_secs / 3600,
                "avg_epoch_training_time_sec": avg_epoch_time_sec,
                "avg_epoch_training_time_hours": avg_epoch_time_hours,
                "avg_step_training_time_sec": avg_step_time_sec,
                "avg_step_training_time_hours": avg_step_time_hours,
                "avg_gpu_flops": avg_gpu_flops,
                "avg_gpu_mfu": avg_gpu_mfu,
                "training_loss": log_history[-1]["loss"]
                if log_history and "loss" in log_history[-1]
                else None,
                "validation_loss": avg_validation_loss,
            }
            save_summary_stats_json(
                summary={**summary, **metrics_summary},
                output_file=os.path.join(
                    output_dir, f"training_summary_{dist.get_rank()}.json"
                ),
            )

        except Exception as e:
            save_summary_stats_json(
                summary={
                    "error": str(e),
                },
                output_file=os.path.join(
                    output_dir, f"training_summary_{dist.get_rank()}.json"
                ),
            )
            print_rank("Fine-tuning failed to complete!")
            raise e


class FlopCounter:
    def __init__(self, model: torch.nn.Module):
        self.total_flops = 0
        self.last_flops = 0
        self.total_time_ms = 0
        self.step_count = 0
        self._start_time = None
        self._cfg = getattr(model, "config", None)
        self._num_params = sum(p.numel() for p in model.parameters())

        # Register both hooks on the model directly
        model.register_forward_pre_hook(self._pre_hook)
        model.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, inputs):
        self._start_time = time.perf_counter()

    def _post_hook(self, module, inputs, output):
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        # Extract input_ids from whatever form inputs arrive in
        input_ids = None
        if isinstance(inputs, tuple) and len(inputs) > 0:
            if isinstance(inputs[0], torch.Tensor):
                input_ids = inputs[0]
            elif isinstance(inputs[0], dict):
                input_ids = inputs[0].get("input_ids")

        if input_ids is None:
            return

        batch, seq = input_ids.shape[:2]

        param_flops = 6 * self._num_params * batch * seq
        attn_flops = self._attention_flops(batch, seq)
        step_flops = param_flops + attn_flops

        self.last_flops = step_flops
        self.total_flops += step_flops
        self.total_time_ms += elapsed_ms
        self.step_count += 1

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        tflops_per_sec = (step_flops / 1e12) / max(elapsed_ms / 1000, 1e-9)
        print(
            f"[flops] step={self.step_count} | "
            f"{step_flops / 1e12:.2f} TFLOPs | "
            f"{tflops_per_sec:.1f} TFLOPs/s | "
            f"{elapsed_ms:.1f}ms"
        )

    def _attention_flops(self, batch, seq):
        if self._cfg is None:
            return 0
        num_layers = getattr(self._cfg, "num_hidden_layers", 0)
        num_heads = getattr(self._cfg, "num_attention_heads", 0)
        hidden = getattr(self._cfg, "hidden_size", 0)
        head_dim = hidden // max(num_heads, 1)
        # 4 * B * H * S^2 * D per layer, times 3 for fwd + bwd
        return 4 * batch * num_heads * (seq**2) * head_dim * num_layers * 3

    def summary(self):
        if self.step_count == 0:
            return
        avg_tflops_per_sec = (self.total_flops / 1e12) / max(
            self.total_time_ms / 1000, 1e-9
        )
        print(
            f"\n[flops summary]\n"
            f"  steps:           {self.step_count}\n"
            f"  total TFLOPs:    {self.total_flops / 1e12:.2f}\n"
            f"  avg TFLOPs/s:    {avg_tflops_per_sec:.1f}\n"
            f"  avg ms/step:     {self.total_time_ms / self.step_count:.1f}\n"
        )


'''

FLOPS = []


def compute_tflops_per_step(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    elapsed_seconds: float,
    num_gpus: int = 1,
) -> float:
    """Megatron-style analytical FLOP counting."""
    B, S, L, H = batch_size, seq_len, num_layers, hidden_size

    # --- Per layer ---
    # QKV projection: 3 weight matrices of shape [H, H]
    qkv = 2 * B * S * 3 * H * H
    # Attention scores + weighted sum (quadratic term)
    attn = 2 * B * S * S * H  # QK^T and AV
    # Output projection
    out_proj = 2 * B * S * H * H
    # SwiGLU MLP: gate + up projections + down projection
    mlp = 2 * B * S * (2 * H * intermediate_size + intermediate_size * H)

    per_layer = qkv + attn + out_proj + mlp

    # --- Total forward ---
    forward = L * per_layer

    # --- Embedding / LM head (once, not per layer) ---
    lm_head = 2 * B * S * H * vocab_size

    total_forward = forward + lm_head

    # fwd + bwd (bwd ≈ 2× fwd)
    total_flops = 3 * total_forward

    tflops_per_gpu = total_flops / elapsed_seconds / num_gpus / 1e12
    return tflops_per_gpu


class MegatronFlopsCallback(TrainerCallback):
    def __init__(
        self,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        max_len: int,
        num_gpus=1,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_gpus = num_gpus
        self._t0 = None
        self.max_len = max_len

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0

        # grab live batch dims from the dataloader state
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        seq_len = self.max_len  # or however you track it

        tflops = compute_tflops_per_step(
            batch_size=batch_size,
            seq_len=seq_len,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            elapsed_seconds=elapsed,
            num_gpus=self.num_gpus,
        )

        FLOPS.append(tflops)
        print(f"Step {state.global_step:>6} | {tflops:.1f} TFLOP/s per GPU")'''

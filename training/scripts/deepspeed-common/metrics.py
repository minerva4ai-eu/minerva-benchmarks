import time

import torch
from transformers import TrainerCallback


class CustomMetricsCallback(TrainerCallback):
    """Callback to log additional metrics during training."""

    def __init__(self):
        self.step_start_time = None
        self.last_tokens = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens."""
        if logs is not None and state.global_step > 0:
            # Add GPU memory
            if torch.cuda.is_available():
                logs["gpu_mem_gb"] = torch.cuda.memory_allocated() / 1e9

            # Add throughput info
            if hasattr(kwargs["model"], "total_tokens_this_gpu"):
                trainer = kwargs.get("model")
                logs["cumulative_tokens"] = trainer.total_tokens_this_gpu

            # Print custom format
            if args.local_rank in [-1, 0]:  # Only on main process
                print(f"\n🔥 Step {state.global_step} Metrics:")
                for k, v in sorted(logs.items()):
                    print(f"   {k:30s}: {v}")

    def on_step_begin(self, args, state, control, **kwargs):
        """Track step start time."""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate per-step metrics."""
        step_time = time.time() - self.step_start_time if self.step_start_time else 0
        if state.global_step % args.logging_steps == 0:
            print(f"   ⏱️  Step time: {step_time:.3f}s")

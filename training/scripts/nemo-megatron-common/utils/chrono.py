from math import ceil
from time import time
import numpy as np
import torch


class TrainingChronometer:
    """
    Tracks training performance using CUDA Events for accurate GPU timing.
    
    Measures three phases per training step: Host-to-Device transfer,
    forward pass, and backward pass. Uses CUDA Events instead of CPU 
    wall-clock timers to accurately measure GPU execution time despite 
    the asynchronous nature of CUDA operations.
    """
    def __init__(self) -> None:
        self._reset_state()

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all timers and recorded events."""
        # CPU Timer
        self.saved_cpu_time = None
        self.start_training_time = None
        self.end_training_time = None

        # GPU Timer
        self.start_global = []
        self.end_global = []
        self.start_HtoD = []
        self.end_HtoD = []
        self.start_fwd = []
        self.end_fwd = []
        self.start_bwd = []
        self.end_bwd = []
    
    def cpu_timer(self, start: bool=True) -> None | float:
        """Start or stop a simple CPU wall-clock timer.

        Args:
            start: If True, records the current time as reference.
                   If False, returns elapsed time (in seconds) since last 
                   reference and updates the reference.
        """
        if start:
            self.saved_cpu_time = time()
        else:
            if self.saved_cpu_time is None:
                raise RuntimeError("Timer was not started. Call timer(start=True) first.")
            else:
                previous_time = self.saved_cpu_time
                self.saved_cpu_time = time()
                return self.saved_cpu_time - previous_time
        
    def track_cpu_training_time(self, start: bool=True) -> None:
        """Start or stop the CPU wall-clock timer of the full training run."""
        if start:
            self.start_training_time = time()
        else:
            self.end_training_time = time()

    def track_gpu_step_time(self, start:bool=True) -> None:
        """Record a CUDA Event marking the start or end of a global step."""
        gpu_timer = torch.cuda.Event(enable_timing=True)
        gpu_timer.record()
        if start:
            self.start_global.append(gpu_timer)
        else:
            self.end_global.append(gpu_timer)

    def track_gpu_HtoD_step_time(self, start: bool=True) -> None:
        """Record a CUDA Event marking the start or end of a Host-to-Device transfer."""
        gpu_timer = torch.cuda.Event(enable_timing=True)
        gpu_timer.record()
        if start:
            self.start_HtoD.append(gpu_timer)
        else:
            self.end_HtoD.append(gpu_timer)
                
    def track_gpu_fwd_step_time(self, start: bool=True) -> None:
        """Record a CUDA Event marking the start or end of a forward pass."""
        gpu_timer = torch.cuda.Event(enable_timing=True)
        gpu_timer.record()
        if start:
            self.start_fwd.append(gpu_timer)
        else:
            self.end_fwd.append(gpu_timer)
                
    def track_gpu_bwd_step_time(self, start: bool=True) -> None:
        """Record a CUDA Event marking the start or end of a backward pass."""
        gpu_timer = torch.cuda.Event(enable_timing=True)
        gpu_timer.record()
        if start:
            self.start_bwd.append(gpu_timer)
        else:
            self.end_bwd.append(gpu_timer)

    def print_percentile_summary(self, data: list, name: str) -> None:
        """Display distribution statistics for a list of timing measurements.
        
        Uses percentiles (min, p10, median, p90, max) to show the variability of the measurements.
        Requires at least 10 data points.
        """
        if len(data) >= 10:
            # Use percentiles since it works for any distribution
            p0 = np.min(data)
            p10 = np.percentile(data, 10)
            p50 = np.median(data)
            p90 = np.percentile(data, 90)
            p100 = np.max(data)

            print(f">>> {name}: min {p0:.3f} s | 10th percentile {p10:.3f} s | median {p50:.3f} s | 90th percentile {p90:.3f} s | max {p100:.3f} s")
        else:
            print(f">>> {name}: insufficient data (n={len(data)})")
                
    def display_training_results(self, total_batches_per_epoch: int, grad_acc: int, skip_steps: int=0) -> float:
        """Print a summary of training performance."""
        # Should be executed at the end of training. This ensures that GPU stopped working.
        torch.cuda.synchronize()

        # Compute time performance in seconds
        time_perf_global = [gpu_start_timer.elapsed_time(gpu_end_timer) / 1000 for (gpu_start_timer, gpu_end_timer) in zip(self.start_global, self.end_global)]
        time_perf_HtoD = [gpu_start_timer.elapsed_time(gpu_end_timer) / 1000 for (gpu_start_timer, gpu_end_timer) in zip(self.start_HtoD, self.end_HtoD)]
        time_perf_fwd = [gpu_start_timer.elapsed_time(gpu_end_timer) / 1000 for (gpu_start_timer, gpu_end_timer) in zip(self.start_fwd, self.end_fwd)]
        time_perf_bwd = [gpu_start_timer.elapsed_time(gpu_end_timer) / 1000 for (gpu_start_timer, gpu_end_timer) in zip(self.start_bwd, self.end_bwd)]

        # Skip the first n steps to avoid outliers that may be due to warm-up
        time_perf_global = time_perf_global[skip_steps:]
        time_perf_HtoD = time_perf_HtoD[skip_steps:]
        time_perf_fwd = time_perf_fwd[skip_steps:]
        time_perf_bwd = time_perf_bwd[skip_steps:]
        
        # Step statistics
        training_duration = self.end_training_time - self.start_training_time
        print(">>> Training complete in: " + str(training_duration))
        self.print_percentile_summary(time_perf_HtoD, "Host to Device performance time")
        self.print_percentile_summary(time_perf_fwd, "Forward pass performance time")
        self.print_percentile_summary(time_perf_bwd, "Backward pass performance time")

        # Total training time estimation
        if time_perf_global != []:
            time_perf_step = time_perf_global
        else:
            time_perf_step = [x + y + z for (x,y,z) in zip(time_perf_HtoD, time_perf_fwd, time_perf_bwd)]
        self.print_percentile_summary(time_perf_step, "Step performance time")
        print(f'>>> Number of weight updates per epoch: {ceil(total_batches_per_epoch / grad_acc)}')
        print(f'>>> Estimated training time of 1 epoch: {np.median(time_perf_step) * total_batches_per_epoch / 3600} h')

        return training_duration

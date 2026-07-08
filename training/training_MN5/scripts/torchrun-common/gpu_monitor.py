# finetune_llama8b.py
import threading
import time

import pynvml
from transformers import TrainerCallback


class GPUMonitorCallback(TrainerCallback):
    def __init__(self, n_gpus: int = 1):
        self.n_gpus = n_gpus
        self.mem, self.util, self.power = [], [], []

    def on_step_end(self, args, state, control, **kwargs):
        gpu = get_gpu_stats(n_gpus=self.n_gpus)
        self.mem.append(sum(gpu["mem_used"]) / len(gpu["mem_used"]))
        self.util.append(sum(gpu["util"]) / len(gpu["util"]))
        self.power.append(sum(gpu["power"]) / len(gpu["power"]))

    def summarize(self):
        return {
            "avg_gpu_memory_gb": sum(self.mem) / len(self.mem),
            "peak_gpu_memory_gb": max(self.mem),
            "avg_gpu_utilization_percent": sum(self.util) / len(self.util),
            "peak_gpu_utilization_percent": max(self.util),
            "avg_gpu_power_watts": sum(self.power) / len(self.power),
            "peak_gpu_power_watts": max(self.power),
        }


def start_gpu_monitor(interval_sec=5, n_gpus: int = 1):
    """Start a background thread that periodically samples GPU stats."""
    stats = {"mem": [], "util": [], "power": [], "timestamps": []}
    stop_flag = {"stop": False}

    def monitor():
        while not stop_flag["stop"]:
            gpu = get_gpu_stats(n_gpus=n_gpus)
            stats["mem"].append(sum(gpu["mem_used"]) / len(gpu["mem_used"]))
            stats["util"].append(sum(gpu["util"]) / len(gpu["util"]))
            stats["power"].append(sum(gpu["power"]) / len(gpu["power"]))
            stats["timestamps"].append(time.time())
            time.sleep(interval_sec)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stats, stop_flag


def get_gpu_stats(n_gpus: int = 1):
    """Return average and peak GPU memory, utilization, and power (GB, %, W)."""
    stats = {"mem_used": [], "util": [], "power": []}
    try:
        pynvml.nvmlInit()
        # n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = (
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            )  # milliwatts → watts

            stats["mem_used"].append(mem.used / 1024**3)
            stats["util"].append(util.gpu)
            stats["power"].append(power)

        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"Warning: Could not collect GPU stats: {e}")
        # n_gpus = torch.cuda.device_count()
        stats["mem_used"] = [0] * n_gpus
        stats["util"] = [0] * n_gpus
        stats["power"] = [0] * n_gpus

    return stats

import threading
import time
from amdsmi import (
    amdsmi_init,
    amdsmi_shut_down,
    amdsmi_get_processor_handles,
    amdsmi_get_gpu_memory_usage,
    amdsmi_get_gpu_activity,
    amdsmi_get_power_info,
    AmdSmiMemoryType,
)
from transformers import TrainerCallback

# Init une seule fois au chargement du module
amdsmi_init()
_handles = amdsmi_get_processor_handles()


def _parse_power(power_info):
    """MI250X uses average_socket_power, MI300A uses current_socket_power."""
    for field in ("average_socket_power", "current_socket_power"):
        try:
            return float(power_info.get(field))
        except (ValueError, TypeError):
            continue
    return 0.0


def get_gpu_stats(n_gpus: int = 1):
    stats = {"mem_used": [], "util": [], "power": []}
    try:
        n = min(n_gpus, len(_handles))
        for i in range(n):
            mem_gb = amdsmi_get_gpu_memory_usage(_handles[i], AmdSmiMemoryType.VRAM) / 1024**3
            util   = amdsmi_get_gpu_activity(_handles[i]).get("gfx_activity", 0)
            power  = _parse_power(amdsmi_get_power_info(_handles[i]))
            stats["mem_used"].append(mem_gb)
            stats["util"].append(util)
            stats["power"].append(power)
    except Exception as e:
        print(f"Warning: Could not collect GPU stats: {e}")
        stats["mem_used"] = [0.0] * n_gpus
        stats["util"]     = [0]   * n_gpus
        stats["power"]    = [0.0] * n_gpus
    return stats


class GPUMonitorCallback(TrainerCallback):
    def __init__(self, n_gpus: int = 1):
        self.n_gpus = n_gpus
        self.mem, self.util, self.power = [], [], []

    def on_step_end(self, args, state, control, **kwargs):
        gpu = get_gpu_stats(n_gpus=self.n_gpus)
        if not gpu["mem_used"]:
            return
        self.mem.append(sum(gpu["mem_used"]) / len(gpu["mem_used"]))
        self.util.append(sum(gpu["util"])    / len(gpu["util"]))
        self.power.append(sum(gpu["power"])  / len(gpu["power"]))

    def summarize(self):
        return {
            "avg_gpu_memory_gb":            sum(self.mem)   / len(self.mem),
            "peak_gpu_memory_gb":           max(self.mem),
            "avg_gpu_utilization_percent":  sum(self.util)  / len(self.util),
            "peak_gpu_utilization_percent": max(self.util),
            "avg_gpu_power_watts":          sum(self.power) / len(self.power),
            "peak_gpu_power_watts":         max(self.power),
        }


def start_gpu_monitor(interval_sec=5, n_gpus: int = 1):
    """Start a background thread that periodically samples GPU stats."""
    stats = {"mem": [], "util": [], "power": [], "timestamps": []}
    stop_flag = {"stop": False}

    def monitor():
        while not stop_flag["stop"]:
            gpu = get_gpu_stats(n_gpus=n_gpus)
            if not gpu["mem_used"]:
                time.sleep(interval_sec)
                continue
            stats["mem"].append(sum(gpu["mem_used"]) / len(gpu["mem_used"]))
            stats["util"].append(sum(gpu["util"])    / len(gpu["util"]))
            stats["power"].append(sum(gpu["power"])  / len(gpu["power"]))
            stats["timestamps"].append(time.time())
            time.sleep(interval_sec)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stats, stop_flag
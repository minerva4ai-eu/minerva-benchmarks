import time
import sys
import signal
import json
from amdsmi import *

# Usage: python gpu_summary_monitor.py <summary_file.json> [poll_interval]

summary_file = sys.argv[1]
poll_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
fixed_mem = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  # valeur fixe en % de mémoire utilisée

amdsmi_init()
handles = amdsmi_get_processor_handles()
device_count = len(handles)

mem_sum = [0.0] * device_count
power_sum = [0.0] * device_count
mem_peak = [0.0] * device_count
power_peak = [0.0] * device_count
count = [0] * device_count
stop = False

def stop_monitor(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, stop_monitor)
signal.signal(signal.SIGTERM, stop_monitor)

print(f"[gpu_monitor] Monitoring {device_count} GPUs every {poll_interval}s. Writing summary to {summary_file}", flush=True)

# Touch file early so it exists even if killed early
open(summary_file, "w").close()

try:
    while not stop:
        for i in range(device_count):
            info = amdsmi_get_power_info(handles[i])
            mem_used = fixed_mem * 131072 # MB
            power = info.get('current_socket_power') # W

            mem_sum[i] += mem_used
            power_sum[i] += power
            count[i] += 1

            mem_peak[i] = max(mem_peak[i], mem_used)
            power_peak[i] = max(power_peak[i], power)

        time.sleep(poll_interval)

finally:
    amdsmi_shut_down()

    gpu_stats = []
    total_mem_avg = total_power_avg = total_mem_peak = total_power_peak = 0.0

    for i in range(device_count):
        avg_mem = mem_sum[i] / count[i] if count[i] > 0 else 0
        avg_power = power_sum[i] / count[i] if count[i] > 0 else 0
        gpu_stats.append({
            "id": i,
            "avg_memory_mb": round(avg_mem, 2),
            "peak_memory_mb": round(mem_peak[i], 2),
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(power_peak[i], 2)
        })
        total_mem_avg += avg_mem
        total_power_avg += avg_power
        total_mem_peak += mem_peak[i]
        total_power_peak += power_peak[i]

    summary = {
        "interval": poll_interval,
        "GPUs": gpu_stats,
        "total": {
            "avg_memory_mb": round(total_mem_avg / device_count, 2),
            "peak_memory_mb": round(total_mem_peak / device_count, 2),
            "avg_power_w": round(total_power_avg / device_count, 2),
            "peak_power_w": round(total_power_peak / device_count, 2)
        }
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[gpu_monitor] Summary written to {summary_file}", flush=True)
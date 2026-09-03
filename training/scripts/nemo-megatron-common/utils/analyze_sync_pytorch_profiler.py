import json
import numpy as np


def get_gpu_step_info_from_trace(trace: dict) -> dict:
    """Extract ProfilerStep annotations from a PyTorch profiler trace."""
    
    # Filter all events called `ProfilerStep`
    gpu_step_annotations = [
        e for e in trace["traceEvents"]
        if "ProfilerStep" in e.get("name", "")
        and e.get("cat", "") == "gpu_user_annotation"
    ]
    
    # Filter keys
    keys_to_keep = {'name', 'ts', 'args'}
    gpu_step_annotations = [
        {k: v for k, v in event.items() if k in keys_to_keep}
        for event in gpu_step_annotations
    ]
    
    # Sort by annotation order
    gpu_step_annotations.sort(key=lambda x: x['name'])
    
    # Compute step duration on GPU
    for i in range(len(gpu_step_annotations)-1):
        gpu_step_annotations[i]['dur'] = gpu_step_annotations[i+1]['ts'] - gpu_step_annotations[i]['ts']

    # Extract train_step from name (e.g., "ProfilerStep#17" -> 17)
    return {int(event['name'].split('#')[1]): event for event in gpu_step_annotations}

def categorize_cuda_event(event: dict) -> str:
    """Categorize a GPU-side profiler event into one of five buckets:
    
    - "cpu_gpu_transfer": PCIe data movement between CPU and GPU (HtoD, DtoH)
    - "compute":          CUDA kernels that are not NCCL kernels
    - "comm_overhead":    Communication overhead on default stream (no overlap)
    - "communication":    NCCL kernels
    - "other":            GPU-to-GPU memcpy, memset, annotations, etc.

    Source: https://github.com/pytorch/kineto/blob/22cb9f15e528d116729746d569dec466abea9234/libkineto/include/ActivityType.h#L19
    """
    cat = event.get("cat", "")
    name = event.get("name", "").lower()

    if cat == "gpu_memcpy":
        if "dtoh" in name or "htod" in name:
            return "cpu_gpu_transfer"
        else:
            return "other"
    elif cat == "kernel":
        if ("split_with_sizes_copy_out_contiguous") in name or ("chunk_cat_cuda_kernel") in name:
            return "comm_overhead"
        elif ("nccl" in name):
            return "communication"
        else:
            return "compute"
    else:
        return "other"

def wire_bytes_per_gpu_from_pytorch_profiler(coll_op: str, count: int, dtype: str, nranks: int) -> float:
    """Estimate the number of bytes transferred per GPU for a given NCCL collective.

    Adapted for PyTorch profiler output.

    Source: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth
    """
    N = nranks

    # Factor to convert count to total data involved in the collective
    count_factor = {
        'allreduce': 1,
        '_allgather_base': N,
        '_reduce_scatter_base': N,
    }

    # Convert element counts to bytes
    dtype_size = {
        'BFloat16': 2,
    }

    # Ratio between size of the data involved in the collective and the average volume of data transfer per GPU
    bus_bandwidth_factor = {
        'allreduce': 2 * (N - 1) / N,
        '_allgather_base': (N - 1) / N,
        '_reduce_scatter_base': (N - 1) / N,
    }

    return count * count_factor[coll_op] * dtype_size[dtype] * bus_bandwidth_factor[coll_op]

def parse_sequential_trace(profiler_file: str='profile/xp/jzxh117_870000.1773996581193469969.pt.trace.json') -> tuple[dict, dict]:
    """Parse a PyTorch profiler trace JSON and extract per-step GPU events.

    Reads the Chrome Trace format exported by torch.profiler, identifies training steps via gpu_user_annotation events, 
    assigns each GPU event to its corresponding step, and computes the idle latency between consecutive GPU events.
    Assumes CUDA_LAUNCH_BLOCKING=1 (all GPU operations are serialized).
    Collects all GPU events into a single timeline per step.
    """
    ## LOAD PROFILE
    with open(profiler_file, "r") as f:
        trace = json.load(f)
    
    
    ## GET STEP DURATION
    gpu_step_annotations = get_gpu_step_info_from_trace(trace)
    train_steps = list(gpu_step_annotations.keys())
    
    
    ## GET EVENTS PER STEP
    events_per_step = {train_step: [] for train_step in train_steps}
    ts_per_step = {train_step: event['ts'] for train_step, event in gpu_step_annotations.items()}
    
    # Categorize stream events per STEP
    for event in trace["traceEvents"]:
        if event.get("args", {}).get("stream", -1) != -1 and event.get("cat", "") in ["kernel", "gpu_memcpy", "gpu_memset"]:
            step = max((train_step for train_step, ts_step in ts_per_step.items() if event['ts'] >= ts_step), default=None)
            if step != None:
                events_per_step[step].append(event)

    # Sort by events order
    for step_event in events_per_step.values():
        step_event.sort(key=lambda x: x['ts'])
    
    # Compute latency between events on GPU
    for i, step in enumerate(train_steps[:-1]):  # skip the last training step
        next_step = train_steps[i + 1]
        step_event = events_per_step[step]
        for j in range(len(step_event)-1):
            step_event[j]['latency'] = step_event[j+1]['ts'] - (step_event[j]['ts'] + step_event[j]['dur'])
        step_event[-1]['latency'] = gpu_step_annotations[next_step]['ts'] - (step_event[-1]['ts'] + step_event[-1]['dur'])
    
    
    ## PROCESSING
    # Remove last step
    last_step = train_steps[-1]
    del gpu_step_annotations[last_step]
    del events_per_step[last_step]
    
    # Convert microseconds to millisecond
    for event in gpu_step_annotations.values():
        event['ts'] = event['ts'] / 1000
        event['dur'] = event['dur'] / 1000
    
    for step_event in events_per_step.values():
        for event in step_event:
            event['ts'] = event['ts'] / 1000
            event['dur'] = event['dur'] / 1000
            event['latency'] = event['latency'] / 1000

    return gpu_step_annotations, events_per_step

def analyze_sequential_step_breakdown(gpu_step_annotations: dict,
                                      events_per_step: dict,
                                      train_step: int = None) -> dict:
    """Compute time breakdown for a training step.

    Sums event durations by category and total idle time between events.
    Assumes CUDA_LAUNCH_BLOCKING=1 (all GPU operations are serialized).
    
    Categories:
        cpu_bound:        Idle time on GPU waiting for CPU to launch next kernel.
                          Captures Python/PyTorch overhead, kernel launch latency.
                          High values suggest CPU bottleneck.
        cpu_gpu_transfer: Data movement over PCIe (HtoD, DtoH).
        compute:          CUDA kernels excluding NCCL.
        comm_overhead:    Tensor processing for NCCL collectives on default stream.
                          Cannot overlap with compute.
        communication:    NCCL collective operations (AllReduce, AllGather, etc.).
        other:            GPU-to-GPU memcpy, memset, etc.
    """
    train_steps = list(gpu_step_annotations.keys())

    # Select step with median duration if not specified
    if train_step is None:
        durations = [gpu_step_annotations[s]['dur'] for s in train_steps]
        median_dur = np.median(durations)
        idx = int(np.argmin([abs(d - median_dur) for d in durations]))
        train_step = train_steps[idx]  # convert index to actual train_step key
    
    # Categorize events
    events_by_category = {
        "cpu_gpu_transfer": [],
        "compute": [],
        "comm_overhead": [],
        "communication": [],
        "other": [],
    }
    for event in events_per_step[train_step]:
        events_by_category[categorize_cuda_event(event)].append(event)

    print(f"{gpu_step_annotations[train_step]['name']}:")
    print(f"  CPU-bound:              {sum(event['latency'] for event in events_per_step[train_step]):8.1f} ms")
    print(f"  CPU-GPU transfer:       {sum(event['dur'] for event in events_by_category['cpu_gpu_transfer']):8.1f} ms")
    print(f"  Compute:                {sum(event['dur'] for event in events_by_category['compute']):8.1f} ms")
    print(f"  Communication overhead: {sum(event['dur'] for event in events_by_category['comm_overhead']):8.1f} ms")
    print(f"  Communication:          {sum(event['dur'] for event in events_by_category['communication']):8.1f} ms")
    print(f"  Other:                  {sum(event['dur'] for event in events_by_category['other']):8.1f} ms")
    print(f"  STEP TOTAL:             {gpu_step_annotations[train_step]['dur']:8.1f} ms")

    return {
        "step": gpu_step_annotations[train_step]['name'],
        "cpu_bound": sum(event["latency"] for event in events_per_step[train_step]),
        "cpu_gpu_transfer": sum(event["dur"] for event in events_by_category["cpu_gpu_transfer"]),
        "compute": sum(event["dur"] for event in events_by_category["compute"]),
        "comm_overhead": sum(event["dur"] for event in events_by_category["comm_overhead"]),
        "communication": sum(event["dur"] for event in events_by_category["communication"]),
        "other": sum(event["dur"] for event in events_by_category["other"]),
        "step_duration": gpu_step_annotations[train_step]["dur"],
    }
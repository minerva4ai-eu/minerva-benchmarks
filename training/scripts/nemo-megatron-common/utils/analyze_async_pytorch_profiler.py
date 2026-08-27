import json
import numpy as np

from .analyze_sync_pytorch_profiler import get_gpu_step_info_from_trace, categorize_cuda_event


def compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> tuple[float, float, float]:
    """Intersection between two intervals [a_start, a_end) and [b_start, b_end)."""
    inter_end = min(a_end, b_end)
    inter_start = max(a_start, b_start)
    o = inter_end - inter_start 
    return max(0.0, o), inter_start, inter_end

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals into non-overlapping ones."""
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged

def compute_nccl_during_compute(compute_events: list, nccl_events: list) -> float:
    """Compute total NCCL time that overlaps with compute kernels (in ms).

    Merges overlapping NCCL intervals to avoid double-counting when multiple NCCL kernels run simultaneously.
    """

    # Find all nccl events that overlap with compute events
    nccl_during_compute = []
    for comp_ev in compute_events:
        comp_start = comp_ev["ts"]
        comp_end = comp_start + comp_ev["dur"]

        for nccl_ev in nccl_events:
            nccl_start = nccl_ev["ts"]
            nccl_end = nccl_start + nccl_ev["dur"]

            overlap, inter_start, inter_end = compute_overlap(comp_start, comp_end, nccl_start, nccl_end)
            if overlap > 0:
                nccl_during_compute.append((inter_start, inter_end))

    if not nccl_during_compute:
        return 0.0

    # Merge overlapping intervals
    merged = merge_intervals(nccl_during_compute)

    return sum(end - start for start, end in merged)

def classify_default_stream_idle_gaps(compute_events: list, nccl_events: list, step_start: float, step_end: float) -> dict:
    """For each idle gap on the compute stream, determine how much is communication-bound vs CPU-bound.
 
    Args:
        compute_events: sorted list of events on the compute stream (stream 7)
        nccl_events:    sorted list of NCCL events on other streams
        step_start:     step start timestamp (ms)
        step_end:       step end timestamp (ms)
    """
    ## Build list of idle gaps on the compute stream
    gaps = []
 
    # Gap before first event
    if compute_events[0]["ts"] > step_start:
        gaps.append((step_start, compute_events[0]["ts"]))
 
    # Gaps between consecutive events
    for i in range(len(compute_events) - 1):
        gap_start = compute_events[i]["ts"] + compute_events[i]["dur"]
        gap_end = compute_events[i + 1]["ts"]
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end))
 
    # Gap after last event
    last_end = compute_events[-1]["ts"] + compute_events[-1]["dur"]
    if last_end < step_end:
        gaps.append((last_end, step_end))


    ## For each gap, compute how much is covered by NCCL on other streams
    total_comm_bound = 0.0
    total_cpu_bound = 0.0
    gap_details = []
 
    for gap_start, gap_end in gaps:
        gap_dur = gap_end - gap_start
        if gap_dur <= 0:
            continue
 
        # Filter NCCL events that is within this gap
        nccl_in_gap = []
        for nccl_ev in nccl_events:
            nccl_start = nccl_ev["ts"]
            nccl_end = nccl_ev["ts"] + nccl_ev["dur"]
            o, inter_start, inter_end = compute_overlap(gap_start, gap_end, nccl_start, nccl_end)
            if o > 0:
                nccl_in_gap.append((inter_start, inter_end))
 
        # Merge overlapping NCCL intervals within this gap
        merged = merge_intervals(nccl_in_gap)
 
        comm_time = sum(end - start for start, end in merged)
        cpu_time = gap_dur - comm_time
 
        total_comm_bound += comm_time
        total_cpu_bound += cpu_time
 
        gap_details.append({
            "start": gap_start,
            "end": gap_end,
            "duration": gap_dur,
            "comm_bound": comm_time,
            "cpu_bound": cpu_time,
        })
 
    return {
        "comm_bound": total_comm_bound,
        "cpu_bound": total_cpu_bound,
        "total_idle": total_comm_bound + total_cpu_bound,
        "gaps": gap_details,
    }

def parse_overlap_trace(profiler_file: str='profile/xp/jzxh018_3913107.1774026182903860725.pt.trace.json') -> tuple[dict, dict, dict]:
    """Parse a PyTorch profiler trace JSON and extract per-step GPU events.

    Reads the Chrome Trace format exported by torch.profiler, identifies training steps via gpu_user_annotation events, 
    assigns each GPU event to its corresponding step, and computes the idle latency between consecutive GPU events.
    Assumes asynchronous execution (default CUDA behavior).
    Separates compute stream events from NCCL communication events to enable overlap analysis.
    """
    ## LOAD PROFILE
    with open(profiler_file, "r") as f:
        trace = json.load(f)
    

    ## GET STEP DURATION
    gpu_step_annotations = get_gpu_step_info_from_trace(trace)
    train_steps = list(gpu_step_annotations.keys())


    ## GET EVENTS PER STEP
    ts_per_step = {train_step: event['ts'] for train_step, event in gpu_step_annotations.items()}

    # Categorize compute stream events per STEP and NCCL events per STEP
    default_stream_event_per_step = {step: [] for step in train_steps}
    communication_stream_event_per_step = {step: [] for step in train_steps}
    for event in trace["traceEvents"]:
        if event.get("args", {}).get("stream", -1) == 7 and event.get("cat", "") in ["kernel", "gpu_memcpy", "gpu_memset"]:
            step = max((train_step for train_step, ts_step in ts_per_step.items() if event['ts'] >= ts_step), default=None)
            if step != None:
                default_stream_event_per_step[step].append(event)
        elif event.get("args", {}).get("stream", -1) != -1 and event.get("cat", "") == "kernel" and "nccl" in event.get("name", ""):
            step = max((train_step for train_step, ts_step in ts_per_step.items() if event['ts'] >= ts_step), default=None)
            if step != None:
                communication_stream_event_per_step[step].append(event)

    # Sort by events order
    for step in train_steps:
        default_stream_event_per_step[step].sort(key=lambda x: x['ts'])
        communication_stream_event_per_step[step].sort(key=lambda x: x['ts'])

    # Compute latency between events on GPU
    for i, step in enumerate(train_steps[:-1]):  # skip the last training step
        next_step = train_steps[i + 1]

        step_default_event = default_stream_event_per_step[step]
        next_step_default_event = default_stream_event_per_step[next_step]
        for j in range(len(step_default_event) - 1):
            step_default_event[j]['latency'] = step_default_event[j + 1]['ts'] - (step_default_event[j]['ts'] + step_default_event[j]['dur'])
        step_default_event[-1]['latency'] = next_step_default_event[0]['ts'] - (step_default_event[-1]['ts'] + step_default_event[-1]['dur'])

        step_comm_event = communication_stream_event_per_step[step]
        next_step_comm_event = communication_stream_event_per_step[next_step]
        for j in range(len(step_comm_event) - 1):
            step_comm_event[j]['latency'] = step_comm_event[j + 1]['ts'] - (step_comm_event[j]['ts'] + step_comm_event[j]['dur'])
        step_comm_event[-1]['latency'] = next_step_comm_event[0]['ts'] - (step_comm_event[-1]['ts'] + step_comm_event[-1]['dur'])


    ## PROCESSING
    # Remove last step
    last_step = train_steps[-1]
    del gpu_step_annotations[last_step]
    del default_stream_event_per_step[last_step]
    del communication_stream_event_per_step[last_step]

    # Convert microseconds to millisecond
    for event in gpu_step_annotations.values():
        event['ts'] = event['ts'] / 1000
        event['dur'] = event['dur'] / 1000

    for step_event in list(default_stream_event_per_step.values()) + list(communication_stream_event_per_step.values()):
        for event in step_event:
            event['ts'] = event['ts'] / 1000
            event['dur'] = event['dur'] / 1000
            event['latency'] = event['latency'] / 1000

    return gpu_step_annotations, default_stream_event_per_step, communication_stream_event_per_step

def get_total_comm_step(communication_stream_event_per_step: dict, train_step: int) -> float:
    """Sum of all NCCL kernel durations for a given step (in ms)."""
    return sum([event['dur'] for event in communication_stream_event_per_step[train_step]])

def analyze_overlap_step_breakdown(gpu_step_annotations: dict,
                                   default_stream_event_per_step: dict,
                                   communication_stream_event_per_step: dict,
                                   train_step: int=None) -> dict:
    """Compute time breakdown for a training step with overlap analysis.

    Sums event durations by category and classifies idle gaps on the default stream
    as either communication-bound (hidden by NCCL on other streams) or CPU-bound.
    Assumes asynchronous execution (default CUDA behavior).

    Categories:
        cpu_bound:        Idle time on default stream not overlapped by NCCL.
                          Captures Python/PyTorch overhead, kernel launch latency.
                          High values suggest CPU bottleneck.
        cpu_gpu_transfer: Data movement over PCIe (HtoD, DtoH).
        compute:          CUDA kernels excluding NCCL (can overlap with communication).
        comm_overhead:    Tensor processing for NCCL collectives on default stream.
                          Cannot overlap with compute.
        comm_bound:       Idle time on default stream overlapped by NCCL.
        other:            GPU-to-GPU memcpy, memset, etc.
    """
    train_steps = list(gpu_step_annotations.keys())

    # Select step with median duration if not specified
    if train_step is None:
        durations = [gpu_step_annotations[s]['dur'] for s in train_steps]
        median_dur = np.median(durations)
        idx = int(np.argmin([abs(d - median_dur) for d in durations]))
        train_step = train_steps[idx]  # convert index to actual train_step key

    # Classify idle gaps in default stream
    step_start = gpu_step_annotations[train_step]["ts"]
    step_end = step_start + gpu_step_annotations[train_step]["dur"]
    
    idle_classification = classify_default_stream_idle_gaps(
        compute_events=default_stream_event_per_step[train_step],
        nccl_events=communication_stream_event_per_step[train_step],
        step_start=step_start,
        step_end=step_end,
    )

    # Categorize default stream events
    events_by_category = {
        "cpu_gpu_transfer": [],
        "compute": [],
        "comm_overhead": [],
        "other": [],
    }
    for event in default_stream_event_per_step[train_step]:
        try:
            events_by_category[categorize_cuda_event(event)].append(event)
        except:
            print(f'This event type was in default stream: {categorize_cuda_event(event)}')

    print(f"{gpu_step_annotations[train_step]['name']}:")
    print(f"  CPU-bound:                              {idle_classification['cpu_bound']:8.1f} ms")
    print(f"  CPU-GPU transfer:                       {sum(event['dur'] for event in events_by_category['cpu_gpu_transfer']):8.1f} ms")
    print(f"  Compute [1]:                            {sum(event['dur'] for event in events_by_category['compute']):8.1f} ms")
    print(f"  Communication overhead [1]:             {sum(event['dur'] for event in events_by_category['comm_overhead']):8.1f} ms")
    print(f"    [1] -> Communication Overlapped [2]: ({compute_nccl_during_compute(default_stream_event_per_step[train_step], communication_stream_event_per_step[train_step]):8.1f} ms)")
    print(f"  Communication-bound [2]:                {idle_classification['comm_bound']:8.1f} ms")
    print(f"    [2] -> Sum of NCCL Kernels:          ({get_total_comm_step(communication_stream_event_per_step, train_step):8.1f} ms)")
    print(f"  Other:                                  {sum(event['dur'] for event in events_by_category['other']):8.1f} ms")
    print(f"  STEP TOTAL:                             {gpu_step_annotations[train_step]['dur']:8.1f} ms")

    return {
        "step": gpu_step_annotations[train_step]["name"],
        "cpu_bound": idle_classification["cpu_bound"],
        "cpu_gpu_transfer": sum(event["dur"] for event in events_by_category["cpu_gpu_transfer"]),
        "compute": sum(event["dur"] for event in events_by_category["compute"]),
        "comm_overhead": sum(event["dur"] for event in events_by_category["comm_overhead"]),
        "comm_overlap": compute_nccl_during_compute(default_stream_event_per_step[train_step], communication_stream_event_per_step[train_step]),
        "comm_bound": idle_classification["comm_bound"],
        "total_communication": get_total_comm_step(communication_stream_event_per_step, train_step),
        "other": sum(event["dur"] for event in events_by_category["other"]),
        "step_duration": gpu_step_annotations[train_step]["dur"],
    }

def classify_pt_events_by_fabric(gpu_step_annotations: dict,
                                 communication_stream_event_per_step: dict,
                                 comms_profile: dict,
                                 train_step: int=None) -> dict:
    """Classify PyTorch profiler NCCL events by network fabric and compute total duration per fabric.

    Matches PyTorch profiler NCCL events to NCCL log events based on collective operation, element count, datatype, and group size.
    Then classifies each event by the fabric (NVLink, Network, etc.) used by its corresponding NCCL communicator.
    """
    train_steps = list(gpu_step_annotations.keys())

    # Select step with median duration if not specified
    if train_step is None:
        durations = [gpu_step_annotations[s]['dur'] for s in train_steps]
        median_dur = np.median(durations)
        idx = int(np.argmin([abs(d - median_dur) for d in durations]))
        train_step = train_steps[idx]  # convert index to actual train_step key

    coll_map = {
        'AllReduce': 'allreduce',
        'ReduceScatter': '_reduce_scatter_base',
        'AllGather': '_allgather_base'
    }
    dtype_map = {
        9: 'BFloat16'
    }

    # Filter NCCL events for target step
    step_mask = [step == train_step for step in comms_profile['train_step']]
    nccl_coll_ops = [x for x, keep in zip(comms_profile['coll_operation'], step_mask) if keep]
    nccl_nelems = [x for x, keep in zip(comms_profile['data_volume'], step_mask) if keep]
    nccl_dtypes = [x for x, keep in zip(comms_profile['datatype'], step_mask) if keep]
    nccl_nranks = [x for x, keep in zip(comms_profile['nranks'], step_mask) if keep]
    nccl_fabrics = [x for x, keep in zip(comms_profile['fab'], step_mask) if keep]
    nccl_comm_per_gpu = [x for x, keep in zip(comms_profile['comm_per_gpu'], step_mask) if keep]

    # Extract PyTorch event info
    pt_events = communication_stream_event_per_step[train_step]
    pt_coll_ops = [e['args']['Collective name'] for e in pt_events]
    pt_in_nelems = [e['args']['In msg nelems'] for e in pt_events]
    pt_out_nelems = [e['args']['Out msg nelems'] for e in pt_events]
    pt_nelems = [min(i, o) for i, o in zip(pt_in_nelems, pt_out_nelems)]
    pt_dtypes = [e['args']['dtype'] for e in pt_events]
    pt_nranks = [e['args']['Group size'] for e in pt_events]

    def events_match(pt_idx, nccl_idx):
        return (
            pt_coll_ops[pt_idx] == coll_map[nccl_coll_ops[nccl_idx]]
            and pt_nelems[pt_idx] == nccl_nelems[nccl_idx]
            and pt_dtypes[pt_idx] == dtype_map[nccl_dtypes[nccl_idx]]
            and pt_nranks[pt_idx] == nccl_nranks[nccl_idx]
        )

    # Match PT -> NCCL
    pt_to_nccl = [None] * len(pt_events)
    unmatched_nccl = []
    pt_idx = 0

    for nccl_idx in range(len(nccl_coll_ops)):
        for skipped in unmatched_nccl:
            if pt_idx >= len(pt_events):
                break
            if events_match(pt_idx, skipped):
                pt_to_nccl[pt_idx] = skipped
                unmatched_nccl.remove(skipped)
                pt_idx += 1

        if pt_idx >= len(pt_events):
            break
        if events_match(pt_idx, nccl_idx):
            pt_to_nccl[pt_idx] = nccl_idx
            pt_idx += 1
        else:
            unmatched_nccl.append(nccl_idx)

    # Classify by fabric and sum durations
    metrics_by_fabric = {}

    for pt_idx, pt_event in enumerate(pt_events):
        nccl_idx = pt_to_nccl[pt_idx]

        fab = nccl_fabrics[nccl_idx]
        bytes_per_gpu = nccl_comm_per_gpu[nccl_idx]

        # Initialize fabric entry
        if fab not in metrics_by_fabric:
            metrics_by_fabric[fab] = {'duration_ms': 0.0, 'bytes': 0.0, 'bandwidth_gbps': 0.0}

        # Accumulate duration and bytes
        metrics_by_fabric[fab]['duration_ms'] += pt_event['dur']
        metrics_by_fabric[fab]['bytes'] += bytes_per_gpu

    # Compute bandwidth for each fabric
    for fab, metrics in metrics_by_fabric.items():
        if metrics['duration_ms'] > 0:
            # bandwidth = bytes / time -> GB/s = bytes / (ms * 1e6)
            metrics['bandwidth_gbps'] = metrics['bytes'] / (metrics['duration_ms'] * 1e6)

    return metrics_by_fabric

def plot_sequential_vs_overlap(breakdown_seq: dict, breakdown_ovl: dict) -> None:
    """Compare sequential and overlap execution as grouped bar chart by category."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Run: pip install plotly")
        return None

    categories = [
        ("cpu_bound", "CPU-bound"),
        ("cpu_gpu_transfer", "CPU-GPU transfer"),
        ("compute", "Compute"),
        ("comm_overhead", "Comm overhead"),
        ("communication", "Comm-bound"),
        ("comm_bound", "Comm-bound"),
        ("other", "Other"),
    ]

    # Filter categories that have data in at least one breakdown
    filtered = [(key, label) for key, label in categories
                if breakdown_seq.get(key, 0) > 0 or breakdown_ovl.get(key, 0) > 0]

    keys = [k for k, _ in filtered]
    labels = [l for _, l in filtered]
    unique_labels = list(dict.fromkeys(labels))

    seq_values = [breakdown_seq.get(k, 0) for k in keys if breakdown_seq.get(k, 0) > 0]
    ovl_values = [breakdown_ovl.get(k, 0) for k in keys if breakdown_ovl.get(k, 0) > 0]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"Sequential ({breakdown_seq['step_duration']:.1f} ms)",
        x=unique_labels,
        y=seq_values,
        marker_color="#636EFA",
        text=[f"{v:.1f}" for v in seq_values],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name=f"Overlap ({breakdown_ovl['step_duration']:.1f} ms)",
        x=unique_labels,
        y=ovl_values,
        marker_color="#00CC96",
        text=[f"{v:.1f}" for v in ovl_values],
        textposition='outside',
    ))

    fig.update_layout(
        barmode='group',
        title="Sequential vs Overlap Execution",
        xaxis_title="Category",
        yaxis=dict(
            title="Time (ms)",
            range=[0,max(seq_values+ovl_values)*1.1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        width=600,
        height=450,
    )

    fig.show()

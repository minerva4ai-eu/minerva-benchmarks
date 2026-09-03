from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def wire_bytes_per_gpu(coll_op: str, nccl_count: int, datatype: int, nranks: int):
    """
    Estimate the number of bytes transferred per GPU for a given NCCL collective.

    The computation follows the same conventions as NCCL-tests bandwidth calculations.

    Source: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth
    Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#send-and-receive-counts
    Source: https://github.com/NVIDIA/nccl/blob/master/contrib/nccl_ep/python/nccl_ep/nccl_wrapper.py#L56
    """
    N = nranks
    
    # `count` in NCCL logs are either sendcount or recvcount
    # sendcount is the number of elements in sendbuff involved for this collective
    # recvcount is the number of elements in recvbuff involved for this collective
    # sometimes sendcount != recvcount, but what we want is max(sendcount,recvcount)
    count_factor = {
        'AllReduce':     1,
        'ReduceScatter': N,
        'AllGather':     N,
        'AlltoAll':      N,
        'Broadcast':     1,
        'Reduce':        1,
        "Send":          1,
        "Recv":          1,
    }

    # Convert element counts to bytes
    datatype_factor = {
        0: 1,  # ncclInt8
        1: 1,  # ncclUint8
        2: 4,  # ncclInt32
        4: 8,  # ncclInt64
        6: 2,  # ncclFloat16
        7: 4,  # ncclFloat32
        8: 8,  # ncclFloat64
        9: 2,  # ncclBfloat16
    }

    # Ratio between size of the data involved in the collective and the average volume of data transfer per GPU
    bus_bandwidth_factor = {
        'AllReduce':     2 * (N - 1) / N,
        'ReduceScatter': (N - 1) / N,
        'AllGather':     (N - 1) / N,
        'AlltoAll':      (N - 1) / N,
        # nccl-tests reports busBw = baseBw (factor 1) for Broadcast/Reduce, so this
        # returns S = count*typesize per GPU. NOTE: this is the bus-bandwidth metric,
        # NOT the aggregate fabric traffic. Actual total wire volume is (N-1)*S for both.
        # Traffic may also be distributed unevenly across GPUs/links depending on the
        # algorithm NCCL picks, unlike the symmetric AllReduce/AllGather/ReduceScatter.
        'Broadcast':     1,
        'Reduce':        1,
        "Send":          1,
        "Recv":          1,
    }

    return nccl_count * count_factor[coll_op] * datatype_factor[datatype] * bus_bandwidth_factor[coll_op]

def parse_nccl_fabric(log_file: str) -> dict:
    """
    Parse NCCL debug log to extract communicator fabric information.

    Analyzes NCCL_DEBUG=INFO logs to identify which transport fabrics
    (NVLink, InfiniBand, etc.) are used by each NCCL communicator.
    """
    comms = defaultdict(lambda: {
        "nranks": None,
        "nnodes": None,
        "max_bw_gpu_to_gpu": None,
        "fab": set(),
    })

    current_comm = None
    with open(log_file, "r") as f:
        for line in f:
            # Detect Init START - begin tracking this communicator
            m = re.search(
                r"ncclCommInitRankConfig comm (0x\w+) rank \d+ nranks (\d+).*Init START",
                line,
            )
            if m:
                current_comm = m.group(1)
                comms[current_comm]["nranks"] = int(m.group(2))
                continue

            # Detect Init COMPLETE - stop tracking this communicator
            if current_comm and "Init COMPLETE" in line and current_comm in line:
                current_comm = None
                continue

            # Only parse if we're between Init START and Init COMPLETE
            if current_comm is None:
                continue

            # Extract nNodes
            m = re.search(r"comm (0x\w+).*nNodes (\d+)", line)
            if m and m.group(1) == current_comm:
                comms[current_comm]["nnodes"] = int(m.group(2))
                continue

            # Extract maxBw
            m = re.search(r"=== System : maxBw ([\d.]+) totalBw [\d.]+ ===", line)
            if m:
                comms[current_comm]["max_bw_gpu_to_gpu"] = float(m.group(1))
                continue

            # Detect NVLink in topology (+ NVL[bw] - GPU/...)
            if re.search(r"\+ NVL\[[\d.]+\] -", line):
                comms[current_comm]["fab"].add("NVLink")
                continue

            # Detect Network in topology (+ NET[bw] - NET/...)
            if re.search(r"\+ NET\[[\d.]+\] -", line):
                comms[current_comm]["fab"].add("Network")
                continue

    return dict(comms)

def comm_profiler(log_files: list[str], processes: list[int]) -> dict:
    """
    Parse NCCL INFO logs for a single GPU and visualize collective communication volumes.

    Expects a log file where training markers are interleaved with NCCL traces via `NCCLTagger`:

        >>> --- Step 1 --- Phase Forward
        [2026-03-12 13:57:45] host:pid:tid [rank] NCCL INFO AllGather: opCount ...
        >>> --- Step 1 --- Phase Backward
        ...
    """

    coll_dict = {
        'timestamp': [],
        'process': [],
        'coll_operation': [],
        'data_volume': [],
        'comm_per_gpu': [],
        'datatype': [],
        'op': [],
        'fab': [],
        'nranks': [],
        'train_step': [],
        'phase': [],
    }

    for (log_file, process) in zip(log_files, processes):
        current_step = None   # current training step
        current_phase = None  # current phase in the training step

        comms_info = parse_nccl_fabric(log_file)
        with open(log_file, "r") as f:
            for line in f:

                ## FIND THE USER-DEFINED TAG
                # Training tag: >>> --- Step 1 --- Phase Forward
                if line.startswith(">>> --- Step"):
                    # tokens: ['>>>', '---', 'Step', '1', '---', 'Phase', 'Forward']
                    tokens = line.strip().split()
                    current_step = int(tokens[3])
                    current_phase = tokens[6]
                    continue
                if current_step is None:
                    continue

                # Example line:
                # [2026-03-12 13:57:49] jzxh115:3915132:3917207 [3] NCCL INFO ReduceScatter: opCount 0 sendbuff 0x14df1c000000 recvbuff 0x14e100000000 count 58264448 ...
                if "count" not in line:
                    continue

                tokens = line.split()
    
                timestamp = tokens[0][1:] + " " + tokens[1][:-1]  # "2026-03-12 13:57:45"
                coll_op = tokens[6][:-1]                          # AllGather: -> AllGather
                count = int(tokens[tokens.index("count") + 1])
                datatype = int(tokens[tokens.index("datatype") + 1])
                op = int(tokens[tokens.index("op") + 1])
                communicator = tokens[tokens.index("comm") + 1]
                nranks = int(re.search(r'nranks=(\d+)', line).group(1))
    
                coll_dict['timestamp'].append(timestamp)
                coll_dict['process'].append(process)
                coll_dict['coll_operation'].append(coll_op)
                coll_dict['data_volume'].append(count)
                coll_dict['comm_per_gpu'].append(wire_bytes_per_gpu(coll_op, count, datatype, nranks))
                coll_dict['datatype'].append(datatype)
                coll_dict['op'].append(op)
                coll_dict['fab'].append(frozenset(comms_info[communicator]['fab']))
                coll_dict['nranks'].append(nranks)
                coll_dict['train_step'].append(current_step)
                coll_dict['phase'].append(current_phase)

    return coll_dict

def _print_percentile_summary(data: list, name: str) -> None:
    """Display distribution statistics for a list of volume measurements.
    
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

        print(f">>> {name}: min {p0/1e9:.1f} GB | 10th percentile {p10/1e9:.1f} GB | median {p50/1e9:.1f} GB | 90th percentile {p90/1e9:.1f} GB | max {p100/1e9:.1f} GB")
    else:
        print(f">>> {name}: insufficient data (n={len(data)})")

def get_comm_results(coll: dict, skip_steps: int=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate NCCL communication volumes by phase and transport fabric.

    Processes raw NCCL communication measurements to compute per-step
    communication volumes, grouped by training phase and by transport
    fabric (NVLink, InfiniBand, etc.).

    Note:
        - Only processes data from a single GPU to avoid double-counting
        - Prints percentile summaries for each phase and fabric
        - Missing values (phases/fabrics not present in some steps) are filled with 0
    """
    df = pd.DataFrame(coll)
    df = df[df['train_step'] > skip_steps].reset_index()
    df_1gpu = df[df['process'] == df['process'].unique()[0]].copy()

    # Group by `train_step` and `phase`
    df_1gpu_per_phase = df_1gpu.groupby(['train_step', 'phase'], sort=False)['comm_per_gpu'].sum().reset_index()

    # Pivot: colonnes = phases, index = train_step, valeurs = comm_per_gpu
    pivot = df_1gpu_per_phase.pivot(
        index='train_step',
        columns='phase',
        values='comm_per_gpu'
    ).fillna(0)  # Remplir les valeurs manquantes avec 0

    # Communication volumes estimation per phase
    for phase in pivot.columns:
        _print_percentile_summary(pivot[phase].tolist(), f"{phase} communication volume per step")

    # Total communication volumes estimation
    comm_vol_step = pivot.sum(axis=1).tolist()
    _print_percentile_summary(comm_vol_step, "Total communication volume per step")


    # Group by `train_step` and `fab`
    df_1gpu_per_communicator = df_1gpu.groupby(['train_step', 'fab'], sort=False)['comm_per_gpu'].sum().reset_index()

    # Pivot: colonnes = fab, index = train_step
    pivot_fab = df_1gpu_per_communicator.pivot(
        index='train_step',
        columns='fab',
        values='comm_per_gpu'
    ).fillna(0)

    # Communication volumes estimation per fabrics used
    for fab in pivot_fab.columns:
        _print_percentile_summary(pivot_fab[fab].tolist(), f"{fab} communication volume per step")

    return pivot, pivot_fab

def plot_comm_profiler(coll: dict, n_display=50) -> None:
    """Visualize NCCL collective communication patterns across training steps.

    Generates multiple bar plots to analyze communication volumes:
    1. Per-operation breakdown
    2. Aggregate communication by operation type
    3. Total communication per step and phase
    """

    df = pd.DataFrame(coll)

    # Check if df is empty 
    assert not df.empty, "No NCCL collective operations found in the log file."

    # Check that every process has the same number of collective operations
    counts = df.groupby('process').size()
    assert counts.nunique() == 1, f"Mismatch in collective counts per process: {counts.to_dict()}"

    df_plot = pd.DataFrame()
    for r in df['process'].unique():
        df_plot[f'process: {r}'] = df[df['process'] == r].comm_per_gpu.reset_index(drop=True)
    
    # Build operation labels
    nccldtype = {
        0: 'ncclInt8',
        1: 'ncclUint8',
        2: 'ncclInt32',
        3: 'ncclUint32',
        4: 'ncclInt64',
        5: 'ncclUint64',
        6: 'ncclFloat16',
        7: 'ncclFloat32',
        8: 'ncclFloat64',
        9: 'ncclBfloat16',
    }
    single_process = df[df['process'] == df['process'].unique()[0]].reset_index(drop=True)
    df_plot['label'] = (
        "Step "
        + single_process['train_step'].astype(str)
        + " - "
        + single_process['phase']
        + " - "
        + single_process['coll_operation']
        + " ("
        + single_process['datatype'].map(nccldtype).astype(str)
        + ")"
    )
    df_plot = df_plot.set_index('label')

    # --- Per-operation bar plot ---
    plot_title = f'Collective Communication Profiler - Total operations: {len(df)}'

    if n_display:
        df_plot.iloc[:n_display].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'{plot_title} (first {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
        
        df_plot.iloc[-n_display:].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'{plot_title} (last {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
    else:
        df_plot.plot.bar(
            figsize=(18, 6), rot=90,
            title=plot_title,
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()

    # --- Aggregate bar plot grouped by label ---
    dfagg = df_plot.groupby('label', sort=False).sum()

    if n_display:
        dfagg.iloc[:n_display].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Aggregate Collective Communication Profiler - Total volume: {df[df['process'] == df['process'].unique()[0]].comm_per_gpu.sum()/1e9:.2f} GB/GPU (first {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
        
        dfagg.iloc[-n_display:].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Aggregate Collective Communication Profiler - Total volume: {df[df['process'] == df['process'].unique()[0]].comm_per_gpu.sum()/1e9:.2f} GB/GPU (last {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
    else:
        dfagg.plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Aggregate Collective Communication Profiler - Total volume: {df[df['process'] == df['process'].unique()[0]].comm_per_gpu.sum()/1e9:.2f} GB/GPU',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()

    # --- Total communication per step per phase ---
    dfphase = df_plot.copy().reset_index(drop=True)  # vire l'index 'label'
    
    single_process = df[df['process'] == df['process'].unique()[0]].reset_index(drop=True)
    dfphase['step_phase'] = (
        "Step "
        + single_process['train_step'].astype(str)
        + " - "
        + single_process['phase']
    )
    dfphase = dfphase.groupby('step_phase', sort=False).sum()

    if n_display:
        dfphase.iloc[:n_display].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Total Communication per Step per Phase (first {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
        
        dfphase.iloc[-n_display:].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Total Communication per Step per Phase (last {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
    else:
        dfphase.plot.bar(
            figsize=(18, 6), rot=90,
            title='Total Communication per Step per Phase',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()

    return df
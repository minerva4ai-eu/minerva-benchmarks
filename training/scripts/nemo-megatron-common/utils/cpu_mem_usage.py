"""CPU and memory usage utilities for monitoring process resources."""

from typing import Optional, Tuple, Dict


def get_proc_status(keys: Optional[Tuple[str, ...]] = None) -> Dict[str, str] | Tuple[str, ...]:
    """Get values from /proc/self/status file.

    Args:
        keys: Tuple of keys to retrieve. If None, returns all key-value pairs.

    Returns:
        Dictionary of all status values if keys is None, otherwise a tuple of values for the requested keys.
    """
    with open('/proc/self/status') as f:
        data = dict(map(str.strip, line.split(':', 1)) for line in f)

    if keys is None:
        return data
    return tuple(data[k] for k in keys)


def memory_usage() -> None:
    """Print memory usage of the current process.

    Displays VmPeak (peak virtual memory) and VmHWM (high water mark for RSS).
    """
    try:
        peak, hwm = get_proc_status(('VmPeak', 'VmHWM'))
        print(f"VmPeak: {peak}, VmHWM: {hwm}")
    except FileNotFoundError:
        print("Memory usage unavailable: /proc/self/status not found")
    except KeyError as e:
        print(f"Memory usage unavailable: key {e} not found in /proc/self/status")

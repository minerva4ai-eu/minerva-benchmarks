import os
from pathlib import Path


class NCCLTagger:
    """Write training phase markers into the NCCL debug log file.

    Singleton.

    Instead of opening the log file separately (which creates a second file
    descriptor with its own offset, causing tags to be overwritten by NCCL),
    this class locates the file descriptor already opened by NCCL via
    /proc/<pid>/fd and writes directly through it with os.write(). This
    ensures tags and NCCL traces share the same file offset and interleave
    correctly.

    Must be instantiated after NCCL has opened its log file (i.e. after
    init_process_group and ideally after a first collective like dist.barrier).
    """
    _instance = None

    def __new__(cls):
        """Avoid creating another instance.
        
        __new__ is called to create the object
        __init__ is called afterward.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):        
        self._fd = self._find_nccl_fd()

    def _find_nccl_fd(self):
        """Find the file descriptor used by NCCL to write tags."""
        nccl_log_path = os.environ.get("NCCL_DEBUG_FILE", None)
        if nccl_log_path is None:
            return None

        nccl_log_path = nccl_log_path.replace("%p", str(os.getpid()))
        target = os.path.realpath(nccl_log_path)
        # Check all fd of this process
        fd_dir = Path(f"/proc/{os.getpid()}/fd")
        for fd_link in fd_dir.iterdir():
            try:
                if os.path.realpath(fd_link) == target:
                    fd_num = int(fd_link.name)
                    # Avoid stdin/stdout/stderr
                    if fd_num > 2:
                        return fd_num
            except (OSError, ValueError):  # FD may have been closed, FD may not be a number
                continue
        return None

    def tag(self, step: int, phase: str):
        if self._fd is None:
            return
        msg = f"\n>>> --- Step {step} --- Phase {phase}\n"
        os.write(self._fd, msg.encode())
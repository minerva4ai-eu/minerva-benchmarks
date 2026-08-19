# benchmark/state.py
import subprocess
from dataclasses import dataclass
from pathlib import Path

import scripts.slurm.utils as u
from omegaconf import DictConfig

SLURM_STATUS_DASHBOARD = {
    # Active / Lifecycle States
    "pending": {
        "code": "PD",
        "code_complete": "pending",
        "icon": "⏳",
        "utf8": "\u23f3",
        "description": "Queued and waiting for resources",
    },
    "running": {
        "code": "R",
        "code_complete": "running",
        "icon": "🏃",
        "utf8": "\U0001f3c3",
        "description": "Actively executing on compute nodes",
    },
    "completing": {
        "code": "CG",
        "code_complete": "completing",
        "icon": "🚶",
        "utf8": "\U0001f6b6",
        "description": "Finishing up and cleaning up node processes",
    },
    # Interrupted / Paused States
    "suspended": {
        "code": "S",
        "code_complete": "suspended",
        "icon": "⏸️",
        "utf8": "\u23ef\ufe0f",
        "description": "Paused, cores released for other jobs",
    },
    "stopped": {
        "code": "ST",
        "code_complete": "stopped",
        "icon": "🛑",
        "utf8": "\U0001f6d1",
        "description": "Paused, but retaining hold on cores",
    },
    "preempted": {
        "code": "PR",
        "code_complete": "preempted",
        "icon": "💥",
        "utf8": "\U0001f4a5",
        "description": "Evicted by a higher priority workload",
    },
    "requeued": {
        "code": "RQ",
        "code_complete": "requeued",
        "icon": "🔄",
        "utf8": "\U0001f504",
        "description": "Kicked out but returned to the queue",
    },
    # Termination / Success & Failure States
    "completed": {
        "code": "CD",
        "code_complete": "completed",
        "icon": "✅",
        "utf8": "\u2705",
        "description": "Finished successfully with exit code 0",
    },
    "failed": {
        "code": "F",
        "code_complete": "failed",
        "icon": "❌",
        "utf8": "\u274c",
        "description": "Terminated with a non-zero exit code",
    },
    "timeout": {
        "code": "TO",
        "code_complete": "timeout",
        "icon": "⏰",
        "utf8": "\u23f0",
        "description": "Killed for exceeding specified wall-clock limit",
    },
    "cancelled": {
        "code": "CA",
        "code_complete": "cancelled",
        "icon": "🚫",
        "utf8": "\U0001f6ab",
        "description": "Manually killed by user or admin via scancel",
    },
    "out_of_memory": {
        "code": "OOM",
        "code_complete": "out_of_memory",
        "icon": "🚨",
        "utf8": "\U0001f6a8",
        "description": "Terminated for exceeding requested RAM limit",
    },
    "node_fail": {
        "code": "NF",
        "code_complete": "node_fail",
        "icon": "💀",
        "utf8": "\U0001f480",
        "description": "Aborted due to physical server hardware crash",
    },
}


def load_all(run: str) -> list[dict]:
    return u.read_jsonl(run)


@dataclass
class JobInfo:
    job_id: str
    job_name: str
    raw_state: str
    exit_code: str
    status_meta: dict


def get_job_info(job_id: str) -> JobInfo:
    """
    Queries Slurm accounting database via sacct for a specific Job ID.
    Returns a parsed dictionary containing job details and normalized status metadata.
    """
    # 1. Execute sacct command.
    # Using -p (parsable, pipe-delimited) and -n (no header) makes extraction trivial.
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--allocations",
        "-p",
        "-n",
        "--format=JobID,JobName,State,ExitCode",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = result.stdout.strip()

    # If stdout is empty, the Job ID doesn't exist in the cluster history
    if not stdout:
        raise Exception(
            f"Could not find job '{job_id}' in cluster history! Please, check that jobis provided are valid!"
        )

    # 2. Parse the pipe-delimited string layout
    # Example output format: "1234567|my_train_job|COMPLETED|0:0"
    parts = stdout.split("|")

    raw_state = parts[2].split()[
        0
    ]  # Grab first word in case of state variations (e.g. "CANCELLED by 1000")
    state_key = raw_state.lower()  # Normalize to match dictionary lookup keys

    # Parse into JobInfo dataclass
    job_info = JobInfo(
        job_id=parts[0],
        job_name=parts[1],
        raw_state=raw_state,
        exit_code=parts[3],
        status_meta=SLURM_STATUS_DASHBOARD.get(
            state_key,
            {
                "code": "??",
                "code_complete": state_key,
                "icon": "❓",
                "description": "Unknown or unmapped state",
            },
        ),
    )

    return job_info


def print_job_status(job: dict, job_info: JobInfo):
    dil = f"{u.YELLOW}|{u.RESET}"
    print(
        f"{u.POINT_BULLET} {job_info.status_meta['icon']} ({job_info.status_meta['code_complete']}) {u.ARROW_CHEVRON} "
        + f"{job['id']} {dil} {job['cfg_id']} {dil} {job['dependency']}"
    )


def filter_by_status(runs_dir: Path, status: str) -> list[DictConfig]:
    return [c for c in load_all(runs_dir) if c.status == status]

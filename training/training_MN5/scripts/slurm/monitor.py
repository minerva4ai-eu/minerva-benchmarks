# benchmark/state.py
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_all(runs_dir: Path) -> list[DictConfig]:
    return [OmegaConf.load(f) for f in sorted(runs_dir.glob("combo_*.yaml"))]


def update_status(combo: DictConfig, runs_dir: Path, status: str, job_id: str = None):
    combo.status = status
    if job_id:
        combo.job_id = job_id
    OmegaConf.save(combo, runs_dir / f"{combo.combo_id}.yaml")


def filter_by_status(runs_dir: Path, status: str) -> list[DictConfig]:
    return [c for c in load_all(runs_dir) if c.status == status]

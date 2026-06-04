# benchmark/cli.py
from pathlib import Path

import click
from configs_hydra.hydra_app import generate_valid_combos
from omegaconf import OmegaConf
from scripts.slurm.monitor import filter_by_status, load_all
from scripts.slurm.submitter import submit_job

RUNS_DIR = Path("runs")
BASE_DIR = Path(".")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dry-run", is_flag=True)
@click.option(
    "--configs-path",
)
@click.option(
    "--config-name",
)
@click.option(
    "--output",
)
def run(dry_run, configs_path, config_name, output):
    """Generate all valid configs and submit all pending jobs."""
    valid, skipped = generate_valid_combos(
        config_path=configs_path, config_name=config_name, outpath=output
    )
    click.echo(f"\nSubmitting {len(valid)} jobs...")
    dependency_jobid = ""
    for cfg in valid:
        if dry_run:
            click.echo(f"  [dry] {cfg.id}")
        else:
            for repeat_id in range(1, cfg.experiment.repeat + 1):
                pass
                dependency_jobid = submit_job(
                    cfg=cfg,
                    base_dir=BASE_DIR,
                    run_id=repeat_id,
                    dependency=dependency_jobid,
                )


@cli.command()
@click.option("--failed", "mode", flag_value="failed", default=True)
@click.option("--pending", "mode", flag_value="pending")
@click.option(
    "--id", "cfg_id", default=None, help="Rerun a specific benchmark configuration"
)
def rerun(mode, cfg_id):
    """Rerun failed or pending jobs, or a specific combo by id."""
    if cfg_id:
        path = RUNS_DIR / f"{cfg_id}.yaml"
        combos = [OmegaConf.load(path)]
    else:
        combos = filter_by_status(RUNS_DIR, mode)

    click.echo(f"Resubmitting {len(combos)} jobs (status={mode})...")
    for combo in combos:
        submit_job(combo, BASE_DIR, RUNS_DIR)


@cli.command()
def status():
    """Print a summary of all run statuses."""
    all_combos = load_all(RUNS_DIR)
    from collections import Counter

    counts = Counter(c.status for c in all_combos)
    for st, n in sorted(counts.items()):
        icon = {"done": "✓", "running": "⟳", "failed": "✗", "pending": "·"}.get(st, "?")
        click.echo(f"  {icon} {st:10s} {n}")


if __name__ == "__main__":
    cli()

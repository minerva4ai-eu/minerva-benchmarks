# benchmark/cli.py
import os
import secrets
from pathlib import Path

import click
import scripts.slurm.monitor as m
import scripts.slurm.utils as u
from configs_hydra.hydra_app import generate_valid_combos
from omegaconf import OmegaConf
from scripts.slurm.submitter import submit_job

RUNS_DIR = Path("results/")
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
    short_id = secrets.token_hex(4)
    click.echo(f"\nSlurm monitor ID: {short_id}")
    click.echo(f"\nSubmitting {len(valid)} jobs...")
    dependency_jobid = ""
    jobs_submitted = []
    for cfg in valid:
        if dry_run:
            click.echo(f"  [dry] {cfg.id}")
        else:
            for repeat_id in range(1, cfg.experiment.repeat + 1):
                job_desc = {
                    "id": None,
                    "cfg_id": None,
                    "dependency": dependency_jobid,
                }
                dependency_jobid = submit_job(
                    cfg=cfg,
                    base_dir=BASE_DIR,
                    run_id=repeat_id,
                    dependency=dependency_jobid,
                )
                job_desc["id"] = dependency_jobid
                job_desc["cfg_id"] = cfg.id
                jobs_submitted.append(job_desc)
    slurm_monitor_dir = os.path.join(output, "slurm-monitor")
    run_monitor_dir = os.path.join(slurm_monitor_dir, short_id)
    os.makedirs(run_monitor_dir, exist_ok=True)
    u.write_jsonl(
        d=jobs_submitted, p=os.path.join(run_monitor_dir, "jobs_submitted.jsonl")
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
        combos = m.filter_by_status(RUNS_DIR, mode)

    click.echo(f"Resubmitting {len(combos)} jobs (status={mode})...")
    for combo in combos:
        submit_job(combo, BASE_DIR, RUNS_DIR)


@cli.command()
@click.option(
    "--run-id",
    "run_id",
    type=str,
    help="ID of training benchmark run generated automatically into folder 'slurm-monitor'",
    required=True,
)
def status(run_id):
    """Print a summary of all run statuses."""
    run_jobs = m.load_all(f"{RUNS_DIR}/slurm-monitor/{run_id}/jobs_submitted.jsonl")

    print(f"\nJob status for run {u.CYAN}{run_id}{u.RESET}:\n")
    s1 = " " * 20
    s2 = " " * 40
    s3 = " " * 49
    print(f"{s1} {u.YELLOW}JOBID{s2}RUNID{s3}DEPJOB")
    for job in sorted(run_jobs, key=lambda j: j["id"]):
        job_info = m.get_job_info(job["id"])

        m.print_job_status(job, job_info)


if __name__ == "__main__":
    cli()

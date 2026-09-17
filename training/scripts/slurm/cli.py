# benchmark/cli.py
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import click
import configs_hydra.model_framework_dataset as mfd
import scripts.slurm.utils as u
from configs_hydra.hydra_app import generate_valid_combos
from scripts.slurm.cli_utils import *
from scripts.slurm.submitter import submit_job, write_config

if TYPE_CHECKING:
    from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
# )

import logging

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
LOG_DIR = os.path.join("outputs", "logs", "pycli", TIMESTAMP)
LOG_DIR = os.environ.get("LOG_DIR", LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, "minerva.log")
LOG_FILE = os.environ.get("LOG_FILE", LOG_FILE)
LOG_DIR = "/".join(LOG_FILE.split("/")[:-1])
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# FIXME: logging level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s |  %(levelname)s | %(name)s : %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MINERVA SLURM job submission CLI for LLM training and fine-tuning benchmarks."""


@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate configs and build launch folders without submitting jobs.",
)
@click.option(
    "--configs-path",
    default=DEFAULT_CONFIGS_PATH,
    help="Path to the Hydra config directory (default: ./configs_hydra/configs).",
)
@click.option(
    "--config-name",
    default=DEFAULT_CONFIG_NAME,
    help="Base config name to compose (e.g., 'base', 'base-MN5').",
    required=True,
)
@click.option(
    "--runs-dir",
    default=RUNS_DIR,
    help="Output directory for generated configs and results (default: benchmark-runs/).",
)
@click.option(
    "--per-model-jobs",
    is_flag=True,
    help="Group experiments per model",
)
@click.option(
    "--per-nodes-jobs",
    is_flag=True,
    help="Group experiments per nodes",
)
@click.option(
    "--models",
    type=str,
    default="",
    help=(
        "Run a benchmark configurations by providing the names of the models to run. "
        "Must be provided in a comma separated format, e.g. bash miberva-cli.sh run --config-path MN5-vevn --models gemma3-1b, mistral_7b"
        f"Valid model names: {mfd.MODELS}"
    ),
)
@click.option(
    "--frameworks",
    type=str,
    default="",
    help=(
        "Run a benchmark configurations by providing the names of the frameworks to run. "
        "Must be provided in a comma separated format, e.g. bash miberva-cli.sh run --config-path MN5-vevn --frameworks accelerate-cuda130, deepspeed-cuda130"
        f"Valid framework names: {mfd.FRAMEWORKS}"
    ),
)
@click.option(
    "--datasets",
    type=str,
    default="",
    help=(
        "Run a benchmark configurations by providing the names of the datasets to run. "
        "Must be provided in a comma separated format, e.g. bash miberva-cli.sh run --config-path MN5-vevn --datasets alpaca, squadv2"
        f"Valid dataset names: {mfd.DATASETS}"
    ),
)
@click.option(
    "--yaml",
    "yamls",
    multiple=True,
    default=None,
    help=(
        "Run a benchmark configuration by providing the path to a BenchmarkConfig YAML file. "
        "Can be repeated for multiple configs, e.g. '--yaml path1.yaml --yaml path2.yaml. "
        "First run a '--dry-run' to compose YAML configuration files and then use their paths to run them individually."
    ),
)
@click.option(
    "--nnodes",
    default=None,
    type=int,
    help="Output directory for generated configs and results (default: benchmark-runs/).",
)
def run(
    dry_run,
    configs_path,
    config_name,
    runs_dir,
    per_model_jobs,
    per_nodes_jobs,
    models,
    frameworks,
    datasets,
    yamls,
    nnodes,
):
    """Generate benchmark configurations and submit SLURM job with steps.

    Composes valid config combinations from Hydra configs or runs specific
    YAML files. Supports dry-run mode for config examination.
    """
    logger.info("Running cli...")
    click.echo("\n")
    click.echo(
        f"{u.POINT_DIAMOND} {u.CYAN} Running {u.MAGENTA} MINERVA Benchmarks {u.CYAN} for LLMs training and fine-tuning {u.POINT_DIAMOND} {u.RESET}"
    )

    # TODO: Check desired behavior
    if config_name == DEFAULT_CONFIG_NAME:
        click.echo(
            f"\t{u.FAILURE_HEAVY} {u.RED}!WARNING! Argument '--config-name' is defaulting to '{DEFAULT_CONFIG_NAME}'...{u.RESET}"
        )
        click.echo(
            f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Make sure to provide the correct '--config-name' pointing to a .yaml configuration profile inside {DEFAULT_CONFIGS_PATH}{u.RESET}"
        )
        exit(1)

    # TODO: review output structure
    runs_dir = f"{runs_dir}-{config_name}"
    run_date = datetime.now().date().strftime("%d-%m-%Y")

    cfgs_valid: list[BenchmarkConfig] = []
    cfgs_paths: list[str] = []

    if isinstance(yamls, tuple) and len(yamls) == 1:
        yamls: list[str] = [y.strip() for y in yamls[0].split("--yaml") if y != ""]
    if yamls:
        if dry_run:
            logger.error(
                f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Arguments '--yaml' and '--dry-run' cannot be combined...{u.RESET}"
            )
            exit(1)
        for y in yamls:
            click.echo(f"\t{u.POINT_SQUARE} {u.YELLOW}Searching for {y}{u.RESET}")

            try:
                _cfg = u.load_config(y)
                cfgs_valid.append(_cfg)
                cfgs_paths.append(y)

                click.echo(f"\t{u.SUCCESS_HEAVY} {u.GREEN}YAML config FOUND!{u.RESET}")
            except FileNotFoundError:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED}YAML config could NOT be FOUND!{u.RESET}"
                )
                continue
            except Exception as e:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read file...{u.RESET}"
                )
                raise e
    else:
        # Filter benchmarks to run on provided model names
        if models:
            models = models.split(",")
            if len(models) == 0:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Model names provided must be separated by comma ','. Provided: '{models}'{u.RESET}"
                    + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.' {u.RESET}"
                )
                sys.exit(1)
            for m in models:
                if m not in mfd.MODELS:
                    click.echo(
                        f"\t{u.FAILURE_HEAVY} {u.RED} Model '{m}' could not be found in available models.{u.RESET}"
                        + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.' {u.RESET}"
                    )
                    sys.exit(1)
            mfd.MODELS = models

        # Filter benchmarks to run on provided framework names
        if frameworks:
            frameworks = frameworks.split(",")
            if len(frameworks) == 0:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Framework names provided must be separated by comma ','. Provided: '{frameworks}'{u.RESET}"
                    + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.' {u.RESET}"
                )
                sys.exit(1)
            for m in frameworks:
                if m not in mfd.FRAMEWORKS:
                    click.echo(
                        f"\t{u.FAILURE_HEAVY} {u.RED} Framework '{m}' could not be found in available frameworks.{u.RESET}"
                        + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.' {u.RESET}"
                    )
                    sys.exit(1)
            mfd.FRAMEWORKS = frameworks

        # Filter benchmarks to run on provided dataset names
        if datasets:
            datasets = datasets.split(",")
            if len(datasets) == 0:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Dataset names provided must be separated by comma ','. Provided: '{datasets}'{u.RESET}"
                    + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.' {u.RESET}"
                )
                sys.exit(1)
            for m in datasets:
                if m not in mfd.DATASETS:
                    click.echo(
                        f"\t{u.FAILURE_HEAVY} {u.RED} Dataset '{m}' could not be found in available datasets.{u.RESET}"
                        + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.' {u.RESET}"
                    )
                    sys.exit(1)
            mfd.DATASETS = datasets

        cfgs_valid, _ = generate_valid_combos(
            config_path=configs_path,
            config_name=config_name,
            outpath=runs_dir,
            run_date=run_date,
            dry=dry_run,
        )
        for cfg in cfgs_valid:
            cfgs_paths.append(
                write_config(
                    cfg=cfg, base_dir=str(BASE_DIR.absolute()), runs_dir=runs_dir
                )
            )

        if dry_run:
            for cfg in cfgs_valid:
                pass
                # click.echo(f"  {u.YELLOW}[dry]{u.RESET} {cfg.id}")
            return

    logger.info(
        "Calling submit_job(cfg_name=%s, config_path=%s, runs_dir=%s,run_date=%s)",
        config_name,
        configs_path,
        runs_dir,
        run_date,
    )
    if not cfgs_valid:
        click.echo(
            f"{u.FAILURE_HEAVY} {u.RED} No valid benchmark configurations found to run! Exiting...{u.RESET}"
        )
        return
    if per_model_jobs:
        per_model_cfgs: dict[str, dict[str, list[BenchmarkConfig | str]]] = {}
        for cfg, path in zip(cfgs_valid, cfgs_paths):
            if not cfg.model.name in per_model_cfgs:
                per_model_cfgs[cfg.model.name] = {}
                per_model_cfgs[cfg.model.name]["cfgs_valid"] = []
                per_model_cfgs[cfg.model.name]["cfgs_paths"] = []

            per_model_cfgs[cfg.model.name]["cfgs_valid"].append(cfg)
            per_model_cfgs[cfg.model.name]["cfgs_paths"].append(path)

        for model, combinations in per_model_cfgs.items():
            click.echo(
                f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Preparing job for model {u.YELLOW}'{model}'{u.CYAN}... {u.RESET}"
            )
            jobid = submit_job(
                runs_dir=runs_dir,
                cfgs=combinations["cfgs_valid"],
                cfgs_paths=combinations["cfgs_paths"],
                run_date=run_date,
                nnodes=nnodes,
            )
        return
    if per_nodes_jobs:
        per_nodes_cfgs: dict[int, dict[str, list[BenchmarkConfig | str]]] = {}
        for cfg, path in zip(cfgs_valid, cfgs_paths):
            if not cfg.slurm.sbatch.nodes in per_nodes_cfgs:
                per_nodes_cfgs[cfg.slurm.sbatch.nodes] = {}
                per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_valid"] = []
                per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_paths"] = []

            per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_valid"].append(cfg)
            per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_paths"].append(path)

        for model, combinations in per_nodes_cfgs.items():
            click.echo(
                f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Preparing job for {u.YELLOW}'{model}'{u.CYAN} nodes... {u.RESET}"
            )
            jobid = submit_job(
                runs_dir=runs_dir,
                cfgs=combinations["cfgs_valid"],
                cfgs_paths=combinations["cfgs_paths"],
                run_date=run_date,
                nnodes=nnodes,
            )
        return

    # ToDo: Add per-framework-jobs, per-nodes-jobs, etc. grouping options for job-steps optimal
    #       resource and results organization

    click.echo(f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Preparing jobs ... {u.RESET}")
    jobid = submit_job(
        runs_dir=runs_dir,
        cfgs=cfgs_valid,
        cfgs_paths=cfgs_paths,
        run_date=run_date,
        nnodes=nnodes,
    )
    click.echo(
        f"\n\t{u.EMOJI_INFO}   Results in: "
        f"{u.YELLOW}'{os.path.join(runs_dir, run_date, jobid)}'{u.RESET}"
    )


# def _parse_delimiter_separated(value: str | None, delimiter: str = " ") -> set:
#    """Parse a space-separated string into a set of values, or return None."""
#    if not value:
#        return set()
#    return set(value.split(delimiter))
#
#
# def _filter_jobs_by_config(
#    run_jobs: list[dict],
#    runs_dir: str,
#    run_date: str,
#    run_id: str,
#    rerun_id: int | None,
#    model_names: set | None = None,
#    framework_names: set | None = None,
#    parallelism_names: set | None = None,
#    nodes_values: set | None = None,
# ) -> list[dict]:
#    """
#    Filter jobs by loading their YAML configs and matching against provided criteria.
#    All provided filters use AND logic — a job must match ALL specified filters.
#    Returns the filtered list of jobs.
#    """
#    filtered = []
#    for job in run_jobs:
#        config_dir = "/".join(job["launch_folder"].split("/")[:-2])
#        yaml_path = os.path.join(config_dir, job["yaml_filename"])
#
#        try:
#            cfg: BenchmarkConfig = DictConfig(u.load_yaml(yaml_path))
#        except Exception:
#            # If we can't load the config, skip this job
#            continue
#
#        # Check each filter (AND logic)
#        match = True
#
#        if model_names is not None:
#            if cfg.model.name not in model_names:
#                match = False
#
#        if match and framework_names is not None:
#            if cfg.framework.name not in framework_names:
#                match = False
#
#        if match and parallelism_names is not None:
#            if cfg.framework.parallelism_name not in parallelism_names:
#                match = False
#
#        if match and nodes_values is not None:
#            if str(cfg.slurm.sbatch.nodes) not in nodes_values:
#                match = False
#
#        if match:
#            filtered.append(job)
#
#    return filtered


# DEPRECATED subcommand
# ToDo: Reconfigure 'status' to work with steps,
# @cli.command()
# @click.option(
#    "--job-ids",
#    "run_id",
#    type=str,
#    help="SLURM Job IDs in a comma separated format",
#    required=True,
# )
# @click.option(
#    "--runs-dir",
#    "runs_dir",
#    default=RUNS_DIR,
#    help="Output directory for benchmark results (default: benchmark-runs/).",
# )
# @click.option(
#    "--run-date",
#    "run_date",
#    type=str,
#    help="Date of run in format '%d-%m-%Y'.",
#    required=True,
# )
# @click.option(
#    "--model",
#    "model",
#    type=str,
#    default=None,
#    help="Filter by model name(s). Comma-separated for multiple values, e.g. '--model llama3-7b mistral-7b'.",
# )
# @click.option(
#    "--framework",
#    "framework",
#    type=str,
#    default=None,
#    help="Filter by framework name(s). Comma-separated for multiple values, e.g. '--framework accelerate deepspeed'.",
# )
# @click.option(
#    "--parallelism-type",
#    "parallelism",
#    type=str,
#    default=None,
#    help="Filter by parallelism type(s). Comma-separated for multiple values, e.g. '--parallelism-type ddp fsdp'.",
# )
# @click.option(
#    "--nodes",
#    type=str,
#    default=None,
#    help="Filter by number of nodes (exact match). Comma-separated for multiple values, e.g. '--nodes 4 8 16'.",
# )
# @click.option(
#    "--state",
#    "state",
#    type=str,
#    help="Filter by SLURM job state, e.g 'running', 'pending', 'failed', etc.",
# )
# def status(job_ids, runs_dir, run_date, model, framework, parallelism, nodes, state):
#    """Display SLURM job status for a benchmark run.#
#    Shows job states (running, pending, failed, etc.) with optional filtering
#    by model, framework, parallelism type, number of nodes, and SLURM state.
#    Can also check status of specific reruns within a run.
#    """#
#    # Apply config-based filtering
#    job_ids = _parse_delimiter_separated(job_ids, ",")
#    model_names = _parse_delimiter_separated(model, ",")
#    framework_names = _parse_delimiter_separated(framework, ",")
#    parallelism_names = _parse_delimiter_separated(parallelism, ",")
#    nodes_values = _parse_delimiter_separated(nodes, ",")
#    states = _parse_delimiter_separated(state, ",")#
#    if state in states:
#        if state not in m.SLURM_STATUS_DASHBOARD.keys():
#            click.echo(
#                f"{u.RED}Argument '--state' is not valid to filter benchmark jobs for requested run."
#                + f"\n{u.YELLOW}Valid job states: {', '.join(list(m.SLURM_STATUS_DASHBOARD.keys()))}{u.RESET}"
#            )#
#    if any([model_names, framework_names, parallelism_names, nodes_values]):
#        # DEPRECATED code: Will fail
#        job_ids = _filter_jobs_by_config(
#            run_jobs=job_ids,
#            runs_dir=runs_dir,
#            run_date=run_date,
#            model_names=model_names,
#            framework_names=framework_names,
#            parallelism_names=parallelism_names,
#            nodes_values=nodes_values,
#        )
#        if not job_ids:
#            click.echo(
#                f"{u.YELLOW}No jobs match the provided filter criteria.{u.RESET}"
#            )
#            return#
#    click.echo(f"\nJob status for run {u.CYAN}{run_id}{u.RESET}:\n")
#    s1 = " " * 20
#    s2 = " " * 50
#    s3 = " " * 49
#    click.echo(f"{u.YELLOW}JOBID | RUNID | DEPJOB{u.RESET}")
#    for job in sorted(run_jobs, key=lambda j: j["id"]):
#        job_info = m.get_job_info(job["id"])
#        if state and state != job_info.status_meta["code_complete"]:
#            continue
#        m.print_job_status(job, job_info)


# @cli.command()
# @click.option(
#    "--run-date",
#    "run_date",
#    type=str,
#    help="Date of run in format '%d-%m-%Y'.",
#    required=True,
# )
# @click.option(
#    "--run-id",
#    "run_id",
#    type=str,
#    help="Serial id of run on provided date.",
#    required=True,
# )
# @click.option(
#    "--runs-dir",
#    "runs_id",
#    default=RUNS_DIR,
#    help="Output directory for benchmark results (default: benchmark-runs/).",
# )
# @click.option(
#    "--model",
#    "model",
#    type=str,
#    default=None,
#    help="Filter by model name(s). Space-separated for multiple values, e.g. '--model llama3-7b mistral-7b'.",
# )
# @click.option(
#    "--framework",
#    "framework",
#    type=str,
#    default=None,
#    help="Filter by framework name(s). Space-separated for multiple values, e.g. '--framework vllm sglang'.",
# )
# @click.option(
#    "--parallelism-type",
#    "parallelism",
#    type=str,
#    default=None,
#    help="Filter by parallelism type(s). Space-separated for multiple values, e.g. '--parallelism-type dp fsdp'.",
# )
# @click.option(
#    "--nodes",
#    type=str,
#    default=None,
#    help="Filter by number of nodes (exact match). Space-separated for multiple values, e.g. '--nodes 4 8 16'.",
# )
# def cancel(run_date, run_id, runs_id, model, framework, parallelism, nodes):
#    """Cancel all running and pending SLURM jobs for a benchmark run.
#
#    Cancels jobs matching the specified run date and ID, with optional
#    filtering by model, framework, parallelism type, and number of nodes.
#    Only affects running and pending jobs; completed/failed jobs are ignored.
#    """
#
#    click.echo("\n")
#    click.echo(
#        f"{u.POINT_DIAMOND} {u.CYAN} Cancelling {u.MAGENTA} MINERVA Benchmarks {u.CYAN} jobs {u.POINT_DIAMOND} {u.RESET}"
#    )
#
#    run_date = str2date2str(run_date)
#    if not is_valid_date(run_date):
#        raise ValueError(
#            f"{u.RED}Provided invalid '--run-date' value '{run_date}'. Either faulty format or future timestamp!{u.RESET}"
#        )
#
#    run_monitor_folder = f"{runs_id}/slurm-monitor/{run_date}/run_id-{run_id}"
#    run_monitor_path = os.path.join(run_monitor_folder, "jobs_submitted.jsonl")
#
#    if not os.path.exists(run_monitor_path):
#        click.echo(f"{u.RED}Could not find jobs file @ {run_monitor_path}{u.RESET}")
#        exit(1)
#
#    run_jobs = m.load_all(run_monitor_path)
#    if not run_jobs:
#        click.echo(
#            f"{u.YELLOW}No jobs found for run {run_date} | run_id-{run_id}{u.RESET}"
#        )
#        return
#
#    # Apply config-based filtering
#    model_names = _parse_delimiter_separated(model)
#    framework_names = _parse_delimiter_separated(framework)
#    parallelism_names = _parse_delimiter_separated(parallelism)
#    nodes_values = _parse_delimiter_separated(nodes)
#
#    if any([model_names, framework_names, parallelism_names, nodes_values]):
#        run_jobs = _filter_jobs_by_config(
#            run_jobs=run_jobs,
#            runs_dir=runs_id,
#            run_date=run_date,
#            run_id=run_id,
#            rerun_id=None,
#            model_names=model_names,
#            framework_names=framework_names,
#            parallelism_names=parallelism_names,
#            nodes_values=nodes_values,
#        )
#        if not run_jobs:
#            click.echo(
#                f"{u.YELLOW}No jobs match the provided filter criteria.{u.RESET}"
#            )
#            return
#
#    import subprocess
#
#    cancelled = []
#    skipped = []
#
#    for job in sorted(run_jobs, key=lambda j: j["id"]):
#        job_id = job["id"]
#        cfg_id = job["cfg_id"]
#        try:
#            job_info = m.get_job_info(job_id)
#            state = job_info.status_meta["code_complete"]
#        except Exception:
#            click.echo(
#                f"{u.WARNING} Could not query job {job_id} ({cfg_id}), skipping.{u.RESET}"
#            )
#            skipped.append({"id": job_id, "cfg_id": cfg_id, "reason": "query_failed"})
#            continue
#
#        if state in ("running", "pending"):
#            try:
#                result = subprocess.run(
#                    ["scancel", job_id],
#                    capture_output=True,
#                    text=True,
#                )
#                if result.returncode == 0:
#                    cancelled.append({"id": job_id, "cfg_id": cfg_id, "state": state})
#                    click.echo(
#                        f"{u.SUCCESS_HEAVY} {u.GREEN}Cancelled {job_id} ({cfg_id}) [{state}]{u.RESET}"
#                    )
#                else:
#                    skipped.append(
#                        {
#                            "id": job_id,
#                            "cfg_id": cfg_id,
#                            "reason": result.stderr.strip(),
#                        }
#                    )
#                    click.echo(
#                        f"{u.FAILURE_HEAVY} {u.RED}Failed to cancel {job_id} ({cfg_id}): {result.stderr.strip()}{u.RESET}"
#                    )
#            except Exception as e:
#                skipped.append({"id": job_id, "cfg_id": cfg_id, "reason": str(e)})
#                click.echo(
#                    f"{u.FAILURE_HEAVY} {u.RED}Exception cancelling {job_id} ({cfg_id}): {e}{u.RESET}"
#                )
#        else:
#            skipped.append(
#                {
#                    "id": job_id,
#                    "cfg_id": cfg_id,
#                    "state": state,
#                    "reason": "not_running_or_pending",
#                }
#            )
#            click.echo(f"{u.INFO} {job_id} ({cfg_id}) is [{state}], skipping.{u.RESET}")
#
#    click.echo(
#        f"\n{u.POINT_DIAMOND} Summary: {u.GREEN}{len(cancelled)} cancelled{u.RESET}, {u.YELLOW}{len(skipped)} skipped{u.RESET}"
#    )


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Interactive mode — type a command and press Enter:

  run          Submit new benchmark jobs (will prompt for arguments)
  rerun        Rerun failed/pending jobs (will prompt for arguments)
  status       Check status of a run (will prompt for arguments)
  cancel       Cancel jobs of running/pending jobs of a run (will prompt for arguments)
  help         Show this help
  quit/exit    Exit

You can also pass flags directly, e.g.:  run --dry-run
Or let the interactive prompts guide you:  run
"""


def interactive_loop(subcmd: str = ""):
    """Run an interactive REPL loop for the CLI commands."""
    click.echo(f"{u.YELLOW}MINERVA benchmarks CLI — interactive mode{u.RESET}")
    click.echo(
        f"{u.YELLOW}Type 'help' for available commands, 'quit' to exit.\n{u.RESET}"
    )

    while True:
        try:
            if subcmd == "":
                user_input = read_user_input()
            else:
                user_input = subcmd
                subcmd = ""
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            click.echo("Bye.")
            break

        if user_input.lower() == "help":
            click.echo(INTERACTIVE_HELP)
            continue

        # Parse the command and its arguments
        parts = user_input.split()
        if len(parts) > 1:
            click.echo(
                f"{u.RED}Unknown command: '{' '.join(parts)}'. Type 'help' for options.{u.RESET}\n"
            )
            continue
        cmd_name = parts[0].lower()

        if cmd_name not in subcommands_list(cli):
            click.echo(
                f"{u.RED}\tUnknown command: '{cmd_name}'! Type 'help' for options.{u.RESET}\n"
            )
            continue

        # Collect arguments interactively based on command
        extra_args = []
        try:
            if cmd_name == "run":
                click.echo("\n  -- run options --")
                extra_args.extend(prompt_options_interactive(RUN_OPTIONS))

            elif cmd_name == "rerun":
                click.echo("\n  -- re-run options --")
                extra_args.extend(prompt_options_interactive(RERUN_OPTIONS))

            elif cmd_name == "status":
                click.echo("\n  -- status options --")
                extra_args.extend(prompt_options_interactive(STATUS_OPTIONS))
            elif cmd_name == "cancel":
                click.echo("\n  -- cancel options --")
                extra_args.extend(prompt_options_interactive(CANCEL_OPTIONS))
        except EOFError:
            click.echo("\nCancelled.")
            continue

        # Build a Click context and invoke the command
        cmd_map = {"run": run, "rerun": rerun, "status": status, "cancel": cancel}
        _cmd_mao = command_tree(cli)
        cmd = cmd_map[cmd_name]
        try:
            # Use sys.argv temporarily so Click can parse the sub-command args
            old_argv = sys.argv
            sys.argv = ["argv0", *extra_args]
            try:
                cmd(standalone_mode=False)
            except Exception as e:
                click.echo(f"Error: {e}", err=True)
            finally:
                sys.argv = old_argv
        except Exception as e:
            click.echo(f"Error: {e}", err=True)

        click.echo()  # blank line between commands


def cli_entry():
    """Entry point — if no args given, enter interactive mode."""
    # if len(sys.argv) == 1:
    #    interactive_loop()
    # elif len(sys.argv) == 2:
    #    if sys.argv[1].strip() in ["help", "--help", "-h"]:
    #        sys.argv[1] = "--help"
    #        cli()
    #    interactive_loop(sys.argv[1])
    # else:
    #
    cli()


def command_tree(obj):

    if isinstance(obj, click.Group):
        return {name: value for name, value in obj.commands.items()}


def subcommands_list(obj) -> list[str]:
    cmd_tree = command_tree(obj)
    return list(cmd_tree.keys())


if __name__ == "__main__":
    cli_entry()

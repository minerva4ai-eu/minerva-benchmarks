# benchmark/cli.py
import os
import subprocess as s
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import configs_hydra.model_framework_dataset as mfd
import scripts.slurm.utils as u
from configs_hydra.hydra_app import generate_valid_combos
from scripts.slurm.submitter import submit_job, write_config

if TYPE_CHECKING:
    from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
# )

import logging

RUNS_DIR = Path("benchmark-runs/")
DEFAULT_CONFIGS_PATH = "./configs_hydra/configs"
DEFAULT_CONFIG_NAME = "base"
BASE_DIR = Path(".")
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

    # TODO: review output structure
    _runs_dir = runs_dir
    runs_dir = f"{_runs_dir}-{config_name}"
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

            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Arguments '--yaml' and '--dry-run' cannot be combined...{u.RESET}"
            )
            exit(1)
        config_names = set()
        for y in yamls:
            _config_name = [s for s in y.split("/") if s.find(_runs_dir) != -1]

            assert len(_config_name) == 1, (
                f"{u.FAILURE_HEAVY}{u.RED}Could not extract config-name from provided YAML config: {u.YELLOW}'{y}'{u.RESET}"
            )
            _config_name = str(_config_name[0]).replace(f"{_runs_dir}-", "")
            config_names.add(_config_name)
            assert len(config_names) == 1, (
                f"{u.FAILURE_HEAVY}{u.RED} Only provide paths to YAML configs of the same '--config-name' profile! Found: {u.YELLOW}[{config_names}]{u.RESET}"
            )
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
        config_name = next(iter(config_names))
        runs_dir = f"{_runs_dir}-{config_name}"
    else:
        # TODO: Check desired behavior
        if config_name == DEFAULT_CONFIG_NAME:
            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED}!WARNING! Argument '--config-name' is defaulting to '{DEFAULT_CONFIG_NAME}'...{u.RESET}"
            )
            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Make sure to provide the correct '--config-name' pointing to a .yaml configuration profile inside {DEFAULT_CONFIGS_PATH}{u.RESET}"
            )
            exit(1)
        # Filter benchmarks to run on provided model names
        if models:
            models = models.split(",")
            if len(models) == 0:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Model names provided must be separated by comma ','. Provided: '{models}'{u.RESET}"
                    + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.{u.RESET}"
                )
                sys.exit(1)
            for m in models:
                if m not in mfd.MODELS:
                    click.echo(
                        f"\t{u.FAILURE_HEAVY} {u.RED} Model '{m}' could not be found in available models.{u.RESET}"
                        + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.{u.RESET}"
                    )
                    sys.exit(1)
            mfd.MODELS = models

        # Filter benchmarks to run on provided framework names
        if frameworks:
            frameworks = frameworks.split(",")
            if len(frameworks) == 0:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Framework names provided must be separated by comma ','. Provided: '{frameworks}'{u.RESET}"
                    + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.{u.RESET}"
                )
                sys.exit(1)
            for m in frameworks:
                if m not in mfd.FRAMEWORKS:
                    click.echo(
                        f"\t{u.FAILURE_HEAVY} {u.RED} Framework '{m}' could not be found in available frameworks.{u.RESET}"
                        + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.{u.RESET}"
                    )
                    sys.exit(1)
            mfd.FRAMEWORKS = frameworks

        # Filter benchmarks to run on provided dataset names
        if datasets:
            datasets = datasets.split(",")
            if len(datasets) == 0:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Dataset names provided must be separated by comma ','. Provided: '{datasets}'{u.RESET}"
                    + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.{u.RESET}"
                )
                sys.exit(1)
            for m in datasets:
                if m not in mfd.DATASETS:
                    click.echo(
                        f"\t{u.FAILURE_HEAVY} {u.RED} Dataset '{m}' could not be found in available datasets.{u.RESET}"
                        + f"\n\t{u.FAILURE_HEAVY} {u.YELLOW} Run 'bash minerva-cli.sh run --help' for more information.{u.RESET}"
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


@cli.command()
@click.option(
    "--jobids",
    type=str,
    required=True,
    help=(
        "Comma separated ',' jobs ids to collect training results from and export into output directory."
    ),
)
@click.option(
    "--output",
    type=str,
    default="",
    help=(
        "Output file to write collected results. Default: analytics/results/training_summary_jobs_jobid1-jobid2-...-jobidN.csv"
    ),
)
@click.option(
    "--root",
    type=str,
    default="",
    help="Root directory containing machine/date/job results (default: script default).",
)
@click.argument("extra_jobids", nargs=-1)
def results(jobids, output, root, extra_jobids):
    """Generate benchmark results per jobids (comma or space separated)."""
    raw = ",".join([jobids, *extra_jobids]).replace(" ", ",")
    ids = [j for j in raw.split(",") if j]
    if not ids:
        raise click.UsageError("--jobids must contain at least one job id.")
    cmd = [sys.executable, "-m", "analytics.generateSummaryTable_jobs", *ids]
    if output:
        cmd.extend(["--output", output])
    if root:
        cmd.extend(["--root", root])
    # Script imports configs_hydra/scripts, so run from the training dir.
    training_dir = Path(__file__).resolve().parents[2]
    proc = s.run(cmd, cwd=training_dir)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


@cli.command()
@click.option(
    "--configs-path",
    default=DEFAULT_CONFIGS_PATH,
    help="Path to the Hydra config directory (default: ./configs_hydra/configs).",
)
@click.option(
    "--config-name",
    required=True,
    help="Base config name to compose (e.g., 'base-MN5').",
)
@click.option(
    "--runs-dir",
    default=RUNS_DIR,
    help="Output directory for generated configs and results (default: benchmark-runs/).",
)
@click.option(
    "--models",
    type=str,
    default="",
    help="Comma separated model names to prepare (default: all supporting megatron-nemo-2509).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Compose configs and list the jobs without submitting them.",
)
def prepare(configs_path, config_name, runs_dir, models, dry_run):
    # TODO: Check desired behavior
    if config_name == DEFAULT_CONFIG_NAME:
        click.echo(
            f"\t{u.FAILURE_HEAVY} {u.RED}!WARNING! Argument '--config-name' is defaulting to '{DEFAULT_CONFIG_NAME}'...{u.RESET}"
        )
        click.echo(
            f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Make sure to provide the correct '--config-name' pointing to a .yaml configuration profile inside {DEFAULT_CONFIGS_PATH}{u.RESET}"
        )
        exit(1)
    """Submit one SLURM job running only the megatron-nemo-2509 preparation stage for every model/dataset."""
    megatron_prepare(configs_path, config_name, runs_dir, models, dry_run)


def megatron_prepare(configs_path, config_name, runs_dir, models, dry_run):
    from copy import deepcopy

    from configs_hydra.dataclasses_hydra import register_configs
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    framework = "megatron-nemo-2509"
    selected = [m.strip() for m in models.split(",") if m.strip()]
    for m in selected:
        if m not in mfd.MODELS:
            click.echo(
                f"{u.FAILURE_HEAVY} {u.RED} Unknown model '{m}'. Valid: {mfd.MODELS}{u.RESET}"
            )
            sys.exit(1)

    runs_dir = f"{runs_dir}-{config_name}"
    run_date = datetime.now().date().strftime("%d-%m-%Y")

    register_configs()
    GlobalHydra.instance().clear()

    cfgs: list[BenchmarkConfig] = []
    with initialize_config_dir(
        config_dir=os.path.abspath(configs_path), version_base="1.3"
    ):
        base = compose(config_name)
        for model in selected or mfd.MODELS:
            for dataset in mfd.DATASETS:
                cfg = compose(
                    config_name,
                    overrides=[
                        f"model={model}",
                        f"framework={framework}",
                        f"dataset={dataset}",
                        f"slurm={base.machine.name_pattern}",
                        f"arch={base.machine.name_pattern}",
                    ],
                )
                if framework not in cfg.model.frameworks_supported:
                    continue
                if dataset not in cfg.framework.datasets_allowed:
                    continue

                cfg = deepcopy(cfg)
                comb = cfg.model.combinations
                tr = cfg.model.training
                tr.global_batch_size = comb.global_batch_sizes[0]
                tr.batch_size = comb.batch_sizes[0]
                tr.steps = comb.steps[0]
                tr.gradient_checkpointing = comb.gradient_checkpointing[0]
                tr.max_model_length = comb.max_seq_lens[0]
                tr.precision = comb.precisions[0]
                tr.grad_accum = 1

                # Preparation only needs one GPU; parallelism is irrelevant here.
                par = cfg.framework.megatron_parallelism
                par.tp = par.pp = par.cp = par.dp = par.ep = 1
                par.sp = False

                cfg.slurm.sbatch.nodes = 1
                cfg.experiment.output_dir = runs_dir
                cfg.experiment.yaml_filename = f"prepare-{model}-{dataset}.yaml"
                cfg.id = f"{cfg.machine.name}_{model}_{framework}_{dataset}_prepare"
                OmegaConf.resolve(cfg)
                cfgs.append(cfg)

    if not cfgs:
        click.echo(
            f"{u.FAILURE_HEAVY} {u.RED} No models support '{framework}'.{u.RESET}"
        )
        return

    # Read by run-nemo_megatron.slurm: exit after the preparation stage.
    os.environ["PREPARE_ONLY"] = "1"

    cfgs_paths = []
    for cfg in cfgs:
        cfg_path = write_config(
            cfg=cfg, base_dir=str(BASE_DIR.absolute()), runs_dir=runs_dir
        )
        cfgs_paths.append(cfg_path)
        click.echo(
            f"{u.ARROW_SUB_ITEM}{u.CYAN} Prepare {u.YELLOW}{cfg.model.name}{u.CYAN} / {u.YELLOW}{cfg.dataset.name}{u.RESET}"
        )
        click.echo(f"\t{u.POINT_BULLET} {cfg_path}")
    if dry_run:
        return

    submit_job(
        runs_dir=runs_dir,
        cfgs=cfgs,
        cfgs_paths=cfgs_paths,
        run_date=run_date,
        nnodes=1,
    )


def command_tree(obj):

    if isinstance(obj, click.Group):
        return {name: value for name, value in obj.commands.items()}


def subcommands_list(obj) -> list[str]:
    cmd_tree = command_tree(obj)
    return list(cmd_tree.keys())


if __name__ == "__main__":
    cli()

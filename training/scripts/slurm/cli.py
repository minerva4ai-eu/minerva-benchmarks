# benchmark/cli.py
import os
import re
import subprocess as s
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import configs_hydra.model_framework_dataset as mfd
import dotenv
import scripts.slurm.utils as u
from configs_hydra.hydra_app import (
    PROFILES_ROOT,
    generate_valid_combos,
    get_dir_profiles,
)
from scripts.slurm.submitter import submit_job, write_config

if TYPE_CHECKING:
    from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
# )

import logging

# Load training/.env
dotenv.load_dotenv()

RUNS_DIR = Path("benchmark-runs/")
DEFAULT_CONFIGS_PATH = "./configs_hydra/configs"
DEFAULT_CONFIG_NAME = "base"
BASE_DIR = Path(".")
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
# LOG_DIR = os.path.join("outputs", "logs", "pycli", TIMESTAMP)
# LOG_DIR = os.environ["LOG_DIR"]
# LOG_FILE = os.path.join(LOG_DIR, "minerva.log")
LOG_FILE = os.environ["LOG_FILE"]
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


def _submit_group(
    config_name,
    configs_path,
    runs_dir,
    run_date,
    cfgs_valid,
    cfgs_paths,
    cfgs_runs_dirs,
    nnodes,
    per_model_jobs,
    per_nodes_jobs,
):
    """Submit one SLURM job (or one per model/nodes when requested) for the given configs.

    `runs_dir` holds the job-level logs; `cfgs_runs_dirs` gives, per config, the runs_dir
    (one per profile) where its results are written."""
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
        for cfg, path, rd in zip(cfgs_valid, cfgs_paths, cfgs_runs_dirs):
            if not cfg.model.name in per_model_cfgs:
                per_model_cfgs[cfg.model.name] = {}
                per_model_cfgs[cfg.model.name]["cfgs_valid"] = []
                per_model_cfgs[cfg.model.name]["cfgs_paths"] = []
                per_model_cfgs[cfg.model.name]["cfgs_runs_dirs"] = []

            per_model_cfgs[cfg.model.name]["cfgs_valid"].append(cfg)
            per_model_cfgs[cfg.model.name]["cfgs_paths"].append(path)
            per_model_cfgs[cfg.model.name]["cfgs_runs_dirs"].append(rd)

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
                cfgs_runs_dirs=combinations["cfgs_runs_dirs"],
            )
        return
    if per_nodes_jobs:
        per_nodes_cfgs: dict[int, dict[str, list[BenchmarkConfig | str]]] = {}
        for cfg, path, rd in zip(cfgs_valid, cfgs_paths, cfgs_runs_dirs):
            if not cfg.slurm.sbatch.nodes in per_nodes_cfgs:
                per_nodes_cfgs[cfg.slurm.sbatch.nodes] = {}
                per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_valid"] = []
                per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_paths"] = []
                per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_runs_dirs"] = []

            per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_valid"].append(cfg)
            per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_paths"].append(path)
            per_nodes_cfgs[cfg.slurm.sbatch.nodes]["cfgs_runs_dirs"].append(rd)

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
                cfgs_runs_dirs=combinations["cfgs_runs_dirs"],
            )
        return

    click.echo(f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Preparing jobs ... {u.RESET}")
    jobid = submit_job(
        runs_dir=runs_dir,
        cfgs=cfgs_valid,
        cfgs_paths=cfgs_paths,
        run_date=run_date,
        nnodes=nnodes,
        cfgs_runs_dirs=cfgs_runs_dirs,
    )
    for rd in dict.fromkeys(cfgs_runs_dirs):
        click.echo(
            f"\n\t{u.EMOJI_INFO}   Results in: "
            f"{u.YELLOW}'{os.path.join(rd, cfgs_valid[0].machine.name, run_date, jobid)}'{u.RESET}"
        )


def expand_profiles(config_path: str, profile: str | None) -> list[str | None]:
    """Expand a --profile value into the list of profile names to run.

    Accepts a profile name or the name of a directory
    inside PROFILES_ROOT (every *.yaml below it, recursively, is a profile named
    '<dir>/<relative path without extension>'). Returns [None] when no profile is given.
    """
    if not profile:
        return [None]
    if len(profile.split(",")) > 1:
        profiles = []
        mult_profiles = profile.split(",")
        for p in mult_profiles:
            profiles.extend(expand_profiles(config_path, p))
        return profiles

    root = os.path.realpath(os.path.join(os.path.abspath(config_path), PROFILES_ROOT))
    profiles: list[str | None] = []

    item = profile.strip()
    path = os.path.realpath(os.path.join(root, item))

    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"Profile '{item}' must be inside {root}")
    if os.path.isfile(path + ".yaml"):
        profiles.append(item.removesuffix(".yaml"))
    elif os.path.isdir(path):
        found = sorted(
            os.path.relpath(os.path.join(d, f), root).rstrip(".yaml").rstrip(".yml")
            for d, _, files in os.walk(path)
            for f in files
            if (f.endswith(".yaml") or f.endswith("yml"))
        )
        if not found:
            raise FileNotFoundError(f"No profiles (*.yaml) found in {path}")
        profiles.extend(found)
    else:
        # available = sorted(
        #    str(p.relative_to(root).with_suffix(""))
        #    for p in Path(root).rglob("*")
        #    if p.suffix in (".yaml", ".yml")
        # )
        available = get_dir_profiles(config_path)
        avail_str = "\n\t* " + "\n\t* ".join(available)
        raise FileNotFoundError(
            f"Profile '{item}' not found in {root}. Available: {avail_str}"
        )
    profiles = list(dict.fromkeys(profiles))
    if not profiles:
        raise ValueError(f"No profile names found in '{profile}'")
    return profiles


def profile_is_file(p):
    return os.path.isfile(f"{p}.yaml") or os.path.isfile(f"{p}.yml")


def profile_root(p):
    if not p:
        return p
    return p.split("/")[0]


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
    "--profile",
    default=None,
    help=(
        "Benchmark profile(s) from configs_hydra/configs/PROFILES_ROOT/ merged on top of provided '--config-name'. "
        "Selects models/frameworks/datasets and training combinations. Accepts one specified profile name or "
        "a directory name to run all the profiles inside it (e.g. 'tests'). "
        "Multiple profiles can be provided comma-separated, e.g. '--profile prof1,prof2' or "
        "'--profile dir1,dir2' to run all profiles inside each directory.\n"
        "In case of specifying a single profile to run, use only the name omitting the .yaml extension, just like "
        "arguement '--config-name'."
    ),
)
@click.option(
    "--yaml",
    "yamls",
    multiple=True,
    default=None,
    help=(
        "Run a benchmark configuration by providing the path to a BenchmarkConfig YAML file. "
        "Can be repeated for multiple configs, e.g. '--yaml path1.yaml --yaml path2.yaml', "
        "or provided comma-separated in a single value, e.g. '--yaml path1.yaml,path2.yaml'. "
        "First run a '--dry-run' to compose YAML configuration files and then use their paths to run them individually."
    ),
)
@click.option(
    "--nnodes",
    default=None,
    type=int,
    help="Number of nodes of the job (cannot be used with multiple profiles).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help=(
        "Verbose mode: prints extra progress messages (profile/triple counters, "
        "written YAML paths, per-group submission progress) and suppresses "
        "constraint-rule details (per-rule failure reasons and the Sweep Summary table)."
    ),
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
    profile,
    yamls,
    nnodes,
    verbose,
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
    runs_dir = _runs_dir
    run_date = datetime.now().date().strftime("%d-%m-%Y")

    cfgs_valid: list[BenchmarkConfig] = []
    cfgs_paths: list[str] = []
    # (runs_dir, configs, config paths) to submit
    groups: list[tuple[str, list, list]] = []

    if isinstance(yamls, tuple) and len(yamls) == 1:
        yamls: list[str] = [y.strip() for y in yamls[0].split("--yaml") if y != ""]
    # Also accept comma-separated paths within a single --yaml value
    yamls = [y.strip() for part in yamls for y in part.split(",") if y.strip()]
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
                if y in cfgs_paths:
                    click.echo(
                        f"\t{u.SKIPPED}{u.MAGENTA} !YAML path already processed! Skipping...{u.RESET}"
                    )
                    continue
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

        if verbose:
            click.echo(
                f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Loaded {u.BLUE}{len(cfgs_valid)}{u.CYAN} YAML config(s) for submission{u.RESET}"
            )

        start_sub = PROFILES_ROOT
        end_sub = cfgs_valid[0].machine.name

        # Check if YAML paths belong to a specific profile or no-profile runs
        # Parent profile directory pattern in path is everything between
        # PROFILES_ROOT and the machine name
        pattern = re.escape(start_sub) + r"(.*?)" + re.escape(end_sub)
        match = re.search(pattern, y)
        profiles = set()
        no_profile = []

        _runs_dir = f"{_runs_dir}-{config_name}"
        runs_dir = f"{_runs_dir}/no-profile"
        for y in yamls:
            match = re.search(pattern, y)
            if match:
                # match.group(0) includes both substrings
                # To include the first and exclude the second, include group(1) and the start_sub
                result = start_sub + match.group(1)

                profiles.add(result)
                continue
            no_profile.append(y)

        profiles = list(profiles)
        if len(profiles) >= 1 and profiles[0] is not None:
            if len(profiles) > 1:
                profiles_print = [
                    p.replace(PROFILES_ROOT, "").removeprefix("/").removesuffix("/")
                    for p in profiles
                ]
                click.echo(
                    f"{u.FAILURE_HEAVY}{u.RED} Detected yamls from multiple profiles, which is not allowed: {u.CYAN}{profiles_print}{u.RESET}!"
                    f"{u.RED}\n Make sure to provide yamls from a single profile only.{u.RESET}"
                )
                exit(1)
            profile_name = profiles[0]
            if profiles and no_profile:
                click.echo(
                    f"{u.FAILURE_HEAVY}{u.RED} Detected yamls from both configuration profile ('{profile_name}') and no-profile runs, which is not allowed! "
                    f"Make sure to provide yamls from a single profile or from no-profile runs only.{u.RESET}"
                )
                exit(1)
            # In case of directory with profiles
            profile_name = "/".join(profiles[0].split("/")[:-1])
            # In case no profiles or a single .yaml profile are provided
            # use the full
            runs_dir = (f"{_runs_dir}/{profile_name}").rstrip("/")
        groups.append((runs_dir, cfgs_valid, cfgs_paths, [runs_dir] * len(cfgs_valid)))
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

        try:
            profiles = expand_profiles(configs_path, profile)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"\t{u.FAILURE_HEAVY} {u.RED}{e}{u.RESET}")
            sys.exit(1)

        if verbose:
            click.echo(
                f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Resolved {u.BLUE}{len(profiles)}{u.CYAN} profile(s) to compose: {u.RESET}"
            )
            for p in profiles:
                click.echo(
                    f"\t{u.POINT_BULLET}{u.YELLOW} {p if p else '(no profile)'}{u.RESET}"
                )

        # parent_profiles = ["/".join(p.split("/")[:-1]) for p in profiles if p]
        profile_roots = [profile_root(p) for p in profiles if p]
        if len(set(profile_roots)) > 1:
            # common_root = os.path.commonprefix(parent_profiles)
            # if not common_root:
            click.echo(
                f"\n{u.ARROW_SUB_ITEM}{u.RED} Multiple profile roots detected: {u.YELLOW}{', '.join(set(profile_roots))}{u.RESET}"
            )
            click.echo(
                f"{u.RED} Running profiles different roots is not allowed!{u.RESET}"
            )
            exit(1)
        if nnodes and (per_nodes_jobs or per_model_jobs):
            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Argument '--nnodes' cannot be used with '--per-nodes-jobs' or '--per-model-jobs' simultaneously.{u.RESET}"
            )
            sys.exit(1)
        # if len(profiles) > 1 and nnodes and per_profile_jobs:
        #    click.echo(
        #        f"\t{u.FAILURE_HEAVY} {u.RED}!ERROR! Argument '--nnodes' cannot be used with multiple profiles: {profiles}{u.RESET}"
        #    )
        #    sys.exit(1)
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

        _runs_dir = f"{_runs_dir}-{config_name}"
        profile_name = ""
        # Experiments' profile has been detected/provided
        if len(profiles) >= 1 and profiles[0] is not None:
            # if len(profiles) > 1:
            #    click.echo(
            #        f"{u.FAILURE_HEAVY}{u.RED} Detected yamls from multiple profiles, which is not allowed: {u.CYAN}{profiles}{u.RESET}"
            #        f"{u.RED} Make sure to provide yamls from a single profile only.{u.RESET}"
            #    )
            #    exit(1)
            # In case of directory with profiles

            # Taking care of the case where
            _profiles = [
                "/".join(p.split("/")[:-1]) if profile_is_file(p) else p
                for p in profiles
            ]
            profile_name = os.path.commonpath(
                _profiles,
            )
            # In case no profiles or a single .yaml profile are provided
            # use the full
            if len(profiles) == 1:
                profile_name = profiles[0]
                if os.path.isfile(f"{profile_name}.yaml") or os.path.isfile(
                    f"{profile_name}.yaml"
                ):
                    profile_name = "/".join(profiles[0].split("/")[:-1])
            profile_name = profile_root(profile_name)
        runs_dir = (
            f"{_runs_dir}/{'profiles' if profile_name else 'no-profile'}/{profile_name}"
        ).rstrip("/")
        for prof_idx, prof in enumerate(profiles, start=1):
            if prof:
                if verbose:
                    click.echo(
                        f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Composing profile {u.YELLOW}'{prof}' "
                        f"({u.BLUE}{prof_idx}/{len(profiles)}{u.CYAN}){u.RESET}"
                    )
                else:
                    click.echo(
                        f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Composing profile {u.YELLOW}'{prof}'{u.RESET}"
                    )
            prof_cfgs, _ = generate_valid_combos(
                config_path=configs_path,
                config_name=config_name,
                outpath=runs_dir,
                run_date=run_date,
                dry=dry_run,
                profile=prof,
                verbose=verbose,
            )

            os.makedirs(runs_dir, exist_ok=True)
            prof_paths = [
                write_config(
                    cfg=cfg, base_dir=str(BASE_DIR.absolute()), runs_dir=runs_dir
                )
                for cfg in prof_cfgs
            ]
            if verbose:
                click.echo(
                    f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Wrote {u.BLUE}{len(prof_paths)}{u.CYAN} config YAML(s) for profile {u.YELLOW}'{prof if prof else '(no profile)'}'{u.RESET}"
                )
                for p in prof_paths:
                    click.echo(f"\t{u.POINT_BULLET}{u.YELLOW} {p}{u.RESET}")
            groups.append(
                (runs_dir, prof_cfgs, prof_paths, [runs_dir] * len(prof_cfgs))
            )

        if dry_run:
            return

    if len(groups) > 1:
        click.echo(
            f"{u.MAGENTA}>> Grouping multiple profiles into a single job: {u.BLUE}{len(groups)}{u.RESET}"
        )
        click.echo(
            f"{u.MAGENTA}   With respective amount of .yaml files per group: {u.BLUE}{[len(cfgs) for _, cfgs, _, _ in groups]}{u.RESET}"
        )
        click.echo(
            f"{u.MAGENTA}   Total number of experiments/Slurm steps: {u.BLUE}{sum([len(cfgs) for _, cfgs, _, _ in groups])}{u.RESET}"
        )

        click.echo(
            f"{u.MAGENTA}   Group common runs dir: {u.YELLOW}{runs_dir}{u.RESET}"
        )

        groups = [
            (
                runs_dir,
                [cfg for _, cfgs, _, _ in groups for cfg in cfgs],
                [path for _, _, paths, _ in groups for path in paths],
                [rd for _, _, _, rds in groups for rd in rds],
            )
        ]

    for group_idx, (
        group_runs_dir,
        group_cfgs,
        group_paths,
        group_runs_dirs,
    ) in enumerate(groups, start=1):
        if verbose:
            click.echo(
                f"\n{u.ARROW_SUB_ITEM}{u.CYAN} Submitting job for group "
                f"{u.BLUE}{group_idx}/{len(groups)}{u.CYAN} with "
                f"{u.BLUE}{len(group_cfgs)}{u.CYAN} config(s) -> runs dir: {u.YELLOW}{group_runs_dir}{u.RESET}"
            )
        _submit_group(
            config_name,
            configs_path,
            group_runs_dir,
            run_date,
            group_cfgs,
            group_paths,
            group_runs_dirs,
            nnodes,
            per_model_jobs,
            per_nodes_jobs,
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
    """Prepare things for benchmarks to run by launching a slurm job to preprocess models/datasets"""
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


def megatron_prepare(
    configs_path,
    config_name,
    runs_dir,
    models,
    dry_run,
):
    from copy import deepcopy

    from configs_hydra.dataclasses_hydra import register_configs
    from configs_hydra.hydra_app import resolve_config_name
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

    config_name = resolve_config_name(configs_path, config_name)
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
                    config_name=config_name,
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
                tr.attention_mechanism = comb.attention_mechanisms[0]
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

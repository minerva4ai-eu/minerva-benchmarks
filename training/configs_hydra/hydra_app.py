import os
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product

from configs_hydra.constraints import rules
from configs_hydra.dataclasses_hydra import BenchmarkConfig, register_configs
from configs_hydra.model_framework_dataset import *

# sweep/generator.py
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

# Color Codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"

# Style Codes
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
BG_RED = "\033[41m"

# Reset Code (turns colors back to normal)
RESET = "\033[0m"


console = Console()


def single_gpu_config(cfg: BenchmarkConfig) -> bool:
    """Replicates your GPU_CONFIGS=(1 $GPUS_PER_NODE)
    logic for experiments on 1 node that require also
    single GPU run"""
    if (
        cfg.machine.single_gpu_also_valid
        and (cfg.slurm.sbatch.nodes == 1)
        and (cfg.framework.parallelism[cfg.framework.parallelism_name]["min_gpus"] == 1)
    ):
        return True
    return False


def generate_valid_combos(
    config_path: str, config_name: str, outpath: str
) -> tuple[list[BenchmarkConfig], list[rules.RuleResult]]:
    valid, skipped = [], []

    register_configs()
    GlobalHydra.instance().clear()

    # outpath = os.path.join(outpath, "bencharks_to_run")
    os.makedirs(outpath, exist_ok=True)

    cfg_seen = set()
    with initialize_config_dir(
        config_dir=os.path.abspath(config_path), version_base="1.3"
    ):
        raw_total = 0
        for model, framework, dataset in product(MODELS, FRAMEWORKS, DATASETS):
            _init_cfg: BenchmarkConfig = compose(
                config_name,
            )

            assert framework in AVAILABLE_FRAMEWORKS, (
                f"{RED}Framework {YELLOW}'{framework}'{RED} not found! Available frameworks: \n{BLUE}[{', '.join(AVAILABLE_FRAMEWORKS)}]{RESET}"
            )

            cfg: BenchmarkConfig = compose(
                config_name,
                overrides=[
                    f"model={_init_cfg.machine.name_pattern}/{model}-{_init_cfg.machine.name_pattern}",
                    f"framework={framework}",
                    f"dataset={dataset}",
                    f"slurm={_init_cfg.machine.name_pattern}",
                    f"arch={_init_cfg.machine.name_pattern}",
                ],
            )
            print(
                f"Composed base configuration from {os.path.abspath(f'{config_name}.yaml')} for:"
                + f"\n\t· model: {model}"
                + f"\n\t· framework: {framework}"
                + f"\n\t· dataset: {dataset}"
            )

            if framework not in cfg.model.frameworks_supported:
                print(
                    f"{YELLOW}\tFramework '{framework}' is not supported by model '{model}'! Skipping...{RESET}"
                )
                continue

            if dataset not in cfg.framework.datasets_allowed:
                print(
                    rf"{YELLOW}Dataset '{dataset}' is not supported by framework '{framework}'! Skipping...{RESET}"
                )
                continue

            if cfg.framework.megatron_parallelism:
                cfg_seen, valid, skipped, raw_total = _multi_parallelism_framework(
                    cfg, outpath, cfg_seen, valid, skipped, raw_total
                )

            else:
                cfg_seen, valid, skipped, raw_total = _single_parallelism_framework(
                    cfg, outpath, cfg_seen, valid, skipped, raw_total
                )

        _print_summary(raw_total, valid, skipped)
    return valid, skipped


def _single_parallelism_framework(
    cfg: BenchmarkConfig,
    outpath: str,
    cfg_seen: set,
    valid: list,
    skipped: list,
    raw_total: int,
):
    """Specific to frameworks torchrun, accelerate, deepspeed-accelerate, deepspeed

    Args:
        cfg (BenchmarkConfig): _description_
        outpath (str): _description_
        cfg_seen (set): _description_
        valid (list): _description_
        skipped (list): _description_
        raw_total (int): _description_

    Returns:
        _type_: _description_
    """
    for parallelism in cfg.model.parallelism_supported:
        tmp_cfg = deepcopy(cfg)

        if parallelism not in cfg.framework.parallelism.keys():
            continue
        # print(f"cfg.framework.parallelism: \n{cfg.framework.parallelism}")
        # print(f"parallelism: \n{parallelism}")
        parallelism_spex = cfg.framework.parallelism[parallelism]

        target_parallelism = OmegaConf.create(
            DictConfig({parallelism: parallelism_spex})
        )
        # print(f"target_parallelism: \n{target_parallelism}")
        # print(f"cfg.framework.parallelism: \n{cfg.framework.parallelism}")
        tmp_cfg.framework.parallelism_name = parallelism
        tmp_cfg.framework.parallelism = target_parallelism
        # print(f"\tCreating configuration for parallelism {parallelism}:")
        # print(OmegaConf.to_yaml(cfg))

        for (
            bs,
            grad_acc,
            precision,
            lr,
            optimizer,
            steps,
            enable_compile,
        ) in product(
            cfg.model.combinations.batch_sizes,
            cfg.model.combinations.grad_accums,
            cfg.model.combinations.precisions,
            cfg.model.combinations.lr,
            cfg.model.combinations.optimizer,
            cfg.model.combinations.steps,
            cfg.model.combinations.enable_compile,
        ):
            # Replace combinations from cfg.trainings* into tmp_cfg.model.training.*
            # to each experiment combination
            tmp_cfg.model.training.batch_size = bs
            tmp_cfg.model.training.grad_accum = grad_acc
            tmp_cfg.model.training.precision = precision
            tmp_cfg.model.training.lr = lr
            tmp_cfg.model.training.optimizer = optimizer
            tmp_cfg.model.training.steps = steps
            tmp_cfg.model.training.enable_compile = enable_compile
            tmp_cfg.experiment.output_dir = outpath
            # Will bee used later to take care of configuration
            # of 1 node and 1 gpu

            # Get min number of nodes to run based on model and hpc architecture
            min_nodes = rules.MinNodesMemoryRule()._min_nodes_required(tmp_cfg)
            candidate_nodes = rules.MinNodesMemoryRule()._nodes_candidates(
                tmp_cfg.arch.node.gpus_per_node,
                max_gpus_scale=tmp_cfg.model.max_gpus_scale,
            )
            nodes_to_run = [n for n in candidate_nodes if n >= min_nodes]

            for nodes in nodes_to_run:
                tmp_cfg.slurm.sbatch.nodes = nodes

                parameters_combo = f"{cfg.machine.name}/{cfg.model.name}/{cfg.framework.name}/{cfg.dataset.name}/nodes-{nodes}"

                tmp_cfg.slurm.sbatch.chdir = os.path.join(
                    tmp_cfg.experiment.output_dir, parameters_combo
                )

                experiment_parameters = (
                    f"bs{bs}"
                    + f"-grad_accum{grad_acc}"
                    + f"-compile{enable_compile}"
                    + f"-prec{precision}"
                    + f"-steps{steps}"
                )

                # By default, if config is using 1 node,
                # all cfg.arch.node.gpus_per_node will ne utilized
                # Unless, perallelism defines min_gpus = 1
                if single_gpu_config(tmp_cfg):
                    tmp_cfg.slurm.sbatch.gpus_per_node = 1
                    tmp_cfg.slurm.sbatch.gres = "gpu:1"

                total_gpus = (
                    tmp_cfg.slurm.sbatch.gpus_per_node * tmp_cfg.slurm.sbatch.nodes
                )
                msg = (
                    f"\t· parallelism: {parallelism}"
                    + f" | nodes:{nodes}"
                    + f" | gpus:{total_gpus}"
                    + f" | bs:{bs}"
                    + f" | grad_accum:{grad_acc}"
                    + f" | compilation: {enable_compile}"
                    + f" | precision:{precision}"
                    + f" | steps:{steps}"
                )

                tmp_cfg.id = (
                    f"{tmp_cfg.machine.name}"
                    + f"_{tmp_cfg.model.name}"
                    + f"_{tmp_cfg.framework.name}"
                    + f"_{tmp_cfg.framework.parallelism_name}"
                    + f"_{tmp_cfg.dataset.name}"
                    + f"_nodes-{nodes}"
                    + f"_gpus-{total_gpus}"
                    + f"--{experiment_parameters}"
                )
                yaml_filename = f"{parallelism}--{experiment_parameters}" + ".yaml"
                if tmp_cfg.id in cfg_seen:
                    print(
                        f"{YELLOW}Config id '{tmp_cfg.id} has been seen already, skipping duplicate...'{RESET}"
                    )
                    continue
                cfg_seen.add(tmp_cfg.id)

                # MAKE SURE BEFORE DELETE
                # outpath_yaml = os.path.join(run_path, yaml_filename)
                passed, results_msg = rules.is_valid(tmp_cfg)
                if not passed:
                    skipped.append([tmp_cfg, results_msg])
                    msg += f"-> {RED}❌ Failed:{RESET}"
                    raw_total += 1
                    for f in results_msg:
                        msg += f"\n\t{YELLOW}  {f.rule_name} --> {f.reason}{RESET}"
                    print(msg)
                    continue
                msg += f" --> {GREEN}✅ Passed!{RESET}"
                for s in results_msg:
                    if s.reason == "":
                        continue
                    # msg += (
                    #    f"\n\t{YELLOW}  {s.rule_name} --> {s.reason}{RESET}"
                    # )

                print(msg)
                raw_total += 1

                # MAKE SURE BEFORE DELETE
                # tmp_cfg.slurm.sbatch.chdir = run_path
                tmp_cfg.experiment.yaml_filename = yaml_filename
                # Resolve all yaml config parameter references before finish
                OmegaConf.resolve(tmp_cfg)
                valid.append(deepcopy(tmp_cfg))

    return cfg_seen, valid, skipped, raw_total


def _multi_parallelism_framework(
    cfg: BenchmarkConfig,
    outpath: str,
    cfg_seen: set,
    valid: list,
    skipped: list,
    raw_total: int,
):
    """Specific to framework megatron-nemo-2509

    Args:
        cfg (BenchmarkConfig): _description_
        outpath (str): _description_
        cfg_seen (set): _description_
        valid (list): _description_
        skipped (list): _description_
        raw_total (int): _description_

    Returns:
        _type_: _description_
    """
    for parallelism in cfg.model.megatron_parallelism_supported:
        assert parallelism in cfg.framework.megatron_parallelism.keys(), (
            f"{RED}Megatron parallelism {YELLOW}'{parallelism}'{RED} is not supported! Available choices {BLUE}{MEGATRON_PARALLELISM_AVAILABLE}"
        )

    tmp_cfg = deepcopy(cfg)

    for (
        bs,
        grad_acc,
        steps,
    ) in product(
        cfg.model.combinations.batch_sizes,
        cfg.model.combinations.grad_accums,
        cfg.model.combinations.steps,
    ):
        # Replace combinations from cfg.trainings* into tmp_cfg.model.training.*
        # to each experiment combination
        tmp_cfg.model.training.batch_size = bs
        tmp_cfg.model.training.grad_accum = grad_acc
        tmp_cfg.model.training.steps = steps
        tmp_cfg.experiment.output_dir = outpath

        # Get min number of nodes to run based on model and hpc architecture
        min_nodes = rules.MinNodesMemoryRule()._min_nodes_required(tmp_cfg)
        candidate_nodes = rules.MinNodesMemoryRule()._nodes_candidates(
            tmp_cfg.arch.node.gpus_per_node,
            max_gpus_scale=tmp_cfg.model.max_gpus_scale,
        )
        nodes_to_run = [n for n in candidate_nodes if n >= min_nodes]

        for n in nodes_to_run:
            parallelism_combinations = rules.megatron_parallelism_combos(
                gpus_per_node=tmp_cfg.arch.node.gpus_per_node,
                nnodes=n,
                axes=tmp_cfg.model.megatron_parallelism_supported,
            )
            for c in parallelism_combinations:
                tmp_cfg.slurm.sbatch.nodes = c["nnodes"]

                total_gpus = (
                    tmp_cfg.slurm.sbatch.gpus_per_node * tmp_cfg.slurm.sbatch.nodes
                )

                parallelism_comb_str = ""
                if c.get("tp", -1) != -1:
                    tmp_cfg.framework.megatron_parallelism.tp = c["tp"]
                    parallelism_comb_str += (
                        f"tp-{tmp_cfg.framework.megatron_parallelism.tp}"
                    )
                if c.get("pp", -1) != -1:
                    tmp_cfg.framework.megatron_parallelism.pp = c["pp"]
                    parallelism_comb_str += (
                        f"_pp-{tmp_cfg.framework.megatron_parallelism.pp}"
                    )
                if c.get("cp", -1) != -1:
                    tmp_cfg.framework.megatron_parallelism.cp = c["cp"]
                    parallelism_comb_str += (
                        f"_cp-{tmp_cfg.framework.megatron_parallelism.cp}"
                    )
                if c.get("dp", -1) != -1:
                    tmp_cfg.framework.megatron_parallelism.dp = c["dp"]
                    parallelism_comb_str += (
                        f"_dp-{tmp_cfg.framework.megatron_parallelism.dp}"
                    )
                if c.get("ep", -1) != -1:
                    tmp_cfg.framework.megatron_parallelism.ep = c["ep"]
                    parallelism_comb_str += (
                        f"_ep-{tmp_cfg.framework.megatron_parallelism.ep}"
                    )
                if c.get("sp", -1) != -1:
                    tmp_cfg.framework.megatron_parallelism.sp = c["sp"]
                    parallelism_comb_str += (
                        f"_sp-{tmp_cfg.framework.megatron_parallelism.sp}"
                    )

                parameters_combo = (
                    f"{cfg.machine.name}/{cfg.model.name}/{cfg.framework.name}/"
                    + f"{cfg.dataset.name}/nodes-{tmp_cfg.slurm.sbatch.nodes}"
                )

                tmp_cfg.slurm.sbatch.chdir = os.path.join(
                    tmp_cfg.experiment.output_dir, parameters_combo
                )
                experiment_parameters = f"bs{bs}" + f"-grad_accum{grad_acc}"
                yaml_filename = (
                    f"{parallelism_comb_str}--{experiment_parameters}" + ".yaml"
                )

                tmp_cfg.id = (
                    f"{tmp_cfg.machine.name}"
                    + f"_{tmp_cfg.model.name}"
                    + f"_{tmp_cfg.framework.name}"
                    + f"_{parallelism_comb_str}"
                    + f"_{tmp_cfg.dataset.name}"
                    + f"_nodes-{tmp_cfg.slurm.sbatch.nodes}"
                    + f"_gpus-{total_gpus}"
                    + f"--{experiment_parameters}"
                )
                if tmp_cfg.id in cfg_seen:
                    # print(
                    #    f"{YELLOW}Config id '{tmp_cfg.id} has been seen already, skipping duplicate...'{RESET}"
                    # )
                    continue
                cfg_seen.add(tmp_cfg.id)

                # MAKE SURE BEFORE DELETE
                # outpath_yaml = os.path.join(run_path, yaml_filename)
                passed, results_msg = rules.is_valid(tmp_cfg)
                msg = (
                    f"\t· tp:{c['tp']}-pp:{c['pp']}-cp:{c['cp']}-dp:{c['dp']}"
                    + f" | nodes:{tmp_cfg.slurm.sbatch.nodes}"
                    + f" | gpus:{total_gpus}"
                    + f" | bs:{bs}"
                    + f" | grad_accum:{grad_acc}"
                )
                if not passed:
                    skipped.append([tmp_cfg, results_msg])
                    msg += f"-> {RED}❌ Failed:{RESET}"
                    raw_total += 1
                    for f in results_msg:
                        msg += f"\n\t{YELLOW}  {f.rule_name} --> {f.reason}{RESET}"
                    print(msg)
                    continue
                msg += f" --> {GREEN}✅ Passed!{RESET}"
                for s in results_msg:
                    if s.reason == "":
                        continue
                    # msg += (
                    #    f"\n\t{YELLOW}  {s.rule_name} --> {s.reason}{RESET}"
                    # )

                print(msg)
                raw_total += 1

                # MAKE SURE BEFORE DELETE
                # tmp_cfg.slurm.sbatch.chdir = run_path
                tmp_cfg.experiment.yaml_filename = yaml_filename
                # Resolve all yaml config parameter references before finish
                OmegaConf.resolve(tmp_cfg)
                valid.append(deepcopy(tmp_cfg))

    return cfg_seen, valid, skipped, raw_total


def _print_summary(
    total: int,
    valid: list[BenchmarkConfig],
    skipped: list[tuple[BenchmarkConfig, rules.RuleResult]],
):
    table = Table(title="Sweep Summary")
    table.add_column("Status")
    table.add_column("Count")
    table.add_row("Total combos", str(total))
    table.add_row("[green]Valid[/green]", str(len(valid)))
    table.add_row("[yellow]Skipped[/yellow]", str(len(skipped)))
    print()
    console.print(table)
    # ToDo: Write summary file for failed or skipped configurations


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="no-config-path-provided",
        required=True,
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="no-config-path-provided",
        required=True,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    GlobalHydra.instance().clear()
    """with initialize_config_dir(
        config_dir=os.path.abspath(args.config_path), version_base=None
    ):
        cfg = compose(config_name=args.config_name)
        my_app(cfg)"""
    generate_valid_combos(
        config_path=os.path.abspath(args.config_path),
        config_name=args.config_name,
        outpath="./benchmarks_to_run",
    )

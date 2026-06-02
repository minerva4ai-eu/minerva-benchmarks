import os
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product

from constraints import rules

# sweep/generator.py
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_dataclasses import register_configs
from hydra_dataclasses.benchmark import BenchmarkConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from constraints.rules import ALL_RULES
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


##############################################################
# All configuration modules to be included are written below #
# Keep as is, to have a reference of complete foncigurations #
# to be used for all benchmarks generated.                   #
#                                                            #
# If needed to generate only specific or customized new      #
# benchmarks, copy and override values.                      #
#
# Example:
#
#   MODELS = ["llama3_8b", "mistral_7b"]
#   FRAMEWORKS = ["torchrun", "deepspeed"]
#   DATASETS = ["alpaca", "squadv2"]
#   ...
#   MODELS = ["new_model_config"]
#   FRAMEWORKS = ["torchrun", "deepspeed"]
#   DATASETS = ["alpaca", "squadv2", "new_dataset_config"]
MODELS = ["llama3_8b", "mistral_7b"]
FRAMEWORKS = ["torchrun", "accelerate", "deepspeed"]
DATASETS = ["alpaca", "squadv2"]

####### ########
REPEATS = 1


console = Console()


def expand_arch_gpu_configs(cfg: DictConfig) -> list[int]:
    """Replicates your GPU_CONFIGS=(1 $GPUS_PER_NODE) logic."""
    _p = list(cfg.framework.parallelism.keys())
    if len(_p) > 1:
        raise Exception(
            f"Fuction expand_arch_gpu_configs() received List instead of Dict for cfg.framework.parallelism: '{cfg.framework.parallelism}'"
        )
    p_name = _p[0]
    if (
        cfg.arch.slurm.get("single_gpu_also_valid")
        and (cfg.arch.slurm.nodes == 1)
        and (cfg.framework.parallelism[p_name]["min_gpus"] == 1)
    ):
        return [1, cfg.arch.slurm.gpus_per_node]
    return [cfg.arch.slurm.gpus_per_node]


def generate_valid_combos(config_path: str, config_name: str, outpath: str) -> list:
    valid, skipped = [], []

    register_configs()
    GlobalHydra.instance().clear()

    # outpath = os.path.join(outpath, "bencharks_to_run")
    os.makedirs(outpath, exist_ok=True)

    with initialize_config_dir(config_dir=config_path, version_base="1.1"):
        raw_total = 0

        def init_benchmark_compo(c) -> BenchmarkConfig:
            compo = compose(c)
            return compo

        init_cfg = init_benchmark_compo(config_name)
        print(OmegaConf.to_yaml(init_cfg))
        print(
            f"Loaded base configuration from {os.path.abspath(f'{config_name}.yaml')}"
        )
        for model, framework, dataset in product(MODELS, FRAMEWORKS, DATASETS):
            print(
                f"Iteration on: \
                \n\t· model:{model} \
                \n\t· framework:{framework} \
                \n\t· dataset:{dataset}"
            )
            cfg = compose(
                config_name,
                overrides=[
                    f"model={model}",
                    f"framework={framework}",
                    f"dataset={dataset}",
                ],
            )
            for parallelism, spex in init_cfg.framework.parallelism.items():
                tmp_cfg = deepcopy(cfg)
                target_parallelism = OmegaConf.create(DictConfig({parallelism: spex}))
                # print(f"target_parallelism: \n{target_parallelism}")
                # print(f"cfg.framework.parallelism: \n{cfg.framework.parallelism}")
                tmp_cfg.framework.parallelism = target_parallelism
                print(f"\tCreating configuration for parallelism {parallelism}:")
                # print(OmegaConf.to_yaml(cfg))

                for (
                    bs,
                    grad_acc,
                    precision,
                    steps,
                ) in product(
                    
                    cfg.model.training.batch_size,
                    cfg.model.training.grad_accum,
                    cfg.model.training.precision,
                    cfg.model.training.steps,
                ):
                    tmp_cfg.model.training.batch_size = bs
                    tmp_cfg.model.training.grad_accum = grad_acc
                    tmp_cfg.model.training.precision = precision
                    tmp_cfg.model.training.steps = steps

                    gpus_per_node = expand_arch_gpu_configs(tmp_cfg)

                    # Get min number of nodes to run based on model and hpc architecture
                    min_nodes, breakdown = rules.MinNodesMemoryRule()._min_gpus_required(
                        tmp_cfg
                    )
                    nodes = rules.MinNodesMemoryRule()._nodes_candidates(
                        tmp_cfg.arch.slurm.gpus_per_node,
                        max_gpus_scale=tmp_cfg.model.max_gpus_scale,
                    )
                    nodes_to_run = [n for n in nodes if n >= min_nodes]
                
                    for nodes in nodes_to_run:
                        
                        tmp_cfg.arch.slurm.nodes = nodes
                        
                        parameters_combo = f"{cfg.machine.name}/{cfg.model.name}/{cfg.framework.name}/{cfg.dataset.name}/nodes-{nodes}"
                        
                        tmp_cfg.arch.slurm.chdir = os.path.join(
                            tmp_cfg.experiment.output_dir, parameters_combo
                        )
                        
                        yaml_filename = (
                            f"{parallelism}--"
                            + f"bs{bs}-"
                            + f"grad_accum{grad_acc}-"
                            + f"prec{precision}-"
                            + f"steps{steps}"
                            + ".yaml"
                        )
                        run_path = os.path.join(
                            outpath,
                            parameters_combo,
                        )
                        os.makedirs(run_path, exist_ok=True)
                        for g in gpus_per_node:
                            if g == 1 and nodes == 1 and parallelism == "none":
                                tmp_cfg.arch.slurm.gpus_per_node = 1
                                tmp_cfg.arch.slurm.gres = "gpu:1"
                            
                            print(f"\t· nodes:{nodes} | bs:{bs} | grad_accum:{grad_acc} | precision:{precision} | steps:{steps}")

                            outpath_yaml = os.path.join(run_path, yaml_filename)
                            passed, fails = rules.is_valid(tmp_cfg)
                            if not passed:
                                skipped.append(outpath_yaml)
                                print(f"\t\t{RED}❌ Failed:{RESET}")
                                for f in fails:
                                    print(f"\t\t{YELLOW}  {f.rule_name} -> {f.reason}{RESET}")
                                continue
                            print(f"\t\t{GREEN}✅ Passed!{RESET}")
                            valid.append(outpath_yaml)
                            OmegaConf.save(tmp_cfg, outpath_yaml)

    _print_summary(raw_total, valid, skipped)
    return valid


def _print_summary(total, valid, skipped):
    table = Table(title="Sweep Summary")
    table.add_column("Status")
    table.add_column("Count")
    table.add_row("Total combos", str(total))
    table.add_row("[green]Valid[/green]", str(len(valid)))
    table.add_row("[yellow]Skipped[/yellow]", str(len(skipped)))
    console.print(table)
    if skipped:
        console.print("\n[yellow]Skipped reasons:[/yellow]")
        for combo, failures in skipped[:10]:  # show first 10
            for f in failures:
                console.print(
                    f"  • {combo.model.name}/{combo.framework.name}"
                    f"/{combo.arch.name}: {f.reason}"
                )


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
        outpath="/home/bsc/bsc206334/Workspace/MINERVA/minerva-benchmarks/training/training_MN5/configs-hydra/benchmarks_to_run",
    )

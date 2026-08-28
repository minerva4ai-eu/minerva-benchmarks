import gc
import os
import time

import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# # sys.path.append("../..")

import torch
import torch.distributed as dist
from gpu_monitor import GPUMonitorCallback, start_gpu_monitor
from shared.custom_train import PerformanceTrackingSFTTrainer
from shared.data import load_and_prepare_raw_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.utils import (
    count_parameters,
    print_rank,
    save_summary_stats_json,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl.trainer.sft_config import (
    SFTConfig,
)
from utils import parse_args, parse_config

args = parse_args()

import logging
from datetime import datetime

RUNID = os.environ.get("SLURM_JOB_ID", datetime.now().strftime('%Y%m%d%H%M%S'))
RUNJD = os.environ.get("SLURM_STEP_ID")
LOG_DIR = os.path.join("outputs", "logs", "pyft", RUNID)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# FIXME: logging level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s |  %(levelname)s | %(name)s : %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"minerva-step{RUNJD}.log"))],
)

logger = logging.getLogger(__name__)

def is_main_process():
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


# --- Main ---
def main():
    # Get main id
    jobid = os.environ["SLURM_JOB_ID"]
    jobstepid = os.environ["SLURM_STEP_ID"]
    jobsteprocid = os.environ["SLURM_PROCID"]
    
    # Get rank
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("type(args.yaml_file) = %s", type(args.yaml_file))
    logger.info("args.yaml_file = %s", args.yaml_file)

    configs = parse_config(args.yaml_file)
    logger.info("configs = %s", configs)
    model_name = configs['model']['path']
    logger.info("model_name = %s", model_name)
    dataset_path = configs['dataset']['path']
    logger.info("data = %s", dataset_path)
    dataset_name = configs['dataset']['name']
    logger.info("dataset = %s", dataset_name)
    precision = configs['model']['training']['precision']
    logger.info("precision = %s", precision)
    batch_size = configs['model']['training']['batch_size']
    logger.info("batch_size = %s", batch_size)
    gradient_accumulation_steps = configs['model']['training']['grad_accum']
    logger.info("gradient_accumulation_steps = %s", gradient_accumulation_steps)
    lr = configs['model']['training']['lr']
    logger.info("lr = %s", lr)
    enable_compile = configs['model']['training']['enable_compile']
    logger.info("enable_compile = %s", enable_compile)
    max_steps = configs['model']['training']['steps']
    logger.info("max_steps = %s", max_steps)
    epochs = configs['model']['training']['epochs']
    logger.info("epochs = %s", epochs)
    max_length = configs['model']['training']['max_model_length']
    logger.info("epochs = %s", max_length)
    # TODO: fix
    max_length = 1024
    run_dir = configs['run_dir']
    logger.info("run_dir = %s", run_dir)
    output_dir = os.path.join(run_dir, args.output_dir)
    logger.info("output_dir = %s", output_dir)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    print_rank(rank, f"Loading tokenizer... {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_rank(rank, "Tokenizer Loaded")

    # ---------------------------------------------------------------------
    # Handle dataset path (string or dict)
    # ---------------------------------------------------------------------
    # train_dataset, eval_dataset, collate_fn, _ = load_dataset(
    #    dataset_name=args.dataset,
    #    dataset_path=args.data,
    #    tokenizer=tokenizer,
    #    max_length=MAX_LENGTH,
    # )
    train_dataset, eval_dataset = load_and_prepare_raw_dataset(
        dataset_name=dataset_name, dataset_path=dataset_path, test_size=0.1
    )

    # --- Precision selection ---
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    trainable_params, total_params, trainable_pct = 0, 0, 0

    training_args = SFTConfig(
        output_dir=output_dir,
        # overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_total_limit=1,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        optim="adamw_torch",
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        eval_steps=None,
        ddp_timeout=1800,
        # Dataloader is created automatically from trainer
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        data_seed=32,
        dataloader_persistent_workers=args.dataloader_num_workers > 1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=8,
        # --- SFT-specific args ---
        max_length=max_length,  # replaces manual truncation in collator
        dataset_text_field="text",  # TODO: set to your dataset's text column name
        # OR remove and use formatting_func below
        packing=True,  # set True to pack short sequences for efficiency
        dataset_kwargs={"skip_prepare_dataset": False},
        # TODO: If your dataset is already tokenized (input_ids present), set:
        #   dataset_kwargs={"skip_prepare_dataset": True}
        #   and remove dataset_text_field above.
        # torch model compilation
        **(
            {
                "torch_compile": True,
                "torch_compile_backend": "inductor",
                # "torch_compile_mode": "max-autotune-no-cudagraphs",
            }
            if bool(enable_compile)
            else {}
        ),
    )
    training_args.num_train_epochs = epochs if epochs is not None else 1
    if max_steps is not None:
        training_args.max_steps = int(max_steps)

    try:
        monitor = GPUMonitorCallback(n_gpus=int(os.environ.get("GPUS_PER_NODE", 1)))

        # Peak GPU TFLOPs for MFU (bf16/fp16 tensor core peak).
        # Set PEAK_GPU_TFLOPS env var for your hardware, e.g.:
        #   A100 SXM4 80GB = 312, H100 SXM5 = 989, MI250X = 383, MI300X = 1307

        gpu_configs = configs['arch']['gpu']
        logger.info("gpu_configs = %s", gpu_configs)
        if precision == 'bf16':
            _peak_gpu_tflops = gpu_configs.get('theoretical_peak_bf16_tensor_tflops')
        peak_gpu_tflops = float(_peak_gpu_tflops) if _peak_gpu_tflops else None
        logger.info("peak_gpu_tflops = %s", peak_gpu_tflops)
        gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
        print_rank(
            0,
            f"GPU_NAME: {gpu_name} | Using peak GPU TFLOPS for MFU calculation: {peak_gpu_tflops} TFLOPS",
        )

        logger.info(f"Loading Model... dtype: {dtype}")
        if enable_compile:
            # torch._dynamo.exc.BackendCompilerFailed: backend='compile_fn' raised:
            # NotImplementedError: DDPOptimizer backend: Found a higher order op in the graph. This is not supported.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=None,  # Trainer will put model on device
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=None,  # Trainer will put model on device
                attn_implementation="flash_attention_2",
            )
        print_rank("Model Loaded")

        flops_callback = mfu_callback_from_hf_config(
            model,
            tokenizer,
            gpu_peak_flops=peak_gpu_tflops,
            seq_length=max_length,
        )
        trainer = PerformanceTrackingSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=collate_fn,
            processing_class=tokenizer,
            callbacks=[
                monitor,
                flops_callback,
            ],
            peak_gpu_tflops=peak_gpu_tflops,
        )

        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )

        # Train Model
        start_time = time.time()
        trainer.train()
        total_finetune_time = time.time() - start_time

        # Stop GPU monitor
        stop_flag["stop"] = True
        time.sleep(2)  # give it a moment to exit cleanly

        trainable_params, total_params, trainable_pct = count_parameters(model)

        # ---- Compute metrics ----
        log_history = trainer.state.log_history
        avg_training_loss = avg_validation_loss = None
        avg_epoch_time_sec = avg_epoch_time_hours = None
        avg_step_time_sec = avg_step_time_hours = None

        total_training_time_secs = getattr(
            trainer.state, "total_training_seconds_custom", total_finetune_time
        )
        tokens_per_gpu_all_epochs = getattr(
            trainer.state, "total_tokens_per_gpu_custom", trainer.total_tokens_this_gpu
        )
        tokens_global_all_epochs = getattr(
            trainer.state, "total_tokens_global_custom", trainer.total_tokens_global
        )

        avg_gpu_flops = getattr(trainer.state, "average_flops_custom")
        # global_avg_gpu_flops = getattr(trainer.state, "global_average_flops_custom")
        avg_gpu_mfu = getattr(trainer.state, "average_mfu_custom")
        # global_avg_gpu_mfu = getattr(trainer.state, "global_average_mfu_custom")

        if training_args.max_steps:
            avg_step_time_sec = total_training_time_secs / training_args.max_steps
            avg_step_time_hours = avg_step_time_sec / 3600
        else:
            avg_epoch_time_sec = (
                total_training_time_secs / training_args.num_train_epochs
            )
            avg_epoch_time_hours = avg_epoch_time_sec / 3600
            avg_step_time_sec = avg_epoch_time_sec / len(train_dataset)  # approximate
            avg_step_time_hours = avg_step_time_sec / 3600

        # ---- Compute derived metrics ----
        effective_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )

        samples_per_sec = (
            effective_batch_size / avg_step_time_sec if avg_step_time_sec else None
        )
        training_throughput_tokens_per_sec_per_gpu = (
            tokens_per_gpu_all_epochs / total_training_time_secs
            if total_training_time_secs
            else None
        )
        training_throughput_tokens_per_sec_global = (
            tokens_global_all_epochs / total_training_time_secs
            if total_training_time_secs
            else None
        )
        avg_gpu_power_watts = (
            sum(gpu_stats_during["power"]) / len(gpu_stats_during["power"])
            if gpu_stats_during["power"]
            else None
        )
        tokens_per_sec_per_watt_global = (
            training_throughput_tokens_per_sec_global / avg_gpu_power_watts
            if training_throughput_tokens_per_sec_global and avg_gpu_power_watts
            else None
        )

        save_summary_stats_json(
            summary={
                "nodes": int(os.environ.get("SLURM_NNODES", 1)),
                "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
                "total_gpus": world_size,
                "model": model_name,
                "dataset": dataset_path,
                "framework": "accelerate",
                "parallelism_type": "ddp",
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "compile": enable_compile,
                "precision": precision,
                "max_length": max_length,
                "trainable_parameters": trainable_params,
                "total_trainable_parameters": total_params,
                "trainable_parameters_percentage": trainable_pct,
                "learning_rate": training_args.learning_rate,
                "avg_gpu_memory_gb": sum(gpu_stats_during["mem"])
                / len(gpu_stats_during["mem"])
                if gpu_stats_during["mem"]
                else None,
                "peak_gpu_memory_gb": max(gpu_stats_during["mem"])
                if gpu_stats_during["mem"]
                else None,
                "avg_gpu_utilization_percent": sum(gpu_stats_during["util"])
                / len(gpu_stats_during["util"])
                if gpu_stats_during["util"]
                else None,
                "peak_gpu_utilization_percent": max(gpu_stats_during["util"])
                if gpu_stats_during["util"]
                else None,
                "avg_gpu_power_watts": avg_gpu_power_watts,
                "peak_gpu_power_watts": max(gpu_stats_during["power"])
                if gpu_stats_during["power"]
                else None,
                "total_execution_time_hours": total_training_time_secs / 3600,
                "training_throughput_tokens_per_sec": training_throughput_tokens_per_sec_global,
                "training_throughput_tokens_per_sec_global": training_throughput_tokens_per_sec_global,
                "training_throughput_tokens_per_sec_per_gpu": training_throughput_tokens_per_sec_per_gpu,
                "tokens_per_sec_per_watt_global": tokens_per_sec_per_watt_global,
                "samples_per_sec": samples_per_sec,
                "total_tokens_per_gpu_all_epochs": tokens_per_gpu_all_epochs,
                "total_tokens_global_all_epochs": tokens_global_all_epochs,
                "avg_training_loss": avg_training_loss,
                "avg_validation_loss": avg_validation_loss,
                "total_training_time_hours": total_training_time_secs / 3600,
                "avg_epoch_training_time_sec": avg_epoch_time_sec,
                "avg_epoch_training_time_hours": avg_epoch_time_hours,
                "avg_step_training_time_sec": avg_step_time_sec,
                "avg_step_training_time_hours": avg_step_time_hours,
                "avg_gpu_flops": avg_gpu_flops,
                "avg_gpu_mfu": avg_gpu_mfu,
            },
            output_file=os.path.join(output_dir, f"training_summary_job{jobid}-step{jobstepid}-task{jobsteprocid}-{rank}.json"),
        )

        del trainer, model
        gc.collect()  # frees Python objects + their GPU tensor wrappers
        torch.cuda.empty_cache()  # returns freed CUDA memory back to the OS/driver
        print_rank(rank, "Fine-tuning completed successfully.")
    except Exception as e:
        save_summary_stats_json(
            summary={
                "nodes": int(os.environ.get("SLURM_NNODES", 1)),
                "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
                "total_gpus": world_size,
                "model": model_name,
                "dataset": dataset_path,
                "framework": "accelerate",
                "parallelism_type": "ddp",
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "compile": enable_compile,
                "precision": precision,
                "max_length": max_length,
                "trainable_parameters": trainable_params,
                "total_trainable_parameters": total_params,
                "trainable_parameters_percentage": trainable_pct,
                "learning_rate": training_args.learning_rate,
                "error": str(e),
            },
            output_file=os.path.join(output_dir, f"training_summary_job{jobid}-step{jobstepid}-task{jobsteprocid}-{rank}.json"),
        )

        print_rank(rank, "Fine-tuning failed to complete!")
        raise e


if __name__ == "__main__":
    main()

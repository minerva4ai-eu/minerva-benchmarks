import gc
import logging
import os
import time

import torch
from scripts.shared.args import construct_config, get_deepspeed_parser
from scripts.shared.custom_train import PerformanceTrackingSFTTrainer
from scripts.shared.data import (
    collate_fn,
    get_train_eval_path,
    load_prepared_packed_dataset,
)

# scripts.shared.data import load_and_prepare_raw_dataset
from scripts.shared.flops import mfu_callback_from_hf_config
from scripts.shared.gpu_monitor import start_gpu_monitor
from scripts.shared.logger import RankAdapter, setup_logging
from scripts.shared.utils import (
    is_main_process,
    setup_distributed,
)
from scripts.slurm.utils import load_config
from transformers import AutoConfig, AutoTokenizer
from trl.trainer.sft_config import (
    SFTConfig,
)

rank, world_size, local_rank = setup_distributed()
config = load_config(get_deepspeed_parser().parse_args().yaml_file)
setup_logging(level=logging.INFO, cfg=config)
logger = logging.getLogger(f"MINERVA_BENCH.{__name__}")
logger_rank = RankAdapter(logger, {})


def main(repeatid: int):

    # Get main id
    jobid = os.environ["SLURM_JOB_ID"]
    jobstepid = os.environ["SLURM_STEP_ID"]
    jobsteprocid = os.environ["SLURM_PROCID"]

    args = get_deepspeed_parser().parse_args()
    cfg = construct_config(args)
    model_path = cfg.model_path
    model_name = cfg.model_name
    train_path, eval_path = get_train_eval_path(cfg)

    torch.cuda.empty_cache()

    output_dir = cfg.output_dir
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        cfg.precision, torch.float32
    )
    logger.info(f"Training dtype: {dtype}")

    # --- Tokenizer ---
    logger.info(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset = load_prepared_packed_dataset(train_path)
    eval_dataset = load_prepared_packed_dataset(eval_path)

    # torch.compile (inductor) is incompatible with DeepSpeed ZeRO-3: parameters
    # remain partitioned (size-0 tensors) and are only gathered at runtime, so the
    # compiled backward graph fails with shape mismatches like
    # "size of tensor a (0) must match the size of tensor b (4096)".
    # Detect ZeRO-3 from the DeepSpeed config and force-disable compilation.
    compilation_args = {}
    if cfg.enable_compile:
        torch_compile_backend = "inductor"
        torch_compile_mode = "default"
        compilation_args = {
            "torch_compile": True,
            "torch_compile_backend": torch_compile_backend,
            "torch_compile_mode": torch_compile_mode,
        }
        logger.info(f"Compilation arguments: {compilation_args}")

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        model_init_kwargs={
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
        },
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        save_total_limit=1,
        fp16=cfg.precision == "fp16",
        bf16=cfg.precision == "bf16",
        optim="adamw_torch",
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        eval_steps=None,
        ddp_timeout=1800,
        warmup_ratio=0.1,
        # Dataloader is created automatically from trainer
        dataloader_drop_last=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
        data_seed=32,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
        # Deepspeed Config
        # deepspeed=args.deepspeed_config_file,
        # --- SFT-specific args ---
        # max_length=MAX_LENGTH,  # replaces manual truncation in collator
        # dataset_text_field="text",  # TODO: set to your dataset's text column name
        # OR remove and use formatting_func below
        # packing=True,  # set True to pack short sequences for efficiency
        dataset_kwargs={"skip_prepare_dataset": True},
        # pad_to_multiple_of=MAX_LENGTH,
        # TODO: If your dataset is already tokenized (input_ids present), set:
        #   dataset_kwargs={"skip_prepare_dataset": True}
        #   and remove dataset_text_field above.
        # torch model compilation
        **compilation_args,
    )

    training_args.num_train_epochs = cfg.epochs if cfg.epochs is not None else 1
    if cfg.max_steps is not None:
        training_args.max_steps = int(cfg.max_steps)

    logger.info(
        f"GPU_NAME: {cfg.gpu_name} | Using peak GPU TFLOPS for MFU calculation: {cfg.peak_flops} TFLOPS",
    )

    logger.info(f"Loading Model... dtype: {dtype}")

    model_config = AutoConfig.from_pretrained(model_path)
    flops_callback = mfu_callback_from_hf_config(
        model_config,
        tokenizer,
        gpu_peak_flops=cfg.peak_flops,
        seq_length=cfg.max_length,
    )
    trainer = PerformanceTrackingSFTTrainer(
        cfg=config,
        model=cfg.model_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        callbacks=[
            flops_callback,
        ],
        peak_gpu_tflops=cfg.peak_flops,
    )

    logger.info("Trainer initialized and model has been wrapped!")

    try:
        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )
        logger.info(f"Distributed type: {trainer.accelerator.distributed_type}")

        trainer.train()

        stop_flag["stop"] = True
        time.sleep(2)

        trainer.write_summary(
            output_file=os.path.join(
                cfg.output_dir,
                f"repeatid-{repeatid}",
                f"training_summary_job{jobid}-step{jobstepid}-nodeid{jobsteprocid}-deviceid{rank}.json",
            ),
            gpu_stats=gpu_stats_during,
        )
        logger.info("Fine-tuning completed successfully.")

    except Exception as e:
        logger.exception("Fine-tuning failed with error!")
        trainer.write_summary(
            output_file=os.path.join(
                cfg.output_dir,
                f"repeatid-{repeatid}",
                f"training_summary_job{jobid}-step{jobstepid}-nodeid{jobsteprocid}-deviceid{rank}.json",
            ),
            gpu_stats={},
            exception_msg=str(e),
        )
        raise e

    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for repeatid in range(config.experiment.repeat):
        main(repeatid)

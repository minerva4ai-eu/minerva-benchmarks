# 🧠 Minerva Benchmarks

Minerva Benchmarks is a collection of **reproducible performance benchmarks** for large language models on EuroHPC systems.

The repository covers two independent benchmark suites:

- **Inference Benchmarks**: Serving, throughput, latency, and GPU utilization.
- **Training & Fine-Tuning Benchmarks**: DDP/FSDP/ZeRO/Megatron scaling, throughput, TFLOPs, memory, GPU utilization, and time-to-train.

Benchmarks are configured per supercomputer.

---

## 📁 Repository Structure

```text
minerva-benchmarks/
├── analytics/       # Analytics, dashboards, and data visualization
│   ├── data/        # Benchmark result CSVs
├── inference/       # Inference & serving benchmarks
│   ├── benchmarks/  # Benchmark scripts (serving, latency, throughput, etc.)
│   ├── configs-*/   # Per-supercomputer configurations
│   ├── envs-sing-imgs/  # Singularity image definitions
│   ├── envs-yaml/       # Conda/YAML environment specs
│   ├── scripts/       # Per-framework scripts (vLLM, SGLang, DeepSpeed-MII)
│   ├── run_*.sh       # Benchmark runner scripts
│   └── README.md
├── training/        # Training & fine-tuning benchmarks
│   ├── configs_hydra/     # Hydra-based configuration system (machines, models, frameworks, datasets)
│   ├── envs/              # Training environments (uv venvs, Singularity/NeMo containers)
│   ├── install/           # Setup scripts (environments, containers, datasets)
│   ├── scripts/           # Per-framework training scripts, SLURM submitter, shared code
│   ├── analytics/         # Result aggregation into summary CSVs
│   ├── minerva-cli.sh     # CLI entry point (compose, validate, submit)
│   ├── MINERVA.job        # SLURM job script (one srun step per config)
│   └── README.md
└── README.md
```

---

## 🚀 Getting Started

### Inference Benchmarks

See: [inference/README.md](inference/README.md)

Covers:

* vLLM, DeepSpeed-MII, SGLang
* Serving, latency, throughput, and prefix caching benchmarks
* GPU monitoring and utilization tracking
* Result aggregation and scoring

### Training & Fine-Tuning Benchmarks

See: [training/README.md](training/README.md)

Covers:

* HuggingFace Accelerate (DDP/FSDP)
* PyTorch TorchRun (single-GPU/DDP/FSDP)
* Microsoft DeepSpeed (ZeRO-1/2/3, ZeRO-3-Offload)
* NVIDIA NeMo / Megatron (TP/PP/CP/DP/EP/SP)
* Models: gemma3_1b, mistral_7b, llama3_8b, llama3_70b, alia_40b, qwen2.5_7b, qwen2.5_72b
* Datasets: alpaca, squadv2, tulu-3-sft-mixture
* Hydra-based configuration, `minerva-cli.sh` submission, venv or Singularity runtimes
* Result aggregation (`analytics.generateSummaryTable_jobs`)

### Analytics

See: [analytics/README.md](analytics/README.md) (when available)

Covers:

* Interactive Plotly dashboards
* Energy consumption plots
* Performance plots
* Cross-system benchmark comparison data

---

## 🖥️ Supported Systems

Benchmarks are organized per system (e.g. MareNostrum5, Leonardo, Jean Zay, Adastra). Training profiles currently exist for MareNostrum 5 (ACC and GPP) and Jean Zay H100.
Each system has its own configuration, environment definitions, and scripts.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## 💬 Support

For questions or contributions, contact:
**[support@minerva4ai.eu](mailto:support@minerva4ai.eu)**

---

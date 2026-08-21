# 🧠 Minerva Benchmarks

Minerva Benchmarks is a collection of **reproducible performance benchmarks** for large language models on EuroHPC systems.

The repository covers two independent benchmark suites:

- **Inference Benchmarks**: Serving, throughput, latency, and GPU utilization.
- **Training & Fine-Tuning Benchmarks**: DDP/FSDP scaling, throughput, memory, and time-to-train.

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
│   ├── configs_hydra/     # OmegaConf/Hydra-based configuration system
│   ├── envs/              # Training environment definitions
│   ├── scripts/           # Per-framework training scripts
│   ├── minerva-cli.sh     # CLI entry point
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
* Torchrun (DDP/FSDP/None)
* DeepSpeed (pure and Accelerate-integrated)
* Dataset handlers and scaling analysis
* Hydra-based configuration management

### Analytics

See: [analytics/README.md](analytics/README.md) (when available)

Covers:

* Interactive Plotly dashboards
* Energy consumption plots
* Performance plots
* Cross-system benchmark comparison data

---

## 🖥️ Supported Systems

Benchmarks are organized per system (e.g. MareNostrum5, Leonardo, Jean Zay, Adastra).
Each system has its own configuration, environment definitions, and scripts.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## 💬 Support

For questions or contributions, contact:
**[minerva_support@bsc.es](mailto:minerva_support@bsc.es)**

---

# 🧠 Minerva Benchmarks

Minerva Benchmarks is a collection of **reproducible performance benchmarks** for large language models on EuroHPC systems.

The repository covers two independent benchmark suites:

- **Inference Benchmarks**: Serving, throughput, latency, and GPU utilization.
- **Training & Fine-Tuning Benchmarks**: DDP/FSDP scaling, throughput, memory, and time-to-train.

Each benchmark suite is **self-contained** and organized per supercomputer.

---

## 📁 Repository Structure

```text
minerva-benchmarks/
├── inference/     # Inference & serving benchmarks
│   ├── inference-MN5/
│   ├── inference-Leonardo/
│   └── README.md
├── training/      # Training & fine-tuning benchmarks
│   ├── training-MN5/
│   ├── training-Leonardo/
│   └── README.md
└── README.md
```

---

## 🚀 Getting Started

### Inference Benchmarks

See: [inference/inference-MN5/README.md](training/inference-MN5/README.md)

Covers:

* vLLM, DeepSpeed-MII, SGLang
* Serving benchmarks
* GPU monitoring
* Result aggregation

### Training & Fine-Tuning Benchmarks

See: [training/training_MN5/README.md](training/training_MN5/README.md)

Covers:

* HuggingFace Accelerate
* Torchrun (DDP/FSDP)
* Dataset handlers
* Scaling and memory analysis

---

## 🖥️ Supported Systems

Benchmarks are organized per system (e.g. MareNostrum5, Leonardo).
Each system directory contains its own configuration, scripts, and results.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## 💬 Support

For questions or contributions, contact:
**[minerva_support@bsc.es](mailto:minerva_support@bsc.es)**

---

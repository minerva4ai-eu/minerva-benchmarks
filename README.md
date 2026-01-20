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

<<<<<<< HEAD
This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).  
You are free to use, modify, and distribute this code, provided that any derivative works are also released under the same license. Commercial and non-commercial use is allowed under the GPL-3.0 terms.

---

### 💬 Suggestions and Feedback

If you have any suggestions or would like to contribute improvements to this repository, please contact us at **minerva_support@bsc.es**.


---

### 📚 References
[1] **vLLM:** Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023, October). Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles (pp. 611-626).

[vLLM GitHub](https://github.com/vllm-project/vllm) and  [vLLM Benchmarks GitHub](https://github.com/vllm-project/vllm/tree/main/benchmarks)

[2] **DeepSpeed-MII:** [DeepSpeed-MII GitHub](https://github.com/deepspeedai/DeepSpeed-MII)

[3] **LLama Models:** 
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
- [HuggingFace Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [HuggingFace Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)

[4] **Gemma Models:** Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., ... & Iqbal, S. (2025). Gemma 3 technical report. arXiv preprint arXiv:2503.19786. [HuggingFace Gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

[5] **Mistral Models:** [HuggingFace Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)


---
=======
This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## 💬 Support

For questions or contributions, contact:
**[minerva_support@bsc.es](mailto:minerva_support@bsc.es)**

---
>>>>>>> main

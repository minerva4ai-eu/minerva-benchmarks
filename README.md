# MINERVA Benchmarks

Reproducible AI performance benchmarks on European supercomputers, developed by the MINERVA Support Team.

## About MINERVA

[MINERVA](https://minerva4ai.eu) (European Support Centre for Scalable AI Research and Deployment) is a centralised, EuroHPC JU-supported initiative designed to accelerate large-scale artificial intelligence research across Europe. 

Led by a consortium of major supercomputing centres, research institutions and companies, MINERVA aims to promote the use of European HPC infrastructure for researchers and startups. To this end, we develop open-source tools and workflows to make it easier for users to get started on HPC systems. We also provide technical support: assistance, consulting, workshops and embedded engineering. This includes guidance on accessing EuroHPC or national computing resources, training and fine-tuning models, and optimising complex AI workloads (e.g. scaling code from a single GPU to multiple nodes).

Consortium members: BSC, CEA, CINECA, CINES, IDRIS, Forschungszentrum Jülich, UNIMORE, Tübingen AI Centre, Domyn and AMD Silo AI.

## Why benchmark supercomputers?

European supercomputers differ in GPU vendors and generations, memory, interconnects and software stacks, and the same model can perform very differently across systems, frameworks and parallelism strategies. The benchmarks assess:

- how each supercomputer performs on the same inference and training workloads;
- which inference engines and training frameworks perform best on each system;
- which parallelism strategies and configurations suit each model size;
- how performance scales from a single GPU to multiple nodes;
- how much energy each system and configuration consumes for the same workload;
- how well AI frameworks are supported on NVIDIA (CUDA) and AMD (ROCm) GPUs.

Results are updated as new frameworks, models and hardware are released, and published openly so that they can be reproduced and extended. The benchmarks characterise workloads and systems without endorsing any hardware vendor.

## Getting started

The repository is divided into three parts: `inference` benchmark, `training` benchmark and `analytics`, which contains the results and interactive dashboards.

Benchmarks are organised per system (e.g. MareNostrum 5, Leonardo, Jean Zay, Adastra). Each system has its own configuration, environment definitions, and scripts.

Environments are pinned for NVIDIA (CUDA) and AMD (ROCm) GPUs, and each system has its own configuration, so results can be reproduced or the suites adapted to another supercomputer, cluster or cloud.

### Benchmark LLM inference

Serve models on a supercomputer and measure their performance under load.

- **Frameworks:** vLLM, SGLang, DeepSpeed-MII
- **Models:** Mistral 7B Instruct v0.3, Llama 3.1 8B Instruct, Gemma 3 12B IT, ALIA 40B Instruct, Llama 3.3 70B Instruct, Llama 3.1 405B (base and Instruct)
- **Datasets:** ShareGPT, Sonnet
- **Metrics:** latency (TTFT, ITL, TPOT), throughput (tokens/s, requests/s), GPU memory, power and energy efficiency (tokens per watt)
- **Documentation:** [inference/README.md](inference/README.md)

### Benchmark training and fine-tuning

Fine-tune models on a supercomputer and compare parallelism strategies.

- **Frameworks:** PyTorch, Hugging Face Accelerate, DeepSpeed, NVIDIA NeMo/Megatron
- **Models:** Gemma 3 1B, Mistral 7B, Llama 3 8B, Qwen2.5 7B, ALIA 40B, Llama 3 70B, Qwen2.5 72B
- **Datasets:** Alpaca, SQuAD v2, Tülu 3 SFT mixture
- **Metrics:** training time (per step, per epoch and total), throughput (tokens/s, TFLOP/s per GPU, model FLOPs utilisation), GPU memory, power and utilisation, training and validation loss
- **Documentation:** [training/README.md](training/README.md)

### Comparison analytics

Explore published results across systems.

- **Tools:** interactive dashboard, energy and performance plots
- **Data:** inference results for the models and datasets above (will be completed)
- **Documentation:** [analytics/README.md](analytics/README.md) (coming soon)

## Upcoming work

- **Training results:** publish the results of the training and fine-tuning benchmarks across the supported systems, alongside the inference results.
- **Vision models, multimodal models and downstream tasks:** extend the benchmarks beyond text generation to vision and multimodal models, and to downstream tasks.
- **Mixture-of-Experts models:** add MoE models to measure the performance of sparse architectures and expert parallelism.

## Contributing and support

Contact [support@minerva4ai.eu](mailto:support@minerva4ai.eu) for contributions or questions.

Request free MINERVA support for your AI project on our [support page](https://minerva4ai.eu/support/).

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

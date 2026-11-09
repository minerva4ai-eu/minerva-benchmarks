# Minerva analytics

Explore the inference benchmark results in [`data/inference/`](data/inference/) without installing anything:
open [`dashboard/index.html`](dashboard/index.html) in a browser, or publish the
`dashboard/` folder with GitHub Pages.

## Using the dashboard

The bar at the top switches between **Inference** and **Training**. Each has its own filters, views
and metric definitions; the address ends in `#training` when the Training view is open, so you can
share a link to it. The steps below describe Inference; Training works the same way (see
[Training view](#training-view)).

1. **Pick systems** with the colored chips at the top. Each system keeps its color in every chart (warm = AMD, cool = NVIDIA). The number on each chip is how many runs that system has under the current filters; dashed means none.
2. **Start from the default filters**, then change anything you like (see below). *Clear all* shows every run; *Reset to defaults* brings the defaults back.
3. **Choose a view**
   - *Compare*: bars by model, system, framework or dataset for any metric. *Sort bars* can put the best value first.
   - *Scaling*: how a metric changes with concurrency, GPUs, nodes or parallelism.
   - *Trade-offs*: one metric against another (for example throughput against latency).
   - *Heatmap*: a *Cluster* selector (All clusters or one), the *Columns* variable, and rows that are the group-by variables you tick (default system and model; also framework, dataset, GPUs, nodes, concurrency, parallelism, max length). With several systems shown and *System* not in the rows or columns, each system gets its own labelled block of columns. Columns are framework (or GPUs, concurrency, nodes, max length, dataset), and each cell is the chosen metric. Empty cells were not measured, so it doubles as a coverage map. *Ratio of two columns* turns two columns into one diverging column, for example vLLM ÷ SGLang (above 1× means the first has the higher value).
   - *Data*: sortable table of the filtered runs, with CSV download. Below it, a *Python notebook* (pandas and matplotlib, run in the browser) works on the filtered runs as `df`. The last column lists each run's extra server arguments (for example `--enforce-eager`).
4. **Check the cards above the chart.** They say what each bar or point is, and show every metric in use with its formula, unit and whether higher or lower is better. When *Best run by* uses another metric, that metric gets its own card tagged *Picks the best run*.
5. **Watch the yellow note.** It appears when a chart mixes values that should not be compared, for example *This view mixes 2 frameworks. Use the filters to compare like with like.* It checks models, datasets, frameworks, concurrent request levels, GPU counts, node counts and max model lengths. Tensor/pipeline parallelism and extra arguments are not flagged, because *Best run* exists to choose among them.
6. **Open *Metric definitions and score formulas*** at the bottom for the full guide.

### What "Best run", "Median" and "Mean" mean

The charts group runs (for example, all runs of one model on one system, which differ in
tensor/pipeline parallelism and other settings) and show one value per group. **Show** picks how:

| Show | Value drawn for each group |
|---|---|
| Best run | The run that wins on the criterion chosen in **Best run by** |
| Median/Mean | The median or mean of the plotted metric across the group |
| Every run (Trade-offs only) | No grouping: one point per run |

**Best run by** defaults to *The plotted metric*: the highest value for throughput, scores and
tokens per watt, the lowest for latency, power, energy per token and memory. You can instead pick
any other metric, for instance *Minerva global score*. The chart then shows the plotted
metric of **the single run with the best global score**, which is often a different run from the
one with the best throughput. Example (Jean Zay H100, Llama-3.1-8B, default filters): the highest
output throughput is 6,680 tokens/s (TP 4), while the run with the best global score delivers
6,264 tokens/s (TP 1).

In Trade-offs, grouped modes make one point per system, model, framework, dataset,
concurrency level and GPU count. In *Best run* mode both axes come from the same winning run.

## Metrics and score formulas

The scores come from [`inference/generateScores.py`](../inference/generateScores.py) and are
read from the CSVs as they are. Every time the dashboard is built, `minerva_data.check_scores` verifies the formulas below and the global-score weights against all runs in `data/inference/`; the weights shown in the dashboard guide come from that check, and a warning appears if the CSVs ever stop matching the formulas. The guide at the bottom of the dashboard shows the same formulas typeset (MathML, no extra libraries), the weight of each component, and each score's median and range.

Notation: the scores are **G** (global), **L** (latency), **T** (throughput) and **E** (energy).
The measurements are **TPS** (output tokens per second), **RPS** (requests per second), **P**
(average power in W), **N** (GPUs used), and **TTFT/ITL/TPOT** (latencies in ms).

| Metric | Definition | Better |
|---|---|---|
| **Minerva global score (G)** | G = 0.34 L + 0.34 T + 0.32 E | higher |
| **Latency score (L)** | L = 10,000 ÷ (TTFT + ITL + TPOT), times in ms (CSV column *Latency Score Scaled*) | higher |
| **Throughput score (T)** | T = 2 ÷ (1/TPS + 1/RPS), the harmonic mean of the two throughputs; usually close to 2 × RPS | higher |
| **Energy score (E)** | E = TPS ÷ P (tokens/s per W) | higher |
| Output throughput (TPS) | Generated tokens per second across all concurrent requests | higher |
| Request throughput (RPS) | Completed requests per second | higher |
| Throughput per GPU | TPS ÷ N (derived) | higher |
| TTFT/ITL/TPOT | Time to first token/average time between tokens/average time per output token after the first | lower |
| Energy per token, per GPU (J) | J = P ÷ TPS = 1 ÷ E, in J/token (derived). Does not depend on the number of GPUs | lower |
| Energy per token, all GPUs (J<sub>all</sub>) | J<sub>all</sub> = P × N ÷ TPS, in J/token (derived): the energy of the whole run per generated token | lower |
| Average power, peak GPU memory | As reported by each run | lower |

Power appears to be a per-GPU average while throughput covers the whole run, so energy metrics
are best used to compare systems with each other, not as absolute figures.

## Default filters

Mixing datasets, frameworks or setups makes numbers incomparable, so the dashboard opens
with a like-for-like selection, defined in [`dashboard/defaults.json`](dashboard/defaults.json):

| Filter | Default |
|---|---|
| Dataset | sharegpt |
| Framework | vllm |
| GPUs | 4 |
| Max model length | 4096 (the only value measured on all six systems) |
| Concurrent requests | 250 |

A default **pauses itself** when the current chart is about that dimension: comparing
frameworks lifts the framework default, and a Scaling chart across concurrency or GPUs lifts
that default so the full curve is drawn. Scaling across nodes (or tensor/pipeline parallelism) also lifts the GPU default, because the GPU count changes with them. The filter panel shows "paused" and a note appears
above the chart. Filters you change yourself are always respected.

**Scaling also selects a model.** Lines that mix models are not comparable, so entering the
Scaling view selects `Llama-3.1-8B-Instruct` (it has the widest coverage: all six systems across
4 to 32 GPUs, five concurrency levels) unless you already chose a model. It is removed again when
you leave Scaling. Change it in the Model filter.

To change either, edit `defaults.json` (`filters` use the filter names `dataset`, `framework`,
`gpus`, `max_len`, `concurrency`, `nodes`, `tp`, `pp`, `model`; `scaling.model` is the Scaling
model) and run `python build_dashboard.py`.

Only Adastra MI300, Jean Zay H100 and MareNostrum 5 have 4-GPU runs. Adastra MI250, LUMI-G and
Leonardo start at 8 GPUs, and the Llama-3.1-405B models only exist at 16 and 32 GPUs, so they
are missing under the default until you add those GPU counts (or use Scaling across GPUs).

## Running it as a shared server (localhost:8080)

`serve.py` keeps the dashboard online for many simultaneous users. All charts are computed in each
visitor's browser, so the server only hands out one cached, compressed file; it uses the Python
standard library only. It rebuilds the page by itself within ~30 seconds when a CSV in `data/inference/` or `data/training/`,
`defaults.json` or `template.html` changes, keeps serving the last good version if a rebuild
fails, and exposes `/healthz` for monitoring.

Use whichever container tool your site supports: Docker on a workstation or VM, Singularity/Apptainer
on HPC systems. Both run the same server.

**Docker.** From this folder:

```bash
docker compose up -d --build     # http://localhost:8080, restarts after crashes and reboots
docker compose logs -f
```

`data/` and `defaults.json` are mounted into the container, so edit them on the host and the live
dashboard updates without a rebuild. Plotly is bundled into the image, so users do not need
internet access.

**Singularity/Apptainer.** The image is described in
[`deploy/minerva-dashboard.def`](deploy/minerva-dashboard.def) and `deploy/singularity.sh` wraps the
commands (it uses `apptainer` or `singularity`, whichever is installed). From this folder:

```bash
FAKEROOT=1 deploy/singularity.sh build     # creates minerva-dashboard.sif (omit FAKEROOT=1 if you are root)
deploy/singularity.sh start                # background instance on http://<this-host>:8080
deploy/singularity.sh status               # instance + /healthz;  also: logs, stop
```

Singularity shares the host network, so there is no port mapping: pass another port with
`deploy/singularity.sh start 9090` if 8080 is taken. `data/` and `defaults.json` are mounted from the
host, so editing them updates the live dashboard within ~30 seconds without rebuilding the image.
The image is read-only and the server builds the page in memory, so nothing is written inside it.
Plotly is bundled in the image, so users do not need internet access. Building needs internet access
once (base image, pip, Plotly); on a site without it, build the `.sif` elsewhere and copy it over.

To keep it running across crashes and reboots, install
[`deploy/minerva-dashboard-singularity.service`](deploy/minerva-dashboard-singularity.service)
(edit the user, paths and the `singularity`/`apptainer` binary at the top of the file).

**Without a container.**

```bash
pip install -r requirements.txt
python fetch_plotly.py           # optional: serve Plotly locally for networks without internet
python serve.py --port 8080      # --host 127.0.0.1 to keep it private to this machine
```

To keep it running across crashes and reboots, install the systemd unit in
[`deploy/minerva-dashboard.service`](deploy/minerva-dashboard.service) (paths are at the top of the file),
which runs `serve.py` directly instead of the image.

`localhost:8080` is only reachable from the machine running it. Other users open
`http://<server-name-or-ip>:8080`, which needs the default `--host 0.0.0.0` and an open port 8080 in
the firewall. There is no login; put it behind your institution's reverse proxy or VPN if the
results should not be public.

## Training view

Reads the training summary files in [`data/training/`](data/training/): one or more CSVs with the
columns of `full_benchmark_summary_results_training_MareNostrum5_ACC.csv` (the final MareNostrum 5 results).
The inference loader ignores this folder, and a malformed or missing training file never breaks the
inference dashboard: Training then shows a message saying what is wrong.

**Synthetic data for a second cluster.** `data/training/example-synthetic-leonardo.csv` holds
**synthetic** Leonardo Booster results so that cross-cluster comparisons can be tried out. They are not
measurements: `tools/make_example_training_data.py` derives each run from a MareNostrum 5 run of the same
configuration (the full sweep of the small models and about 60% of the 70B/72B configurations, about 0.42x the
speed of an H100, MFU recomputed for the A100's 312 TFLOP/s, about 0.72x the power, some out-of-memory and
timeout failures) with a fixed random seed. The cluster is labelled *Leonardo Booster (synthetic)* in every
chart, table and legend. **Delete that file once real Leonardo results exist** (keep `tools/` if you want to
regenerate it).

**Comparing like with like.** Every run processes the same number of samples per optimizer step (the *global
batch*, 256 for almost all runs): the sweeps change the micro-batch per GPU and compensate with gradient
accumulation. What differs between runs is the framework, the parallelism, the micro-batch and the number of GPUs,
so mixing them in one bar is not a fair comparison. The **Compare like with like** buttons at the top of the
filters fix the cluster (MareNostrum 5, the real results), the model, the parallelism, the micro-batch and the GPU
count, and leave one thing free to compare:

| Button | Fixed | Compare |
|---|---|---|
| Mistral 7B, FSDP *(default)* | FSDP, micro-batch 2, 8 GPUs | accelerate vs torchrun, across clusters |
| Gemma 1B, DDP | DDP, micro-batch 2, 8 GPUs | accelerate vs torchrun, across clusters |
| Qwen 7B, Megatron | Megatron, micro-batch 2, 16 GPUs | TP/PP/DP layouts, across clusters |
| Llama 70B, Megatron | Megatron, micro-batch 2, 64 GPUs | TP/PP/DP layouts, across clusters |

A button also opens Compare with the right grouping. Any filter can still be changed, and a default **pauses
itself** when the chart varies by that dimension: a chart by cluster shows every cluster, a chart by GPUs shows
all GPU counts, a heatmap with models as rows shows all models. The filter panel then says "paused" and a note
explains. *Clear all* shows every run. The presets and the Scaling model live in `dashboard/defaults.json`
(`training.presets`, each with a `name`, `filters` and `compare` settings; the first one is the default).

**Failed runs.** A run with a `Comment` (an error message, or "Did not run") has no results. These rows are
hidden by default (turn on *Include failed runs* to see them) and classified as *Out of memory*, *Timeout*,
*NCCL error*, *Process killed*, *Did not run* or *Other error*. The Heatmap always shows them: an orange cell
means every run of that cell failed, and a ⚠ marks cells where some runs failed while others worked.

**Parallelism.** Megatron runs report the sizes of tensor (TP), pipeline (PP), data (DP), context (CP), expert
(EP) and sequence (SP) parallelism; DDP, FSDP and ZeRO runs report none of them. The dashboard therefore has
three levels you can filter or group by:

| Dimension | Values |
|---|---|
| *Parallelism family* | Data parallel (DDP), Sharded data parallel (FSDP, ZeRO-2, ZeRO-3), Tensor/pipeline parallel (Megatron), No parallelism |
| *Parallelism type* | The `TypeParallelism` column: ddp, fsdp, zero2, zero3, none, megatron |
| *Parallelism (TP/PP/DP)* | `TP4 PP2 DP4` for Megatron runs (TP x PP x DP = GPUs in every run), the type name for the others |

The TP, PP, DP, CP, EP and SP filters show `n/a` for runs where they do not apply. Dimensions that have the
same value in every run (here precision and max length) are left out of the filters and listed under the title.

| View | What it shows |
|---|---|
| Compare | Bars of any metric by cluster, framework, model, parallelism, GPUs and so on, with *Best run/Median/Mean*, *Best run by* and *Sort bars* |
| Scaling | A metric against GPUs, nodes, batch size or accumulation, one line per cluster and framework (or cluster, model, parallelism...). Dotted lines show ideal linear scaling (the *Ideal linear scaling* option). Entering Scaling selects `mistral_7b` unless a model is already chosen |
| Distribution | Box plots with every run as a dot, for example GPU utilization or MFU by framework and cluster |
| Trade-offs | One metric against another for every run (for example GPU utilization against MFU); marker shape shows the parallelism family |
| Heatmap | Rows are whatever group-by variables you tick (model, framework, parallelism, GPUs, nodes, batch, accumulation, dataset, cluster; default: model, framework, parallelism and GPUs) against GPUs, clusters, frameworks, batch size...; a *Cluster* selector filters to one cluster, and with several clusters shown each one gets its own labelled block of columns (cluster, then the Columns value inside it) instead of mixing clusters in one cell; *Ratio of two columns* gives Leonardo ÷ MareNostrum 5 per row. Failed cells in orange, ⚠ for partial failures |
| Data | Sortable table with the failure kind, the full comment and the source directory of every run, and CSV download. Below the table, a *Python notebook* runs pandas and matplotlib in the browser (Pyodide, loaded from jsDelivr on the first Run, so it works on GitHub Pages and needs internet): `df` is the runs the filters show, `all_runs` is every run, and there are example cells to start from |

Metrics: training throughput, throughput per GPU, **achieved TFLOP/s per GPU**, **MFU**, total TFLOP/s,
**GPU utilization** (average and peak), time per epoch, per step and in total, power (average and peak),
energy to train (kWh), tokens per joule, **GFLOP/s per watt**, GPU memory (average and peak) and loss.
`Avg. GPU FLOPs` is read as TFLOP/s per GPU: dividing it by `Avg. GPU MFU` gives 989 on every MareNostrum 5
row, the H100 BF16 peak. Metrics with no values in the data (validation loss at the moment) are hidden.
Because FLOP/s and MFU are normalised by the hardware, they are the metrics to compare across models and
clusters; tokens/s is only comparable for the same model.

`GPUs per Node` is not read: in the final results it is 1 for FSDP, ZeRO and Megatron runs although every GPU
of each node is used (total GPUs = 4 x nodes). Runs that never started have no GPU count; it is set to nodes x
the GPUs per node of that system. The yellow note warns when a chart mixes clusters, models, datasets,
frameworks, parallelism strategies, GPU counts or batch sizes.

## Updating the data

Add or replace inference CSVs in `data/inference/` (same 29-column format) and training CSVs in `data/training/`, then:

```bash
pip install -r requirements.txt
python build_dashboard.py      # rewrites dashboard/index.html
```

`minerva_data.py` is the single place where CSVs are loaded and cleaned (`load_runs` for inference, `load_training` for training). Import it from
any analysis script: `from minerva_data import load_runs`.

| Cleaning step | Why |
|---|---|
| Rows with `Framework=results`, `Dataset=vllm` are set to framework `vllm`, dataset `unknown` | Result-path parsing shifted by one level (45 rows, Adastra MI300). The dataset cannot be recovered; fix upstream if possible. |
| Zero-throughput runs and `inf` scores are flagged `failed`, their scores set to empty, and hidden by default | 14 runs; they otherwise break averages and score charts. |
| Added `j_per_tok` and `tok_s_per_gpu` | Derived from throughput, power and GPU count. |
| Training: failed = has a comment or no throughput; added `parallel_config`, `parallel_family`, `failure`, `tok_s_gpu`, `tflops_total`, `gflops_w`, `kwh`, `tok_per_j` | Derived from the parallelism columns, the comment, throughput, FLOPs, power per GPU, GPU count and hours. |

## Files

| Path | Purpose |
|---|---|
| `data/training/` | Training summary CSVs: the final MareNostrum 5 results and a synthetic Leonardo example. |
| `tools/make_example_training_data.py` | Regenerates the synthetic Leonardo example (seeded). |
| `dashboard/defaults.json` | Inference filters applied when the dashboard opens, the Scaling model, and the Training comparisons (presets). |
| `dashboard/assets/minerva-logo.png` | Logo shown in the header and as the browser icon; embedded into `index.html` at build time. Replace the file and rebuild to change it. |
| `dashboard/template.html` | Dashboard source (HTML, CSS, JS). Edit this, then rebuild. |
| `dashboard/index.html` | Generated, self-contained dashboard. Commit it for GitHub Pages. |
| `serve.py` | Always-on server with automatic rebuilds (see above). |
| `fetch_plotly.py` | Downloads Plotly.js for offline serving. |
| `Dockerfile`, `docker-compose.yml` | Docker image and compose file (see above). |
| `deploy/` | Singularity image definition (`minerva-dashboard.def`), helper script (`singularity.sh`) and systemd units. |
| `build_dashboard.py` | CSVs to `index.html`. |
| `minerva_data.py` | Shared loader and cleaner. |

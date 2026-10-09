# Minerva analytics

Explore the inference benchmark results in [`data/`](data/) without installing anything:
open [`dashboard/index.html`](dashboard/index.html) in a browser.

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
standard library only. It rebuilds the page by itself within ~30 seconds when a CSV in `data/`,
`defaults.json` or `template.html` changes, keeps serving the last good version if a rebuild
fails, and exposes `/healthz` for monitoring.

**Docker (recommended).** From this folder:

```bash
docker compose up -d --build     # http://localhost:8080, restarts after crashes and reboots
docker compose logs -f
```

`data/` and `defaults.json` are mounted into the container, so edit them on the host and the live
dashboard updates without a rebuild. Plotly is bundled into the image, so users do not need
internet access.

**Without Docker.**

```bash
pip install -r requirements.txt
python fetch_plotly.py           # optional: serve Plotly locally for networks without internet
python serve.py --port 8080      # --host 127.0.0.1 to keep it private to this machine
```

To keep it running across crashes and reboots, install the systemd unit in
[`deploy/minerva-dashboard.service`](deploy/minerva-dashboard.service) (paths are at the top of the file).

`localhost:8080` is only reachable from the machine running it. Other users open
`http://<server-name-or-ip>:8080`, which needs the default `--host 0.0.0.0` and an open port 8080 in
the firewall. There is no login; put it behind your institution's reverse proxy or VPN if the
results should not be public.

## Updating the data

Add or replace inference CSVs in `data/` (same 29-column format) and training CSVs in `data/training/`, then:

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
| Training: failed = no throughput; added `tok_s_gpu`, `kwh`, `tok_per_j` | Derived from throughput, power per GPU, GPU count and hours. |

## Files

| Path | Purpose |
|---|---|
| `data/training/` | Training summary CSVs (currently one example file for MareNostrum 5 ACC). |
| `dashboard/defaults.json` | Filters applied when the dashboard opens, the Scaling model, and the Training settings. |
| `dashboard/assets/minerva-logo.png` | Logo shown in the header and as the browser icon; embedded into `index.html` at build time. Replace the file and rebuild to change it. |
| `dashboard/template.html` | Dashboard source (HTML, CSS, JS). Edit this, then rebuild. |
| `dashboard/index.html` | Generated, self-contained dashboard. Commit it for GitHub Pages. |
| `serve.py` | Always-on server with automatic rebuilds (see above). |
| `fetch_plotly.py` | Downloads Plotly.js for offline serving. |
| `Dockerfile`, `docker-compose.yml`, `deploy/` | Deployment with Docker or systemd. |
| `build_dashboard.py` | CSVs to `index.html`. |
| `minerva_data.py` | Shared loader and cleaner. |


---

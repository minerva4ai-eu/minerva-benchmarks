from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

# -----------------------------
# Load and normalize CSVs
# -----------------------------
DATA_DIR = Path("data")


def find_col(df, possibilities):
    cols = {c.lower(): c for c in df.columns}
    for p in possibilities:
        if p.lower() in cols:
            return cols[p.lower()]
    for p in possibilities:
        for c_lower, c in cols.items():
            if p.lower() in c_lower:
                return c
    return None


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    canonical = {
        "supercomputer": ["SUPCOMPUTER", "supercomputer", "system", "site"],
        "partition": ["partition"],
        "model": ["model"],
        "dataset": ["dataset"],
        "framework": ["framework"],
        "concurrency_level": ["concurrency level", "concurrency_level", "concurrency"],
        "number_of_nodes": ["number of nodes", "nodes"],
        "total_used_gpus": [
            "Number of GPUs",
            "number of gpus",
            "number_of_gpus",
            "total used gpus",
            "total_used_gpus",
            "total_gpus",
        ],
        "power_avg_w": ["power usage avg (w)", "power_avg_w", "power avg (w)"],
        "output_throughput_toks_s": [
            "output throughput (tokens/s)",
            "output throughput",
            "tokens/s",
        ],
        "request_throughput_reqs_s": [
            "request throughput (requests/s)",
            "request throughput",
            "requests/s",
        ],
        "itl_ms": ["itl (ms)", "itl_ms", "itl"],
        "tpot_ms": ["tpot (ms)", "tpot_ms", "tpot"],
        "ttft_ms": ["ttft (ms)", "ttft_ms", "ttft"],
        "tensor_parallelism": ["Tensor", "Tensor Parallelism", "Tensor Parallel", "tensor_parallelism", "tensor parallel"],
        "pipeline_parallelism": ["Pipeline", "Pipeline Parallelism", "Pipeline Parallel", "pipeline_parallelism", "pipeline parallel"],
        "max_model_length": ["Max Model Length", "Max Length", "Max_Model_Length", "max_model_length", "max length"],
    }
    rename_map = {}
    for canon, variants in canonical.items():
        found = find_col(df, variants)
        if found:
            rename_map[found] = canon
    df = df.rename(columns=rename_map)
    if "supercomputer" not in df.columns:
        df["supercomputer"] = path.stem
    return df


# Load CSVs
files = sorted(DATA_DIR.glob("*.csv"))
if not files:
    raise RuntimeError("No CSV files found in data/")

dfs = [load_and_normalize(f) for f in files]
df_full = pd.concat(dfs, ignore_index=True)

# Numeric columns
numeric_cols = [
    "output_throughput_toks_s",
    "request_throughput_reqs_s",
    "power_avg_w",
    "concurrency_level",
    "number_of_nodes",
    "total_used_gpus",
    "tensor_parallelism",
    "pipeline_parallelism",
    "max_model_length",
]

for c in numeric_cols:
    if c in df_full.columns:
        df_full[c] = pd.to_numeric(df_full[c], errors="coerce")

# Derived metric
if {"output_throughput_toks_s", "power_avg_w"}.issubset(df_full.columns):
    df_full["tokens_per_watt"] = (
        df_full["output_throughput_toks_s"] / df_full["power_avg_w"]
    )
else:
    df_full["tokens_per_watt"] = np.nan

# Fill categorical NaNs
for col in ["supercomputer", "partition", "model", "dataset", "framework"]:
    if col in df_full.columns:
        df_full[col] = df_full[col].fillna("Unknown")

# -----------------------------
# Dash app
# -----------------------------
app = dash.Dash(__name__)
app.title = "MINERVA AI Benchmarks - Inference - Supercomputers Dashboard"

# Add concurrency_level to filter_cols
filter_cols = [
    "supercomputer",
    "partition",
    "model",
    "dataset",
    "framework",
    "concurrency_level",
    "total_used_gpus",
    "number_of_nodes",
    "tensor_parallelism",
    "pipeline_parallelism",
    "max_model_length",
]


def unique_sorted(col):
    if col not in df_full.columns:
        return []
    vals = df_full[col].dropna().unique()
    nums = sorted([v for v in vals if isinstance(v, (int, float))])
    strs = sorted([str(v) for v in vals if not isinstance(v, (int, float))])
    return nums + strs


# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div(
    [
        html.H1("Supercomputer Benchmark Dashboard"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(col.replace("_", " ").title()),
                        dcc.Dropdown(
                            id=f"filter-{col}",
                            options=[
                                {"label": str(v), "value": v}
                                for v in ["All"] + unique_sorted(col)
                            ],
                            value=filter_defaults[col],
                            multi=True,
                        ),
                    ],
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "marginRight": "1%",
                    },
                )
                for col in filter_cols
            ]
        ),
        html.Hr(),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Scatter: Throughput vs Power",
                    children=[dcc.Graph(id="scatter-thr-power")],
                ),
                dcc.Tab(
                    label="Scatter: Latency vs Throughput",
                    children=[
                        html.Div(
                            [
                                html.Label("Latency Metric"),
                                dcc.Dropdown(
                                    id="latency-metric",
                                    options=[
                                        {"label": "ITL (ms)", "value": "itl_ms"},
                                        {"label": "TPOT (ms)", "value": "tpot_ms"},
                                        {"label": "TTFT (ms)", "value": "ttft_ms"},
                                    ],
                                    value="itl_ms",
                                    clearable=False,
                                ),
                                html.Label("Throughput Metric"),
                                dcc.Dropdown(
                                    id="throughput-metric",
                                    options=[
                                        {
                                            "label": "Output Throughput (tokens/s)",
                                            "value": "output_throughput_toks_s",
                                        },
                                        {
                                            "label": "Request Throughput (requests/s)",
                                            "value": "request_throughput_reqs_s",
                                        },
                                    ],
                                    value="output_throughput_toks_s",
                                    clearable=False,
                                ),
                                dcc.Graph(id="scatter-latency-vs-throughput"),
                            ],
                            style={"marginTop": "20px", "marginBottom": "20px"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="Concurrency Curve",
                    children=[dcc.Graph(id="line-concurrency")],
                ),
                dcc.Tab(
                    label="Barplot: Mean Throughput",
                    children=[dcc.Graph(id="bar-throughput")],
                ),
                dcc.Tab(
                    label="Barplot: Tokens per Watt", children=[dcc.Graph(id="bar-tpw")]
                ),
                dcc.Tab(
                    label="HeatMap: Scalability Nodes",
                    children=[dcc.Graph(id="heatmap-scalability-nodes-thr")],
                ),
                dcc.Tab(
                    label="HeatMap: Scalability GPUs",
                    children=[dcc.Graph(id="heatmap-scalability-gpus-thr")],
                ),
                dcc.Tab(
                    label="Filtered Table", children=[html.Div(id="table-container")]
                ),
            ]
        ),
    ]
)


# -----------------------------
# Helper: make heatmap
# -----------------------------
def make_heatmap(df, metric="output_throughput_toks_s", xaxis_col="number_of_nodes"):
    if {metric, xaxis_col}.issubset(df.columns):
        df["sc_partition"] = df["supercomputer"] + " | " + df["partition"]
        agg = df.groupby(["sc_partition", xaxis_col])[metric].mean().reset_index()
        heatmap_data = agg.pivot(index="sc_partition", columns=xaxis_col, values=metric)
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale="Blues",
                text=heatmap_data.values,
                texttemplate="%{text:.2f}",
                textfont={"color": "grey"},
                colorbar=dict(title=metric),
            )
        )
        fig.update_layout(
            xaxis_title=xaxis_col.replace("_", " ").title(),
            yaxis_title="Supercomputer | Partition",
            title=f"Scalability Heatmap ({metric})",
            margin=dict(t=50),
        )
    else:
        fig = go.Figure()
    return fig


# -----------------------------
# Callback
# -----------------------------
@app.callback(
    [
        Output("scatter-thr-power", "figure"),
        Output("scatter-latency-vs-throughput", "figure"),
        Output("line-concurrency", "figure"),
        Output("bar-throughput", "figure"),
        Output("bar-tpw", "figure"),
        Output("heatmap-scalability-nodes-thr", "figure"),
        Output("heatmap-scalability-gpus-thr", "figure"),
        Output("table-container", "children"),
    ],
    [Input(f"filter-{col}", "value") for col in filter_cols]
    + [Input("latency-metric", "value"), Input("throughput-metric", "value")],
)
def update_dashboard(*args):
    df = df_full.copy()
    filter_values = args[: len(filter_cols)]
    latency_metric = args[len(filter_cols)]
    throughput_metric = args[len(filter_cols) + 1]

    # Apply filters
    for col, selected in zip(filter_cols, filter_values):
        if selected and "All" not in selected:
            if col in [
                "total_used_gpus",
                "number_of_nodes",
                "concurrency_level",
                "tensor_parallelism",
                "pipeline_parallelism",
                "max_model_length",
            ]:
                selected_numeric = []
                for s in selected:
                    try:
                        selected_numeric.append(int(s))
                    except ValueError:
                        selected_numeric.append(float(s))
                df = df[df[col].isin(selected_numeric)]
            else:
                df = df[df[col].isin(selected)]

    if df.empty:
        empty_fig = go.Figure()
        return [empty_fig] * 7 + ["No rows match filters"]

    # Scatter: throughput vs power
    fig_scatter = (
        px.scatter(
            df,
            x="power_avg_w",
            y="output_throughput_toks_s",
            color="supercomputer",
            hover_data=["model", "dataset", "total_used_gpus"],
        )
        if {"output_throughput_toks_s", "power_avg_w"}.issubset(df.columns)
        else go.Figure()
    )

    # Latency vs Throughput
    fig_latency_thr = (
        px.scatter(
            df,
            x=throughput_metric,
            y=latency_metric,
            color="supercomputer",
            hover_data=["model", "dataset", "total_used_gpus"],
        )
        if {latency_metric, throughput_metric}.issubset(df.columns)
        else go.Figure()
    )

    # Concurrency curve
    fig_line = go.Figure()
    if {
        "concurrency_level",
        "output_throughput_toks_s",
        "supercomputer",
        "partition",
        "framework",
    }.issubset(df.columns):
        for (sc, part, fw), group in df.groupby(["supercomputer", "partition", "framework"]):
            agg = (
                group.groupby("concurrency_level")["output_throughput_toks_s"]
                .mean()
                .reset_index()
            )
            fig_line.add_trace(
                go.Scatter(
                    x=agg["concurrency_level"],
                    y=agg["output_throughput_toks_s"],
                    mode="lines+markers",
                    name=f"{fw} | {sc} | {part}",
                )
            )
        fig_line.update_layout(
            xaxis_title="Concurrency Level",
            yaxis_title="Avg Output Throughput (tokens/s)",
            title="Concurrency Curve (Supercomputer | Partition | Framework)",
            margin=dict(t=40),
        )

    # Barplots
    if "output_throughput_toks_s" in df.columns and "model" in df.columns:
        if df["model"].nunique() > 1:
            agg = (
                df.groupby(["model", "supercomputer"])["output_throughput_toks_s"]
                .mean()
                .reset_index()
            )
            fig_bar = px.bar(
                agg,
                x="model",
                y="output_throughput_toks_s",
                color="supercomputer",
                barmode="group",
            )
            fig_bar.update_layout(
                xaxis_title="Model",
                yaxis_title="Mean Throughput (tokens/s)",
                title="Mean Throughput per Model & Supercomputer",
                margin=dict(t=50),
            )
        else:
            agg = (
                df.groupby("supercomputer")["output_throughput_toks_s"]
                .mean()
                .reset_index()
            )
            fig_bar = px.bar(agg, x="supercomputer", y="output_throughput_toks_s")
            fig_bar.update_layout(
                xaxis_title="Supercomputer",
                yaxis_title="Mean Throughput (tokens/s)",
                title="Mean Throughput per Supercomputer",
                margin=dict(t=50),
            )
    else:
        fig_bar = go.Figure()

    # Tokens per watt
    if "tokens_per_watt" in df.columns and "model" in df.columns:
        if df["model"].nunique() > 1:
            agg = (
                df.groupby(["model", "supercomputer"])["tokens_per_watt"]
                .mean()
                .reset_index()
            )
            fig_tpw = px.bar(
                agg,
                x="model",
                y="tokens_per_watt",
                color="supercomputer",
                barmode="group",
            )
            fig_tpw.update_layout(
                xaxis_title="Model",
                yaxis_title="Mean Tokens per Watt",
                title="Mean Tokens per Watt per Model & Supercomputer",
                margin=dict(t=50),
            )
        else:
            agg = df.groupby("supercomputer")["tokens_per_watt"].mean().reset_index()
            fig_tpw = px.bar(agg, x="supercomputer", y="tokens_per_watt")
            fig_tpw.update_layout(
                xaxis_title="Supercomputer",
                yaxis_title="Mean Tokens per Watt",
                title="Mean Tokens per Watt per Supercomputer",
                margin=dict(t=50),
            )
    else:
        fig_tpw = go.Figure()

    # Heatmaps
    fig_heat_nodes = make_heatmap(
        df, metric="output_throughput_toks_s", xaxis_col="number_of_nodes"
    )
    fig_heat_gpus = make_heatmap(
        df, metric="output_throughput_toks_s", xaxis_col="total_used_gpus"
    )

    # Table
    table_html = html.Div(
        [
            html.H5("Filtered Data (first 200 rows)"),
            dash.dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df.columns],
                data=df.head(200).to_dict("records"),
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "font_size": "12px"},
            ),
        ]
    )

    return (
        fig_scatter,
        fig_latency_thr,
        fig_line,
        fig_bar,
        fig_tpw,
        fig_heat_nodes,
        fig_heat_gpus,
        table_html,
    )


# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

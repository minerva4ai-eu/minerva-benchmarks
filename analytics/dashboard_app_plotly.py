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
        "supercomputer": ["supercomputer", "system", "site"],
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

filter_defaults = {
    "supercomputer": ["All"],
    "partition": ["All"],
    "model": ["All"],
    "dataset": ["All"],
    "concurrency_level": ["All"],
    "framework": ["vllm"],
    "tensor_parallelism": [4],
    "pipeline_parallelism": [1],
    "max_model_length": [4096],
    "number_of_nodes": [1],
    "total_used_gpus": [4],
}

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
    # fig_scatter = (
    #     px.scatter(
    #         df,
    #         x="power_avg_w",
    #         y="output_throughput_toks_s",
    #         color="supercomputer",
    #         hover_data=["model", "dataset", "total_used_gpus"],
    #     )
    #     if {"output_throughput_toks_s", "power_avg_w"}.issubset(df.columns)
    #     else go.Figure()
    # )
    df["system_partition_framework"] = (
        df["supercomputer"].astype(str)
        + " | "
        + df["partition"].astype(str)
        + " | "
        + df["framework"].astype(str)
    )

    fig_scatter = (
        px.scatter(
            df,
            x="power_avg_w",
            y="output_throughput_toks_s",

            color="system_partition_framework",
            symbol="framework",

            facet_col="dataset",

            # size="total_used_gpus",

            hover_data=[
                "model",
                "framework",
                "dataset",
                "total_used_gpus",
            ],
        )
        if {"output_throughput_toks_s", "power_avg_w"}.issubset(df.columns)
        else go.Figure()
    )

    fig_scatter.update_layout(
        title="Throughput vs Power",
        margin=dict(t=60),
    )

    # # Latency vs Throughput
    # fig_latency_thr = (
    #     px.scatter(
    #         df,
    #         x=throughput_metric,
    #         y=latency_metric,
    #         color="supercomputer",
    #         hover_data=["model", "dataset", "total_used_gpus"],
    #     )
    #     if {latency_metric, throughput_metric}.issubset(df.columns)
    #     else go.Figure()
    # )

    fig_latency_thr = (
        px.scatter(
            df,
            x=throughput_metric,
            y=latency_metric,

            color="system_partition_framework",
            symbol="framework",

            # facet_row="partition",
            facet_col="dataset",

            # size="total_used_gpus",

            hover_data=[
                "model",
                "framework",
                "dataset",
                "total_used_gpus",
            ],
        )
        if {latency_metric, throughput_metric}.issubset(df.columns)
        else go.Figure()
    )

    fig_latency_thr.update_layout(
        title="Latency vs Throughput (grouped by system, partition, framework, dataset)",
        margin=dict(t=60),
    )

    # Concurrency curve
    fig_line = go.Figure()

    if {
        "concurrency_level",
        "output_throughput_toks_s",
        "supercomputer",
        "partition",
        "framework",
        "model",
        "dataset",
    }.issubset(df.columns):

        for (sc, part, fw, md, dt), group in df.groupby(
            ["supercomputer", "partition", "framework", "model", "dataset"]
        ):
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
                    name=f"{sc} | {fw} | {md} | {dt} | {part}",
                )
            )

        fig_line.update_layout(
            xaxis_title="Concurrency Level",
            yaxis_title="Avg Output Throughput (tokens/s)",
            title="Concurrency Curve (all configurations)",
            margin=dict(t=40),
            showlegend=True,
            legend=dict(
                orientation="v",
                tracegroupgap=5
            ),
        )

    # # Barplots
    # if "output_throughput_toks_s" in df.columns and "model" in df.columns:
    #     if df["model"].nunique() > 1:
    #         agg = (
    #             df.groupby(["model", "supercomputer", "framework", "dataset"])["output_throughput_toks_s"]
    #             .mean()
    #             .reset_index()
    #         )
    #         fig_bar = px.bar(
    #             agg,
    #             x="model",
    #             y="output_throughput_toks_s",
    #             color="supercomputer",
    #             barmode="group",
    #         )
    #         fig_bar.update_layout(
    #             xaxis_title="Model",
    #             yaxis_title="Mean Throughput (tokens/s)",
    #             title="Mean Throughput per Model & Supercomputer",
    #             margin=dict(t=50),
    #         )
    #     else:
    #         agg = (
    #             df.groupby("supercomputer")["output_throughput_toks_s"]
    #             .mean()
    #             .reset_index()
    #         )
    #         fig_bar = px.bar(agg, x="supercomputer", y="output_throughput_toks_s")
    #         fig_bar.update_layout(
    #             xaxis_title="Supercomputer",
    #             yaxis_title="Mean Throughput (tokens/s)",
    #             title="Mean Throughput per Supercomputer",
    #             margin=dict(t=50),
    #         )
    # else:
    #     fig_bar = go.Figure()

    # Barplots split by model, supercomputer, framework, and dataset
    group_cols = ["model", "supercomputer", "framework", "dataset"]

    available_cols = [c for c in group_cols if c in df.columns]

    if "output_throughput_toks_s" in df.columns and available_cols:

        agg = (
            df.groupby(available_cols)["output_throughput_toks_s"]
            .mean()
            .reset_index()
        )

        fig_bar = px.bar(
            agg,
            x="framework" if "framework" in available_cols else available_cols[0],
            y="output_throughput_toks_s",
            color="supercomputer" if "supercomputer" in available_cols else None,
            barmode="group",
            facet_row="model" if "model" in available_cols else None,
            facet_col="dataset" if "dataset" in available_cols else None,
        )

        fig_bar.update_layout(
            title="Mean Throughput split by Model, Supercomputer, Framework, and Dataset",
            xaxis_title="Framework",
            yaxis_title="Mean Throughput (tokens/s)",
            margin=dict(t=80),
            height=300 * max(1, agg["model"].nunique() if "model" in agg.columns else 1),
        )

        fig_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    else:
        fig_bar = go.Figure()

    # # Tokens per watt
    # if "tokens_per_watt" in df.columns and "model" in df.columns:
    #     if df["model"].nunique() > 1:
    #         agg = (
    #             df.groupby(["model", "supercomputer"])["tokens_per_watt"]
    #             .mean()
    #             .reset_index()
    #         )
    #         fig_tpw = px.bar(
    #             agg,
    #             x="model",
    #             y="tokens_per_watt",
    #             color="supercomputer",
    #             barmode="group",
    #         )
    #         fig_tpw.update_layout(
    #             xaxis_title="Model",
    #             yaxis_title="Mean Tokens per Watt",
    #             title="Mean Tokens per Watt per Model & Supercomputer",
    #             margin=dict(t=50),
    #         )
    #     else:
    #         agg = df.groupby("supercomputer")["tokens_per_watt"].mean().reset_index()
    #         fig_tpw = px.bar(agg, x="supercomputer", y="tokens_per_watt")
    #         fig_tpw.update_layout(
    #             xaxis_title="Supercomputer",
    #             yaxis_title="Mean Tokens per Watt",
    #             title="Mean Tokens per Watt per Supercomputer",
    #             margin=dict(t=50),
    #         )
    # else:
    #     fig_tpw = go.Figure()

    # Tokens per watt split by model, supercomputer, framework, and dataset
    group_cols = ["model", "supercomputer", "framework", "dataset"]

    available_cols = [c for c in group_cols if c in df.columns]

    if "tokens_per_watt" in df.columns and available_cols:

        agg = (
            df.groupby(available_cols)["tokens_per_watt"]
            .mean()
            .reset_index()
        )

        fig_tpw = px.bar(
            agg,
            x="framework" if "framework" in available_cols else available_cols[0],
            y="tokens_per_watt",
            color="supercomputer" if "supercomputer" in available_cols else None,
            barmode="group",
            facet_row="model" if "model" in available_cols else None,
            facet_col="dataset" if "dataset" in available_cols else None,
        )

        fig_tpw.update_layout(
            title="Mean Tokens per Watt split by Model, Supercomputer, Framework, and Dataset",
            xaxis_title="Framework",
            yaxis_title="Mean Tokens per Watt",
            margin=dict(t=80),
            height=300 * max(1, agg["model"].nunique() if "model" in agg.columns else 1),
        )

        fig_tpw.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    else:
        fig_tpw = go.Figure()

    # # Heatmaps
    # fig_heat_nodes = make_heatmap(
    #     df, metric="output_throughput_toks_s", xaxis_col="number_of_nodes"
    # )
    # HeatMap: Scalability Nodes
    df["sc_model_fw"] = (
        df["supercomputer"].astype(str)
        + " | "
        + df["model"].astype(str)
        + " | "
        + df["framework"].astype(str)
    )

    required_cols = [
        "output_throughput_toks_s",
        "number_of_nodes",
    ]

    if all(c in df.columns for c in required_cols):

        agg = (
            df.groupby([
                "sc_model_fw",
                "dataset",
                "number_of_nodes",
            ])["output_throughput_toks_s"]
            .mean()
            .reset_index()
        )

        # -------------------------------------------------
        # Extract sorting keys
        # -------------------------------------------------
        split_cols = agg["sc_model_fw"].str.split(" | ", regex=False)

        agg["system_sort"] = split_cols.str[0]
        agg["model_sort"] = split_cols.str[1]
        agg["framework_sort"] = split_cols.str[2]

        # -------------------------------------------------
        # Final row ordering
        # -------------------------------------------------
        ordered_rows = (
            agg[[
                "model_sort",
                "framework_sort",
                "system_sort",
                "sc_model_fw",
            ]]
            .drop_duplicates()
            .sort_values([
                "model_sort",
                "framework_sort",
                "system_sort",
                "sc_model_fw",
            ])
            ["sc_model_fw"]
            .tolist()
        )

        datasets = sorted(agg["dataset"].dropna().unique())
        initial_dataset = datasets[0]

        # -------------------------------------------------
        # Heatmap (initial)
        # -------------------------------------------------
        fig_heat_nodes = px.density_heatmap(
            agg[agg["dataset"] == initial_dataset],

            x="number_of_nodes",
            y="sc_model_fw",

            z="output_throughput_toks_s",
            histfunc="avg",

            text_auto=".2f",
            color_continuous_scale="Viridis",
        )

        # -------------------------------------------------
        # X axis ordering
        # -------------------------------------------------
        fig_heat_nodes.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=sorted(
                agg["number_of_nodes"].dropna().unique(),
                key=lambda x: int(x)
            ),
        )

        # -------------------------------------------------
        # Y axis ordering (model → framework → system)
        # -------------------------------------------------
        fig_heat_nodes.update_yaxes(
            categoryorder="array",
            categoryarray=ordered_rows,
        )

        # -------------------------------------------------
        # Dataset dropdown
        # -------------------------------------------------
        buttons = []

        for dataset in datasets:

            df_dataset = agg[agg["dataset"] == dataset]

            temp_fig = px.density_heatmap(
                df_dataset,

                x="number_of_nodes",
                y="sc_model_fw",

                z="output_throughput_toks_s",
                histfunc="avg",

                text_auto=".2f",
                color_continuous_scale="Viridis",
            )

            temp_fig.update_yaxes(
                categoryorder="array",
                categoryarray=ordered_rows,
            )

            buttons.append(
                dict(
                    label=str(dataset),
                    method="update",
                    args=[
                        {
                            "z": [trace.z for trace in temp_fig.data],
                            "x": [trace.x for trace in temp_fig.data],
                            "y": [trace.y for trace in temp_fig.data],
                        },
                        {
                            "title": f"Throughput Heatmap — Dataset: {dataset}"
                        },
                    ],
                )
            )

        # -------------------------------------------------
        # Layout
        # -------------------------------------------------
        fig_heat_nodes.update_layout(
            title=f"Throughput Heatmap — Dataset: {initial_dataset}",
            xaxis_title="Number of Nodes",
            yaxis_title="Supercomputer | Model | Framework",
            margin=dict(t=120),
            height=max(500, 35 * agg["sc_model_fw"].nunique()),

            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="right",
                    showactive=True,
                    x=0.0,
                    y=1.15,
                )
            ],
        )

    else:
        fig_heat_nodes = go.Figure()
    
    # Heatmap: Scalability GPUs
    # fig_heat_gpus = make_heatmap(
    #     df, metric="output_throughput_toks_s", xaxis_col="total_used_gpus"
    # )
    
    # Heatmap: Scalability GPUs
    df["sc_model_fw"] = (
        df["supercomputer"].astype(str)
        + " | "
        + df["model"].astype(str)
        + " | "
        + df["framework"].astype(str)
    )

    required_cols = [
        "output_throughput_toks_s",
        "total_used_gpus",
    ]

    if all(c in df.columns for c in required_cols):

        # Clean GPU column
        df["total_used_gpus"] = pd.to_numeric(
            df["total_used_gpus"], errors="coerce"
        ).astype("Int64")

        agg = (
            df.groupby([
                "sc_model_fw",
                "dataset",
                "total_used_gpus",
            ])["output_throughput_toks_s"]
            .mean()
            .reset_index()
        )

        agg["total_used_gpus"] = agg["total_used_gpus"].astype(int)

        # -------------------------------------------------
        # Sorting keys (model → framework → system)
        # -------------------------------------------------
        split_cols = agg["sc_model_fw"].str.split(" | ", regex=False)

        agg["system_sort"] = split_cols.str[0]
        agg["model_sort"] = split_cols.str[1]
        agg["framework_sort"] = split_cols.str[2]

        ordered_rows = (
            agg[[
                "model_sort",
                "framework_sort",
                "system_sort",
                "sc_model_fw",
            ]]
            .drop_duplicates()
            .sort_values([
                "model_sort",
                "framework_sort",
                "system_sort",
                "sc_model_fw",
            ])
            ["sc_model_fw"]
            .tolist()
        )

        datasets = sorted(agg["dataset"].dropna().unique())
        initial_dataset = datasets[0]

        # -------------------------------------------------
        # Initial heatmap (NO facets)
        # -------------------------------------------------
        fig_heat_gpus = px.density_heatmap(
            agg[agg["dataset"] == initial_dataset],

            x="total_used_gpus",
            y="sc_model_fw",

            z="output_throughput_toks_s",
            histfunc="avg",

            text_auto=".2f",
            color_continuous_scale="Viridis",
        )

        # -------------------------------------------------
        # GPU axis ordering (keep 2^n style if present)
        # -------------------------------------------------
        gpu_scale = sorted(
            agg["total_used_gpus"].dropna().unique().tolist()
        )

        gpu_scale = [
            g for g in gpu_scale
            if g > 0 and (g & (g - 1)) == 0
        ]

        fig_heat_gpus.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=gpu_scale,
        )

        # -------------------------------------------------
        # Y-axis ordering
        # -------------------------------------------------
        fig_heat_gpus.update_yaxes(
            categoryorder="array",
            categoryarray=ordered_rows,
        )

        # -------------------------------------------------
        # Dataset dropdown
        # -------------------------------------------------
        buttons = []

        for dataset in datasets:

            df_dataset = agg[agg["dataset"] == dataset]

            temp_fig = px.density_heatmap(
                df_dataset,

                x="total_used_gpus",
                y="sc_model_fw",

                z="output_throughput_toks_s",
                histfunc="avg",

                text_auto=".2f",
                color_continuous_scale="Viridis",
            )

            temp_fig.update_yaxes(
                categoryorder="array",
                categoryarray=ordered_rows,
            )

            buttons.append(
                dict(
                    label=str(dataset),
                    method="update",
                    args=[
                        {
                            "z": [trace.z for trace in temp_fig.data],
                            "x": [trace.x for trace in temp_fig.data],
                            "y": [trace.y for trace in temp_fig.data],
                        },
                        {
                            "title": f"GPU Scalability Heatmap — Dataset: {dataset}"
                        },
                    ],
                )
            )

        # -------------------------------------------------
        # Layout
        # -------------------------------------------------
        fig_heat_gpus.update_layout(
            title=f"GPU Scalability Heatmap — Dataset: {initial_dataset}",
            xaxis_title="Total Used GPUs",
            yaxis_title="Supercomputer | Model | Framework",
            margin=dict(t=120),
            height=max(500, 35 * agg["sc_model_fw"].nunique()),

            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="right",
                    showactive=True,
                    x=0.0,
                    y=1.15,
                )
            ],
        )

    else:
        fig_heat_gpus = go.Figure()

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

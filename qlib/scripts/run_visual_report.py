#!/usr/bin/env python3
"""
Generate an interactive HTML report from a finished Qlib run (MLflow directory).

Example:
    python scripts/run_visual_report.py examples/mlruns/<exp>/<run>
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def _read_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_pickle(path)


def _figure_to_section(title: str, fig: go.Figure) -> str:
    html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
    return f"<section><h2>{title}</h2>{html}</section>"


def _label_from_dir(run_dir: Path) -> str:
    # Use the dirname without the trailing run-id suffix; expected pattern: <name>_<hash>
    base = run_dir.name.rsplit("_", 1)[0]
    # Keep only letters and underscores to satisfy "仅英文" 需求（下划线视为分隔符）
    cleaned = re.sub(r"[^A-Za-z_]", "", base)
    return cleaned or base


def _cum_return_fig(report_dfs: List[Tuple[str, pd.DataFrame]]) -> go.Figure:
    fig = go.Figure()
    bench_added = False
    for label, report_df in report_dfs:
        cum_df = report_df[["return", "bench"]].cumsum()
        fig.add_trace(
            go.Scatter(
                x=cum_df.index,
                y=cum_df["return"],
                mode="lines",
                name=label,
            )
        )
        if not bench_added:
            fig.add_trace(
                go.Scatter(
                    x=cum_df.index,
                    y=cum_df["bench"],
                    mode="lines",
                    name="Benchmark",
                )
            )
            bench_added = True

    fig.update_layout(
        title="Cumulative Return vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _turnover_fig(report_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=report_df.index,
            y=report_df["turnover"],
            name="Turnover",
            opacity=0.6,
        )
    )
    if "cost" in report_df.columns:
        fig.add_trace(
            go.Scatter(
                x=report_df.index,
                y=report_df["cost"],
                name="Cost",
                mode="lines",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Turnover"),
            yaxis2=dict(title="Cost", overlaying="y", side="right"),
        )
    fig.update_layout(title="Turnover & Cost")
    return fig


def _indicator_fig(indicator_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Bar(
            x=indicator_df.index.tolist(),
            y=indicator_df["value"].tolist(),
            text=[f"{v:.3f}" for v in indicator_df["value"]],
            textposition="auto",
        )
    )
    fig.update_layout(title="Trading Indicators", yaxis_title="Value")
    return fig


def _score_label_fig(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> go.Figure:
    merged = label_df.join(pred_df, how="inner").dropna()
    label_col = merged.columns[0]
    score_col = merged.columns[-1]
    # downsample if dataset is too large
    if len(merged) > 100_000:
        merged = merged.sample(100_000, random_state=42)
    fig = go.Figure(
        go.Scattergl(
            x=merged[score_col],
            y=merged[label_col],
            mode="markers",
            opacity=0.4,
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title="Prediction vs Label Scatter",
        xaxis_title="Prediction Score",
        yaxis_title="Label",
    )
    return fig


def build_report_sections(run_dirs: List[Path]) -> Tuple[str, List[str]]:
    primary_dir = run_dirs[0]
    artifacts = primary_dir / "artifacts"
    port_dir = artifacts / "portfolio_analysis"

    report_df = _read_pickle(port_dir / "report_normal_1day.pkl")
    analysis_df = _read_pickle(port_dir / "port_analysis_1day.pkl")
    indicator_df = _read_pickle(port_dir / "indicator_analysis_1day.pkl")
    pred_df = _read_pickle(artifacts / "pred.pkl")
    label_df = _read_pickle(artifacts / "label.pkl")

    report_dfs = []
    for rd in run_dirs:
        r_artifacts = rd / "artifacts"
        r_port = r_artifacts / "portfolio_analysis"
        report_dfs.append((_label_from_dir(rd), _read_pickle(r_port / "report_normal_1day.pkl")))

    figures: List[Tuple[str, go.Figure]] = [
        ("Cumulative Return", _cum_return_fig(report_dfs)),
        ("Turnover & Cost", _turnover_fig(report_df)),
        ("Trading Indicators", _indicator_fig(indicator_df)),
        ("Prediction vs Label", _score_label_fig(pred_df, label_df)),
    ]

    sections = [_figure_to_section(title, fig) for title, fig in figures]

    # Add tables
    analysis_html = (
        analysis_df.reset_index()
        .rename(columns={"level_0": "metric_group", "level_1": "metric"})
        .to_html(index=False, float_format=lambda x: f"{x:.6f}")
    )
    indicator_html = indicator_df.reset_index().to_html(index=False, float_format=lambda x: f"{x:.6f}")
    sections.append(f"<section><h2>Risk Analysis</h2>{analysis_html}</section>")
    sections.append(f"<section><h2>Indicator Analysis</h2>{indicator_html}</section>")

    title_suffix = ", ".join([rd.name for rd in run_dirs])
    title = f"Qlib Run Report - {title_suffix}"
    return title, sections


def render_html(title: str, sections: Iterable[str]) -> str:
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 1.5rem;
      background: #f8f9fb;
    }}
    h1 {{
      margin-top: 0;
    }}
    section {{
      background: #fff;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    table, th, td {{
      border: 1px solid #ddd;
    }}
    th, td {{
      padding: 0.4rem 0.6rem;
      text-align: left;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {body}
</body>
</html>
"""


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a Qlib run (MLflow directory) in the browser.",
    )
    parser.add_argument(
        "run_path",
        nargs="+",
        help="One or two run directories containing artifacts (e.g., examples/mlruns/<exp>/<run_id>)",
    )
    parser.add_argument(
        "--output",
        help="Path to write the HTML file. Defaults to <run_path>/<english_prefix>_visual_report.html",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Generate the HTML file but do not open it in the browser.",
    )
    parser.set_defaults(open_browser=False)
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    run_dirs = [Path(p).expanduser().resolve() for p in args.run_path]
    if not 1 <= len(run_dirs) <= 2:
        raise ValueError("Please provide one or two run paths.")
    for rd in run_dirs:
        if not rd.exists():
            raise FileNotFoundError(f"Run directory not found: {rd}")

    title, sections = build_report_sections(run_dirs)
    html = render_html(title, sections)

    output_base = run_dirs[0]
    default_name = f"{_label_from_dir(output_base)}_visual_report.html"
    output_path = Path(args.output).expanduser().resolve() if args.output else output_base / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written to {output_path}")

    if args.open_browser:
        webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main(sys.argv[1:])

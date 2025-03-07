"""Plotting functions for multiple data subsets."""

import plotly.graph_objects as go
import pandas as pd
import gradio as gr

import inverse_cai.app.metrics


def _plot_multiple_values(
    votes_df: pd.DataFrame,
    plot_col_name: str,
    plot_col_values: list[str],
    shown_metric_names: list[str],
):
    """Add line traces for multiple values to the figure, with one subplot per principle."""
    gr.Info(f"Plotting with multiple values for '{plot_col_name}'")

    # overwrite
    # TODO: remove once ready
    plot_col_values = votes_df[plot_col_name].unique()
    shown_metric_names = ["perf", "acc", "relevance"]

    # sort plot col values
    plot_col_values = sorted(plot_col_values)

    proficiency_values = [
        "Basic",
        "Intermediate",
        "Advanced",
        "Fluent",
        "Native speaker",
    ]

    if plot_col_name == "english_proficiency":
        plot_col_values = proficiency_values

    HEIGHT_PER_PRINCIPLE = 200
    COLOR_LIST = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    full_metrics = inverse_cai.app.metrics.compute_metrics(votes_df)
    principles = full_metrics["principles"]
    # Create subplot layout
    fig = go.Figure()
    fig.update_layout(
        height=HEIGHT_PER_PRINCIPLE
        * len(principles),  # Adjust height based on number of principles
        showlegend=True,
    )

    # Get metrics for each value
    col_metrics = {}
    for plot_col_value in plot_col_values:
        col_df = votes_df[votes_df[plot_col_name] == plot_col_value]
        col_metrics[plot_col_value] = inverse_cai.app.metrics.compute_metrics(col_df)[
            "metrics"
        ]
        col_metrics[plot_col_value]["data_count"] = len(
            col_df["comparison_id"].unique()
        )

    # sort principles by full perf
    principles = sorted(
        principles,
        key=lambda x: full_metrics["metrics"]["perf"]["by_principle"][x],
        reverse=True,
    )

    # Create one subplot per principle
    for i, principle in enumerate(principles, 1):
        # Create subplot
        fig.add_trace(
            go.Scatter(
                x=[],  # Placeholder, will be filled below
                y=[],
                name="",  # Placeholder
                xaxis=f"x{i}",
                yaxis=f"y{i}",
            )
        )

        # Add traces for each metric
        for j, metric_name in enumerate(shown_metric_names):
            y_values = [
                col_metrics[val][metric_name]["by_principle"][principle]
                for val in plot_col_values
            ]

            fig.add_trace(
                go.Scatter(
                    x=plot_col_values,
                    y=y_values,
                    name=f"{metric_name}",
                    mode="lines+markers",
                    xaxis=f"x{i}",
                    yaxis=f"y{i}",
                    hovertext=[
                        f"{col_metrics[val]['data_count']} examples"
                        for val in plot_col_values
                    ],
                    marker=dict(color=COLOR_LIST[j - 1]),
                )
            )

        # Update layout for this subplot
        fig.update_layout(
            **{
                f"xaxis{i}": {"title": principle},
                f"yaxis{i}": {"title": str(shown_metric_names)},
            }
        )

        # add principle text to left of plot
        fig.add_annotation(
            text=principle,
            xref="paper",
            yref=f"y{i}",
            x=0.1,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )

    # Update subplot layout
    fig.update_layout(
        grid={"rows": len(principles), "columns": 1, "pattern": "independent"},
        title=f"Metrics by '{plot_col_name}' column",
        height=HEIGHT_PER_PRINCIPLE
        * len(principles),  # Adjust height based on number of principles
    )

    return fig

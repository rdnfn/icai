"""Plotting functions for the Inverse CAI app."""

import gradio as gr
import plotly.graph_objects as go
import pandas as pd
from loguru import logger

import inverse_cai.app.metrics
from inverse_cai.app.constants import (
    NONE_SELECTED_VALUE,
    PRINCIPLE_SHORT_LENGTH,
    FIG_PROPORTIONS_X,
    PRINCIPLE_END_X,
    METRICS_START_X,
    MENU_X,
    FONT_FAMILY,
    PAPER_BACKGROUND_COLOR,
    PLOT_BACKGROUND_COLOR,
    PLOTLY_MODEBAR_POSSIBLE_VALUES,
    get_fig_proportions_y,
)
from inverse_cai.app.plotting.single import _plot_examples, _plot_aggregated
from inverse_cai.app.plotting.multiple import _plot_multiple_values

# based on official plotly example
# https://plotly.com/python/horizontal-bar-charts/


def generate_plot(
    votes_df: pd.DataFrame,
    unfiltered_df: pd.DataFrame,
    show_examples: bool = False,
    shown_metric_names: list[str] | None = None,
    default_ordering_metric: str = "perf",
    sort_examples_by_agreement: bool = True,
    plot_col_name: str = NONE_SELECTED_VALUE,
    plot_col_values: list = None,
) -> go.Figure:

    if plot_col_name is not None and plot_col_name != NONE_SELECTED_VALUE:
        # plot per principle and per multiple column values
        gr.Warning(
            "Plots of the selected configuration (based on column values) are currently experimental, results may vary."
        )
        return _plot_multiple_values(
            votes_df=votes_df,
            plot_col_name=plot_col_name,
            plot_col_values=plot_col_values,
            shown_metric_names=shown_metric_names,
        )
    else:
        # plot only per principle (for entire dataset)
        return _generate_hbar_chart(
            votes_df=votes_df,
            unfiltered_df=unfiltered_df,
            show_examples=show_examples,
            shown_metric_names=shown_metric_names,
            default_ordering_metric=default_ordering_metric,
            sort_examples_by_agreement=sort_examples_by_agreement,
            plot_col_name=plot_col_name,
            plot_col_values=plot_col_values,
        )


def _generate_hbar_chart(
    votes_df: pd.DataFrame,
    unfiltered_df: pd.DataFrame,
    show_examples: bool = False,
    shown_metric_names: list[str] | None = None,
    default_ordering_metric: str = "perf",
    sort_examples_by_agreement: bool = True,
    plot_col_name: str = NONE_SELECTED_VALUE,
    plot_col_values: list = None,
) -> go.Figure:

    if shown_metric_names is None:
        shown_metric_names = [
            "perf",
            "perf_base",
            "perf_diff",
            "acc",
            "relevance",
        ]

    logger.debug("Computing metrics...")
    full_metrics: dict = inverse_cai.app.metrics.compute_metrics(unfiltered_df)
    metrics: dict = inverse_cai.app.metrics.compute_metrics(
        votes_df, baseline_metrics=full_metrics
    )
    principles = metrics["principles"]
    logger.debug("Metrics computed.")

    FIG_PROPORTIONS_Y = get_fig_proportions_y(len(principles))

    HEADING_HEIGHT_Y = FIG_PROPORTIONS_Y[1]
    MENU_Y = 1 - (1 - HEADING_HEIGHT_Y) / 3
    SPACE_ALL_NUM_COL = FIG_PROPORTIONS_X[0] - METRICS_START_X - 0.01
    SPACE_PER_NUM_COL = SPACE_ALL_NUM_COL / len(shown_metric_names)

    fig = go.Figure()

    if sort_examples_by_agreement:
        votes_df = votes_df.sort_values(by=["principle", "vote"], axis=0)

    # bar plots for each principle
    if plot_col_name == NONE_SELECTED_VALUE:

        # add plots for single set of values
        if show_examples:
            _plot_examples(fig, votes_df, principles)
        else:
            _plot_aggregated(fig, principles, metrics)

    else:
        logger.warning(
            f"Plotting multiple values for '{plot_col_name}' should not happen here."
        )

    # set up general layout configurations
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=FIG_PROPORTIONS_X,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=FIG_PROPORTIONS_Y,
        ),
        barmode="stack",
        paper_bgcolor=PAPER_BACKGROUND_COLOR,
        plot_bgcolor=PLOT_BACKGROUND_COLOR,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=20 * len(principles) + 50,
        font_family=FONT_FAMILY,
    )

    annotations = []
    headings_added = []
    # adding principle labels to y-axis
    for principle in principles:
        principle_short = (
            principle[:PRINCIPLE_SHORT_LENGTH] + "..."
            if len(principle) > PRINCIPLE_SHORT_LENGTH
            else principle
        )
        # principle
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=PRINCIPLE_END_X,
                y=principle,
                xanchor="right",
                text=principle_short,
                font=dict(size=12, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
                hovertext=principle,
            )
        )

        # add metric values in own columns
        for (
            start,
            value,
            label,
            hovertext,
        ) in inverse_cai.app.metrics.get_metric_cols_by_principle(
            principle,
            metrics,
            shown_metric_names,
            METRICS_START_X,
            FIG_PROPORTIONS_X[0] - METRICS_START_X - 0.01,
        ):
            annotations.append(
                dict(
                    xref="paper",
                    yref="y",
                    x=start,
                    y=principle,
                    xanchor="right",
                    text=f"{value:.2f}",
                    font=dict(size=14, color="rgb(67, 67, 67)"),
                    showarrow=False,
                    align="right",
                )
            )
            # value
            if label not in headings_added:
                headings_added.append(label)
                annotations.append(
                    dict(
                        xref="paper",
                        yref="paper",
                        x=start - SPACE_PER_NUM_COL / 2,
                        y=HEADING_HEIGHT_Y,
                        xanchor="center",
                        yanchor="bottom",
                        text=f"<i>{label}</i>",
                        font=dict(size=14, color="rgb(67, 67, 67)"),
                        showarrow=False,
                        align="left",
                        hovertext=hovertext,
                    )
                )

    # add principle and vote count headings
    for start, label, hovertext in [
        [PRINCIPLE_END_X / 2, "Principles", None],
        [
            FIG_PROPORTIONS_X[0] + (FIG_PROPORTIONS_X[1] - FIG_PROPORTIONS_X[0]) / 2,
            f"Preference reconstruction results ({metrics['num_pairs']} comparisons)",
            "One row per principle, one column per preference",
        ],
    ]:
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=start - SPACE_PER_NUM_COL / 2,
                y=HEADING_HEIGHT_Y,
                xanchor="center",
                yanchor="bottom",
                text=f"<i>{label}</i>",
                font=dict(size=14, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="left",
                hovertext=hovertext,
            )
        )

    # sort by agreement
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=metrics["metrics"][default_ordering_metric]["principle_order"],
    )

    # add sorting menu
    update_method = "relayout"  # "update"  # or "relayout"
    options = inverse_cai.app.metrics.get_ordering_options(
        metrics, shown_metric_names, initial=default_ordering_metric
    )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    {
                        "label": label,
                        "method": update_method,
                        "args": [
                            {
                                "yaxis.categoryorder": "array",
                                "yaxis.categoryarray": option,
                            }
                        ],
                    }
                    for label, option in options
                ],
                x=MENU_X,
                xanchor="left",
                y=MENU_Y,
                yanchor="middle",
            )
        ]
    )

    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=MENU_X,
            y=MENU_Y,
            xanchor="right",
            yanchor="middle",
            text="Sort by:",
            font=dict(size=14, color="rgb(67, 67, 67)"),
            showarrow=False,
            align="left",
        )
    )

    fig.update_layout(
        annotations=annotations,
    )

    # remove/hide modebar
    fig.update_layout(
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(0,0,0,0)",
            activecolor="rgba(0,0,0,0)",
            remove=PLOTLY_MODEBAR_POSSIBLE_VALUES,
        )
    )

    gr.Info("Plotting complete, uploading to interface.", duration=3)

    return fig

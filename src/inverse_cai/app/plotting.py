"""Plotting functions for the Inverse CAI app."""

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import inverse_cai.app.metrics
from inverse_cai.app.constants import (
    PRINCIPLE_SHORT_LENGTH,
    FIG_PROPORTIONS_X,
    FIG_PROPORTIONS_Y,
    SPACE_PER_NUM_COL,
    PRINCIPLE_END_X,
    METRICS_START_X,
    HEADING_HEIGHT_Y,
    MENU_X,
    MENU_Y,
    FONT_FAMILY,
    COLORS_DICT,
    PAPER_BACKGROUND_COLOR,
    PLOT_BACKGROUND_COLOR,
    PLOTLY_MODEBAR_POSSIBLE_VALUES,
)

# based on official plotly example
# https://plotly.com/python/horizontal-bar-charts/


def get_string_with_breaks(
    text: str, line_length: int = 50, max_lines: int = 20
) -> str:
    text = text.replace("\n", "<br>")
    lines = text.split(" ")

    full_text = ""
    current_line = ""
    lines_so_far = 0

    for word in lines:
        if len(current_line) + len(word) > line_length:
            full_text += current_line + "<br>"
            current_line = ""
            lines_so_far += 1
            if lines_so_far >= max_lines:
                current_line += "..."
                break

        current_line += word + " "

    full_text += current_line

    return full_text


def generate_hbar_chart(
    votes_df: pd.DataFrame,
    show_examples: bool = False,
    shown_metric_names: list = ["perf", "acc", "relevance"],
    default_ordering_metric="perf",
    sort_examples_by_agreement: bool = True,
) -> go.Figure:

    metrics: dict = inverse_cai.app.metrics.compute_metrics(votes_df)
    principles = metrics["principles"]

    fig = go.Figure()

    if sort_examples_by_agreement:
        votes_df = votes_df.sort_values(by=["principle", "vote"], axis=0)

    if show_examples:
        votes_df["selected_text"] = votes_df.apply(
            lambda x: x[x["preferred_text"]], axis=1
        )
        votes_df["rejected_text"] = votes_df.apply(
            lambda x: (x["text_a"] if x["preferred_text"] == "text_b" else x["text_b"]),
            axis=1,
        )
        votes_df["hovertext"] = votes_df.apply(
            lambda x: f"<b>{x['vote']}</b> <i>{get_string_with_breaks(x['principle'])}</i><br><br><b>Selected</b><br>{get_string_with_breaks(x['selected_text'])}<br><br><b>Rejected:</b><br>{get_string_with_breaks(x['rejected_text'])}",
            axis=1,
        )
        hover_args = {
            "hoverinfo": "text",
            "hovertext": votes_df["hovertext"],
        }
    else:
        hover_args = {"hoverinfo": "text", "hovertext": None}

    fig.add_trace(
        go.Bar(
            x=votes_df["weight"],
            y=votes_df["principle"],
            orientation="h",
            marker=dict(
                color=votes_df["vote"].apply(lambda x: COLORS_DICT[x]),
                # line=dict(color="black", width=2),
                cornerradius=10,
            ),
            **hover_args,
        )
    )

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

        # add agreement and accuracy values in own columns
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
        height=20 * len(principles),
        font_family=FONT_FAMILY,
    )

    # remove modebar
    fig.update_layout(
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(0,0,0,0)",
            activecolor="rgba(0,0,0,0)",
            remove=PLOTLY_MODEBAR_POSSIBLE_VALUES,
        )
    )

    gr.Info("Plot generated, will be displayed any moment now.")

    return fig

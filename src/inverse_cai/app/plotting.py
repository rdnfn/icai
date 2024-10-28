"""Plotting functions for the Inverse CAI app."""

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import inverse_cai.app.metrics
from inverse_cai.app.constants import (
    PRINCIPLE_SHORT_LENGTH,
    FIG_PROPORTIONS_X,
    PRINCIPLE_END_X,
    METRICS_START_X,
    MENU_X,
    FONT_FAMILY,
    COLORS_DICT,
    PAPER_BACKGROUND_COLOR,
    PLOT_BACKGROUND_COLOR,
    PLOTLY_MODEBAR_POSSIBLE_VALUES,
    get_fig_proportions_y,
)

# based on official plotly example
# https://plotly.com/python/horizontal-bar-charts/


HOVER_TEMPLATE = (
    "<b>Result:</b> {vote} <br><b>Principle:</b> {principle_str}<br>{text_columns_str}"
)


def get_string_with_breaks(
    text: str,
    line_length: int = 50,
    max_lines: int = 20,
    add_spaces_at_end: bool = False,
) -> str:
    text = text.strip()
    text = text.replace("\n", " <br> ")
    lines = text.split(" ")

    full_text = ""
    current_line = ""
    lines_so_far = 0

    for word in lines:
        if len(current_line) + len(word) > line_length or word == "<br>":

            if not add_spaces_at_end:
                spacing = ""
            else:
                spacing = " " * (line_length - len(current_line))

            full_text += current_line + spacing + "<br>"
            current_line = ""
            lines_so_far += 1
            if lines_so_far >= max_lines:
                current_line += "..." + spacing
                break

        if word != "<br>":
            current_line += word + " "

    full_text += current_line

    return full_text


def merge_into_columns(text_1, text_2):

    merged_text = ""

    text_1_lines = text_1.split("<br>")
    text_2_lines = text_2.split("<br>")
    max_lines = max(len(text_1_lines), len(text_2_lines))
    col_width = max([len(line) for line in text_1_lines + text_2_lines])

    for i in range(max_lines):
        line_1 = text_1_lines[i] if i < len(text_1_lines) else ""
        line_2 = text_2_lines[i] if i < len(text_2_lines) else ""

        line_1_without_html = (
            line_1.replace("<br>", "").replace("<b>", "").replace("</b>", "")
        )

        if len(line_1_without_html) < col_width:
            line_1 += " " * (col_width - len(line_1_without_html))

        merged_text += f"{line_1} | {line_2}<br>"

    return merged_text


def generate_hbar_chart(
    votes_df: pd.DataFrame,
    unfiltered_df: pd.DataFrame,
    show_examples: bool = False,
    shown_metric_names: list = [
        "perf",
        "perf_base",
        "perf_diff",
        "acc",
        "relevance",
    ],
    default_ordering_metric="perf",
    sort_examples_by_agreement: bool = True,
) -> go.Figure:

    gr.Info("Computing metrics...")
    full_metrics: dict = inverse_cai.app.metrics.compute_metrics(unfiltered_df)
    metrics: dict = inverse_cai.app.metrics.compute_metrics(
        votes_df, baseline_metrics=full_metrics
    )
    principles = metrics["principles"]
    gr.Info("Metrics computed.")

    FIG_PROPORTIONS_Y = get_fig_proportions_y(len(principles))

    HEADING_HEIGHT_Y = FIG_PROPORTIONS_Y[1]
    MENU_Y = 1 - (1 - HEADING_HEIGHT_Y) / 3
    SPACE_ALL_NUM_COL = FIG_PROPORTIONS_X[0] - METRICS_START_X - 0.01
    SPACE_PER_NUM_COL = SPACE_ALL_NUM_COL / len(shown_metric_names)

    fig = go.Figure()

    if sort_examples_by_agreement:
        votes_df = votes_df.sort_values(by=["principle", "vote"], axis=0)

    # bar plots for each principle
    if show_examples:
        votes_df["selected_text"] = votes_df.apply(
            lambda x: x[x["preferred_text"]], axis=1
        )
        votes_df["rejected_text"] = votes_df.apply(
            lambda x: (x["text_a"] if x["preferred_text"] == "text_b" else x["text_b"]),
            axis=1,
        )
        votes_df["hovertext"] = votes_df.apply(
            lambda x: HOVER_TEMPLATE.format(
                vote=x["vote"],
                principle_str=x["principle"],
                text_columns_str=merge_into_columns(
                    "<b>Selected</b><br>" + get_string_with_breaks(x["selected_text"]),
                    "<b>Rejected</b><br>" + get_string_with_breaks(x["rejected_text"]),
                ),
            ),
            axis=1,
        )
        hover_args = {
            "hoverinfo": "text",
            "hovertext": votes_df["hovertext"],
            # change font
            "hoverlabel": dict(font=dict(family="monospace")),
        }
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
    else:
        vote_dict = {
            "agreed": [],
            "disagreed": [],
            "not_applicable": [],
            "num_votes": [],
        }

        color_name_dict = {
            "agreed": "Agree",
            "disagreed": "Disagree",
            "not_applicable": "Not applicable",
        }

        for principle in principles:
            for vote_type in vote_dict.keys():
                vote_dict[vote_type].append(
                    metrics["metrics"][vote_type]["by_principle"][principle]
                )

        for vote_type, vote_list in vote_dict.items():
            if vote_type != "num_votes":
                color_name = color_name_dict[vote_type]
                fig.add_trace(
                    go.Bar(
                        x=vote_list,
                        y=principles,
                        orientation="h",
                        marker=dict(
                            color=COLORS_DICT[color_name],
                            # line=dict(color="black", width=2),
                            cornerradius=10,
                        ),
                        hoverinfo="text",
                        hovertext=[
                            f"{color_name}: {value} ({(value/sum_val)*100:.1f}%)"
                            for value, sum_val in zip(vote_list, vote_dict["num_votes"])
                        ],
                    )
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

    gr.Info("Plot generated, will be displayed any moment now.")

    return fig

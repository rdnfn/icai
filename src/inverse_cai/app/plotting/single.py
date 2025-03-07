"""Generate a plotly plot for single data subset."""

import plotly.graph_objects as go

from inverse_cai.app.constants import COLORS_DICT
from inverse_cai.app.plotting.utils import (
    get_string_with_breaks,
    merge_into_columns,
    HOVER_TEMPLATE,
)


def _plot_examples(fig, votes_df, principles):
    """Add example-level bar traces to the figure."""
    votes_df["selected_text"] = votes_df.apply(lambda x: x[x["preferred_text"]], axis=1)
    votes_df["rejected_text"] = votes_df.apply(
        lambda x: x["text_a"] if x["preferred_text"] == "text_b" else x["text_b"],
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
        "hoverlabel": dict(font=dict(family="monospace")),
    }

    fig.add_trace(
        go.Bar(
            x=votes_df["weight"],
            y=votes_df["principle"],
            orientation="h",
            marker=dict(
                color=votes_df["vote"].apply(lambda x: COLORS_DICT[x]),
                cornerradius=10,
            ),
            **hover_args,
        )
    )


def _plot_aggregated(fig, principles, metrics):
    """Add aggregated bar traces to the figure."""
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
        for vote_type in vote_dict:
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
                        cornerradius=10,
                    ),
                    hoverinfo="text",
                    hovertext=[
                        f"{color_name}: {value} ({(value/sum_val)*100:.1f}%)"
                        for value, sum_val in zip(vote_list, vote_dict["num_votes"])
                    ],
                )
            )

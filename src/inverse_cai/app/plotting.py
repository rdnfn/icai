import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# based on official plotly example
# https://plotly.com/python/horizontal-bar-charts/

FONT_FAMILY = '"Open Sans", verdana, arial, sans-serif'


### Layout and dimensions
PRINCIPLE_SHORT_LENGTH = 55
# this sets where the actual plot starts and ends (individual datapoints)
FIG_PROPORTIONS_X = [0.40, 0.99]
FIG_PROPORTIONS_Y = [0.01, 0.91]
SPACE_PER_NUM_COL = 0.04
PRINCIPLE_END_Y = FIG_PROPORTIONS_X[0] - 0.01 - 2 * SPACE_PER_NUM_COL
AGREEMENT_END_Y = FIG_PROPORTIONS_X[0] - 0.01 - SPACE_PER_NUM_COL
ACC_END_Y = FIG_PROPORTIONS_X[0] - 0.01
HEADING_HEIGHT_Y = FIG_PROPORTIONS_Y[1]
MENU_X = 0.3
MENU_Y = 0.97


### Colors
LIGHT_GREEN = "#d9ead3"
DARK_GREEN = "#38761d"
LIGHT_RED = "#f4cacb"
DARK_RED = "#a61d00"
LIGHTER_GREY = "#fafafa"
LIGHT_GREY = "#e4e4e7"
DARK_GREY = "rgba(192, 192, 192, 0.8)"
VERY_DARK_GREY = "rgba(48, 48, 48, 0.8)"
COLORS_DICT = {
    "Agree": LIGHT_GREEN,
    "Disagree": LIGHT_RED,
    "Not applicable": DARK_GREY,
}
DARK_COLORS_DICT = {
    "Agree": DARK_GREEN,
    "Disagree": DARK_RED,
    "Not applicable": VERY_DARK_GREY,
}
PAPER_BACKGROUND_COLOR = LIGHT_GREY
PLOT_BACKGROUND_COLOR = LIGHT_GREY


def generate_hbar_chart_original(votes_df: pd.DataFrame) -> go.Figure:

    fig = px.bar(
        votes_df,
        x="weight",
        y="principle",
        color="vote",
        orientation="h",
        hover_data=["text_a", "text_b", "preferred_text"],
    )

    return fig


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


def generate_hbar_chart(votes_df: pd.DataFrame) -> go.Figure:

    principles = votes_df["principle"].unique()

    agreement_by_principle = {
        principle: votes_df[votes_df["principle"] == principle]["vote"]
        .value_counts()
        .get("Agree", 0)
        / len(votes_df[votes_df["principle"] == principle])
        for principle in principles
    }

    principles_by_agreement = sorted(
        principles,
        key=lambda x: agreement_by_principle[x],
    )

    def get_acc(principle: str) -> int:
        value_counts = votes_df[votes_df["principle"] == principle][
            "vote"
        ].value_counts()
        try:
            acc = value_counts.get("Agree", 0) / (
                value_counts.get("Disagree", 0) + value_counts.get("Agree", 0)
            )
            # main sort by accuracy, then by agreement, then by disagreement (reversed)
            return acc, value_counts.get("Agree", 0), -value_counts.get("Disagree", 0)
        except ZeroDivisionError:
            return 0, value_counts.get("Agree", 0), -value_counts.get("Disagree", 0)

    acc_by_principle = {principle: get_acc(principle) for principle in principles}

    principles_by_acc = sorted(
        principles,
        key=lambda x: acc_by_principle[x],
    )

    fig = go.Figure()

    for _, datapoint in votes_df.iterrows():
        pref_text = datapoint["preferred_text"]
        selected_text = datapoint[pref_text]
        rejected_text = (
            datapoint["text_a"] if pref_text == "text_b" else datapoint["text_b"]
        )
        fig.add_trace(
            go.Bar(
                x=[datapoint["weight"]],
                y=[datapoint["principle"]],
                orientation="h",
                marker=dict(
                    color=COLORS_DICT[datapoint["vote"]],
                    line=dict(color="white", width=2),
                ),
                hoverinfo="text",
                hovertext=f"<b>{datapoint['vote']}</b> <i>{get_string_with_breaks(datapoint['principle'])}</i><br><br><b>Selected</b><br>{get_string_with_breaks(selected_text)}<br><br><b>Rejected:</b><br>{get_string_with_breaks(rejected_text)}",
                hoverlabel=dict(bordercolor=DARK_COLORS_DICT[datapoint["vote"]]),
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
            if len(principle) > 50
            else principle
        )
        # principle
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=PRINCIPLE_END_Y,
                y=principle,
                xanchor="right",
                text=principle_short,
                font=dict(size=12, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
                hovertext=principle,
            )
        )
        for start, value, label, hovertext in [
            [
                AGREEMENT_END_Y,
                agreement_by_principle[principle],
                "Agr.",
                "Agreement: proportion of all votes that agree with original preferences",
            ],
            [
                ACC_END_Y,
                acc_by_principle[principle][0],
                "Acc.",
                "Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')<br>that agree with original preferences",
            ],
        ]:
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
        [PRINCIPLE_END_Y / 2, "Principles", None],
        [
            FIG_PROPORTIONS_X[0] + (FIG_PROPORTIONS_X[1] - FIG_PROPORTIONS_X[0]) / 2,
            "Preference reconstruction results",
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
    fig.update_yaxes(categoryorder="array", categoryarray=principles_by_agreement)

    update_method = "relayout"  # "update"  # or "relayout"
    options = [
        ["Agr. (desc.)", principles_by_agreement],
        # ["Num agreed (asc.)", list(reversed(principles_by_agreement))],
        ["Acc. (desc.)", principles_by_acc],
        # ["Accuracy (asc.)", list(reversed(principles_by_acc))],
        # list(reversed(principles_by_agreement)),
        # principles_by_acc,
        # list(reversed(principles_by_acc)),
    ]

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
            remove=[
                "autoScale2d",
                "autoscale",
                "editInChartStudio",
                "editinchartstudio",
                "hoverCompareCartesian",
                "hovercompare",
                "lasso",
                "lasso2d",
                "orbitRotation",
                "orbitrotation",
                "pan",
                "pan2d",
                "pan3d",
                "reset",
                "resetCameraDefault3d",
                "resetCameraLastSave3d",
                "resetGeo",
                "resetSankeyGroup",
                "resetScale2d",
                "resetViewMap",
                "resetViewMapbox",
                "resetViews",
                "resetcameradefault",
                "resetcameralastsave",
                "resetsankeygroup",
                "resetscale",
                "resetview",
                "resetviews",
                "select",
                "select2d",
                "sendDataToCloud",
                "senddatatocloud",
                "tableRotation",
                "tablerotation",
                "toImage",
                "toggleHover",
                "toggleSpikelines",
                "togglehover",
                "togglespikelines",
                "toimage",
                "zoom",
                "zoom2d",
                "zoom3d",
                "zoomIn2d",
                "zoomInGeo",
                "zoomInMap",
                "zoomInMapbox",
                "zoomOut2d",
                "zoomOutGeo",
                "zoomOutMap",
                "zoomOutMapbox",
                "zoomin",
                "zoomout",
            ],
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig

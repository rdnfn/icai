import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# based on official plotly example
# https://plotly.com/python/horizontal-bar-charts/

PRINCIPLE_SHORT_LENGTH = 55
FIG_PROPORTIONS = [0.5, 1]
PRINCIPLE_END_Y = FIG_PROPORTIONS[0] - 0.01


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

    top_labels = [
        "Agree",
        "Disagree",
        "Invalid",
    ]

    colors_dict = {
        # pastel green
        "Agree": "rgba(144, 238, 144, 0.5)",
        # pastel red
        "Disagree": "rgba(255, 99, 71, 0.5)",
        # pastel grey
        "Not applicable": "rgba(192, 192, 192, 0.8)",
    }

    principles = votes_df["principle"].unique()

    principles_by_agreement = sorted(
        principles,
        key=lambda x: votes_df[votes_df["principle"] == x]["vote"]
        .value_counts()
        .get("Agree", 0),
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

    principles_by_acc = sorted(
        principles,
        key=get_acc,
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
                    color=colors_dict[datapoint["vote"]],
                    line=dict(color="rgb(248, 248, 249)", width=1),
                ),
                hoverinfo="text",
                hovertext=f"<b>{datapoint['vote']}</b> <i>{get_string_with_breaks(datapoint['principle'])}</i><br><br><b>Selected</b><br>{get_string_with_breaks(selected_text)}<br><br><b>Rejected:</b><br>{get_string_with_breaks(rejected_text)}",
            )
        )

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=FIG_PROPORTIONS,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode="stack",
        paper_bgcolor="rgb(248, 248, 255)",
        plot_bgcolor="rgb(248, 248, 255)",
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
    )

    annotations = []
    # adding principle labels to y-axis
    for principle in principles:
        principle_short = (
            principle[:PRINCIPLE_SHORT_LENGTH] + "..."
            if len(principle) > 50
            else principle
        )
        principle_rest = (
            "... " + principle[PRINCIPLE_SHORT_LENGTH:] if len(principle) > 50 else None
        )
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=PRINCIPLE_END_Y,
                y=principle,
                xanchor="right",
                text=principle_short,
                font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
                hovertext=principle,
            )
        )

    fig.update_layout(
        annotations=annotations,
        height=20 * len(principles),
    )

    # sort by agreement
    fig.update_yaxes(categoryorder="array", categoryarray=principles_by_agreement)

    update_method = "relayout"  # "update"  # or "relayout"
    options = [
        ["Num agreed (desc.)", principles_by_agreement],
        # ["Num agreed (asc.)", list(reversed(principles_by_agreement))],
        ["Accuracy (desc.)", principles_by_acc],
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
            )
        ]
    )

    return fig

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=0.14,
                y=yd,
                xanchor="right",
                text=str(yd),
                font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
            )
        )
        # labeling the first percentage of each bar (x_axis)
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=xd[0] / 2,
                y=yd,
                text=str(xd[0]) + "%",
                font=dict(family="Arial", size=14, color="rgb(248, 248, 255)"),
                showarrow=False,
            )
        )
        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(
                dict(
                    xref="x",
                    yref="paper",
                    x=xd[0] / 2,
                    y=1.1,
                    text=top_labels[0],
                    font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                    showarrow=False,
                )
            )
        space = xd[0]
        for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(
                dict(
                    xref="x",
                    yref="y",
                    x=space + (xd[i] / 2),
                    y=yd,
                    text=str(xd[i]) + "%",
                    font=dict(family="Arial", size=14, color="rgb(248, 248, 255)"),
                    showarrow=False,
                )
            )
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(
                    dict(
                        xref="x",
                        yref="paper",
                        x=space + (xd[i] / 2),
                        y=1.1,
                        text=top_labels[i],
                        font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                        showarrow=False,
                    )
                )
            space += xd[i]

    fig.update_layout(annotations=annotations)
import textwrap
import matplotlib.pyplot as plt
import numpy as np


def plot_approval_bars_from_results(results, path=None):

    summaries = results["summaries"]
    votes = results["parsed_votes"]

    categories = list(summaries.values())
    votes = list(votes.values())

    plot_approval_bars(categories, votes, path=path)


def plot_approval_bars(categories, votes, path=None):
    """
    Plots a horizontal bar chart with no axis labels and no grid lines, with rounded bars,
    representing votes for, against, and abstentions for multiple categories.

    Parameters:
    - categories: List of strings (multi-line labels for each category).
    - votes: List of tuples (for, against, abstention) representing votes for each category.
    """
    # Constants for bar settings
    bar_height = 0.6  # Make bars vertically narrower
    colors = [
        "#90ee90",
        "#ffcccb",
        "#add8e6",
        "#d3d3d3",
    ]  # Light green, light red, light blue, light grey

    # Create figure and axis
    _, ax = plt.subplots(
        figsize=(10, len(categories) * 1.5)
    )  # Adjust figure size for narrower bars

    # Remove all spines, ticks, labels, and grid lines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.xaxis.grid(False)  # Remove horizontal grid lines
    ax.yaxis.grid(False)  # Ensure vertical grid lines are removed

    # Set y-axis with multi-line labels vertically centered for each bar
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, va="center")

    # For each category, plot each type of vote (for, against, abstention)
    for i, vote in enumerate(votes):
        vote_for, vote_against, vote_abstention, vote_invalid = (
            vote["for"],
            vote["against"],
            vote["abstain"],
            vote["invalid"],
        )
        # Calculate total to display percentages
        total_votes = vote_for + vote_against + vote_abstention + vote_invalid
        widths = [
            vote_for / total_votes * 100,
            vote_against / total_votes * 100,
            vote_abstention / total_votes * 100,
            vote_invalid / total_votes * 100,
        ]
        starts = [
            0,
            widths[0],
            widths[0] + widths[1],
            widths[0] + widths[1] + widths[2],
        ]

        for j, width in enumerate(widths):
            ax.barh(
                -i,
                width,
                left=starts[j],
                height=bar_height,
                color=colors[j],
                edgecolor="none",
            )

            # Add text inside the bar
            if width > 0:  # Only show text if the bar is wide enough
                ax.text(
                    starts[j] + width / 2,
                    -i,
                    f"{width:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

    # Manually add the y-axis labels to the left of the bars to ensure they are vertically centered
    for i, label in enumerate(categories):
        ax.text(
            -5,
            -i,
            textwrap.fill(f"Principle {i}: " + label, 50),
            ha="right",
            va="center",
            fontsize=10,
        )

    # Adjust layout
    ax.set_xlim(0, 100)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

import pandas as pd
import plotly.graph_objects as go

from inverse_cai.app.plotting.multiple import _plot_multiple_values


def test_plot():
    # Create sample data
    data = {
        "model": ["model_A", "model_A", "model_A", "model_B", "model_B", "model_B"],
        "principle": ["honesty", "kindness", "fairness"] * 2,
        "text_a": ["text_a_1", "text_a_2", "text_a_3"] * 2,
        "text_b": ["text_b_1", "text_b_2", "text_b_3"] * 2,
        "preferred_text": ["text_a", "text_b", "text_a", "text_b", "text_a", "text_b"],
        "comparison_id": [1, 1, 1, 2, 2, 2],
        "vote": ["Agree", "Agree", "Agree", "Agree", "Disagree", "Disagree"],
    }
    votes_df = pd.DataFrame(data)

    # Setup test parameters
    plot_col_name = "model"
    plot_col_values = ["model_A", "model_B"]
    shown_metric_names = ["perf"]

    # Create figure and plot
    fig = go.Figure()
    fig = _plot_multiple_values(
        votes_df=votes_df,
        plot_col_name=plot_col_name,
        plot_col_values=plot_col_values,
        shown_metric_names=shown_metric_names,
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    test_plot()

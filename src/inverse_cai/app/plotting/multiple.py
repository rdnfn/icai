"""Plotting functions for multiple data subsets."""

import plotly.graph_objects as go
import pandas as pd

from inverse_cai.app.plotting.utils import HOVER_TEMPLATE


def _plot_multiple_values(
    fig: go.Figure,
    votes_df: pd.DataFrame,
    principles: list[str],
    plot_col_name: str,
    plot_col_values: list[str],
    shown_metric_names: list[str],
):
    """Add bar traces for multiple values to the figure."""
    pass

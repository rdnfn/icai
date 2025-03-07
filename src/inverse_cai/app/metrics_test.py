"""
Tests for the metrics module.
"""

import pandas as pd
import numpy as np
from inverse_cai.app.metrics import (
    get_agreement,
    get_acc,
    get_relevance,
    get_perf,
    get_num_votes,
    get_agreed,
    get_disagreed,
    get_not_applicable,
    compute_metrics,
)


def test_get_agreement():
    """Test agreement calculation for different vote distributions."""
    # Test with all vote types
    value_counts = pd.Series({"Agree": 3, "Disagree": 2, "Not applicable": 1})
    assert get_agreement(value_counts) == 0.5  # 3 / (3 + 2 + 1)

    # Test with no votes
    value_counts = pd.Series({"Not applicable": 5})
    assert get_agreement(value_counts) == 0

    # Test with only agrees
    value_counts = pd.Series({"Agree": 5})
    assert get_agreement(value_counts) == 1.0


def test_get_acc():
    """Test accuracy calculation for different vote distributions."""
    # Test with agree and disagree votes
    value_counts = pd.Series({"Agree": 3, "Disagree": 1})
    assert get_acc(value_counts) == 0.75  # 3 / (3 + 1)

    # Test with only agrees
    value_counts = pd.Series({"Agree": 5})
    assert get_acc(value_counts) == 1.0

    # Test with only disagrees
    value_counts = pd.Series({"Disagree": 5})
    assert get_acc(value_counts) == 0.0

    # Test with no relevant votes
    value_counts = pd.Series({"Not applicable": 5})
    assert get_acc(value_counts) == 0.0


def test_get_relevance():
    """Test relevance calculation for different vote distributions."""
    # Test with all vote types
    value_counts = pd.Series({"Agree": 3, "Disagree": 2, "Not applicable": 1})
    assert np.isclose(
        get_relevance(value_counts), 0.833, rtol=1e-3
    )  # (3 + 2) / (3 + 2 + 1), rounded to 3 decimals

    # Test with no not applicable votes
    value_counts = pd.Series({"Agree": 3, "Disagree": 2})
    assert get_relevance(value_counts) == 1.0

    # Test with only not applicable votes
    value_counts = pd.Series({"Not applicable": 5})
    assert get_relevance(value_counts) == 0.0


def test_get_perf():
    """Test performance calculation for different vote distributions."""
    # Test with perfect performance
    value_counts = pd.Series({"Agree": 4, "Disagree": 0, "Not applicable": 1})
    expected = (1.0 - 0.5) * (4 / 5) * 2  # (acc - 0.5) * relevance * 2
    assert get_perf(value_counts) == expected

    # Test with worst performance
    value_counts = pd.Series({"Agree": 0, "Disagree": 4, "Not applicable": 1})
    expected = (0.0 - 0.5) * (4 / 5) * 2
    assert get_perf(value_counts) == expected

    # Test with neutral performance
    value_counts = pd.Series({"Agree": 2, "Disagree": 2, "Not applicable": 1})
    assert get_perf(value_counts) == 0.0


def test_vote_count_functions():
    """Test basic vote counting functions."""
    value_counts = pd.Series({"Agree": 3, "Disagree": 2, "Not applicable": 1})

    assert get_num_votes(value_counts) == 6
    assert get_agreed(value_counts) == 3
    assert get_disagreed(value_counts) == 2
    assert get_not_applicable(value_counts) == 1


def test_compute_metrics():
    """Test metric computation with sample vote data."""
    votes_df = pd.DataFrame(
        {
            "comparison_id": [1, 1, 2, 2],
            "principle": ["p1", "p2", "p1", "p2"],
            "vote": ["Agree", "Disagree", "Not applicable", "Agree"],
        }
    )

    metrics = compute_metrics(votes_df)

    # Check structure
    assert "principles" in metrics
    assert "num_pairs" in metrics
    assert "metrics" in metrics

    # Check principles
    assert set(metrics["principles"]) == {"p1", "p2"}
    assert metrics["num_pairs"] == 2

    # Check metrics for p1
    p1_metrics = {
        metric: metrics["metrics"][metric]["by_principle"]["p1"]
        for metric in ["agreement", "acc", "relevance", "perf"]
    }
    assert p1_metrics["agreement"] == 0.5  # 1 agree out of 2 total
    assert p1_metrics["acc"] == 1.0  # 1 agree out of 1 relevant vote
    assert p1_metrics["relevance"] == 0.5  # 1 relevant out of 2 total

    # Test with baseline metrics
    baseline_metrics = compute_metrics(votes_df)
    metrics_with_baseline = compute_metrics(votes_df, baseline_metrics=baseline_metrics)

    # Check that diff and base metrics exist
    assert "perf_diff" in metrics_with_baseline["metrics"]
    assert "perf_base" in metrics_with_baseline["metrics"]


def test_compute_metrics_empty_data():
    """Test metric computation with empty input data."""
    empty_df = pd.DataFrame(columns=["comparison_id", "principle", "vote"])
    metrics = compute_metrics(empty_df)

    assert len(metrics["principles"]) == 0
    assert metrics["num_pairs"] == 0

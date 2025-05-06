"""Tests for the filtering module."""

import pytest
from inverse_cai.algorithm.filter import filter_according_to_votes


@pytest.fixture
def sample_votes():
    """Sample votes fixture for testing."""
    return {
        "principle1": {
            "for": 8,
            "against": 2,
            "abstain": 0,
            "invalid": 0,
            "both": 0,
            "neither": 0,
        },
        "principle2": {
            "for": 6,
            "against": 5,
            "abstain": 0,
            "invalid": 0,
            "both": 0,
            "neither": 0,
        },
        "principle3": {
            "for": 3,
            "against": 7,
            "abstain": 0,
            "invalid": 0,
            "both": 0,
            "neither": 0,
        },
    }


def test_filter_by_majority_true(sample_votes):
    """Test filtering principles by majority true votes."""
    result = filter_according_to_votes(
        sample_votes,
        require_majority_true=True,
        require_majority_relevant=False,
        require_majority_valid=False,
        require_minimum_relevance=None,
        order_by="for",
        max_principles=None,
    )

    assert len(result) == 2
    assert "principle1" in result
    assert "principle2" in result
    assert "principle3" not in result


def test_filter_by_relevance(sample_votes):
    """Test filtering principles by relevance threshold."""
    result = filter_according_to_votes(
        sample_votes,
        require_majority_true=False,
        require_majority_relevant=False,
        require_majority_valid=False,
        require_minimum_relevance=0.7,  # 7/10 = 0.7
        order_by="for",
        max_principles=None,
    )

    assert len(result) == 1
    assert "principle1" in result


def test_order_by_for_minus_against(sample_votes):
    """Test ordering principles by for minus against votes."""
    result = filter_according_to_votes(
        sample_votes,
        require_majority_true=False,
        require_majority_relevant=False,
        require_majority_valid=False,
        require_minimum_relevance=None,
        order_by="for_minus_against",
        max_principles=None,
    )

    assert result == ["principle1", "principle2", "principle3"]


def test_max_principles_limit(sample_votes):
    """Test limiting the number of principles."""
    result = filter_according_to_votes(
        sample_votes,
        require_majority_true=False,
        require_majority_relevant=False,
        require_majority_valid=False,
        require_minimum_relevance=None,
        order_by="for",
        max_principles=2,
    )

    assert len(result) == 2
    assert "principle1" in result
    assert "principle2" in result


def test_multiple_filters(sample_votes):
    """Test applying multiple filters together."""
    result = filter_according_to_votes(
        sample_votes,
        require_majority_true=True,
        require_majority_relevant=True,
        require_majority_valid=False,
        require_minimum_relevance=0.5,
        order_by="for",
        max_principles=2,
    )

    assert len(result) == 2
    assert "principle1" in result
    assert "principle2" in result
    assert "principle3" not in result


def test_empty_votes():
    """Test handling of empty votes dictionary."""
    result = filter_according_to_votes(
        {},
        require_majority_true=True,
        require_majority_relevant=True,
        require_majority_valid=True,
        require_minimum_relevance=0.5,
        order_by="for",
        max_principles=None,
    )

    assert result == []


def test_invalid_order_by(sample_votes):
    """Test handling of invalid order_by parameter."""
    with pytest.raises(ValueError):
        filter_according_to_votes(
            sample_votes,
            require_majority_true=False,
            require_majority_relevant=False,
            require_majority_valid=False,
            require_minimum_relevance=None,
            order_by="invalid_ordering",
            max_principles=None,
        )


def test_all_principles_filtered(sample_votes):
    """Test case where all principles are filtered out."""
    result = filter_according_to_votes(
        sample_votes,
        require_majority_true=True,
        require_majority_relevant=True,
        require_majority_valid=True,
        require_minimum_relevance=0.9,  # No principle meets this threshold
        order_by="for",
        max_principles=None,
    )

    assert result == []

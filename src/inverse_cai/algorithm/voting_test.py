"""
Tests for the voting module.
"""

from unittest.mock import patch, MagicMock

from inverse_cai.algorithm.voting import (
    get_preference_vote_for_single_text,
    clean_vote_json,
    parse_individual_pref_vote,
)
from inverse_cai.experiment.core import ExpConfig


@patch("inverse_cai.algorithm.voting.inverse_cai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")
def test_get_preference_vote_for_single_text_flipped(
    mock_random_choice, mock_get_model
):
    """Test preference voting when the order of samples is flipped."""
    # outputs are always flipped
    mock_random_choice.return_value = True
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{0: "A", 1: "B"}'
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
        model_name="openai/gpt-4o-mini-2024-07-18",
    )

    assert result == {1: False, 2: True}


@patch("inverse_cai.algorithm.voting.inverse_cai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")
def test_get_preference_vote_for_single_text_not_flipped(
    mock_random_choice, mock_get_model
):
    """Test preference voting when the order of samples is not flipped."""
    mock_random_choice.return_value = False
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{0: "A", 1: "B"}'
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
        model_name="openai/gpt-4o-mini-2024-07-18",
    )

    assert result == {1: True, 2: False}


@patch("inverse_cai.algorithm.voting.inverse_cai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")
def test_get_preference_vote_for_single_text_invalid_vote(
    mock_random_choice, mock_get_model
):
    """Test preference voting when an invalid vote is returned by model."""
    mock_random_choice.return_value = False
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"0": "C", "1": "None"}'
    mock_get_model.return_value = mock_model

    return_val = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
        model_name="openai/gpt-4o-mini-2024-07-18",
    )
    assert return_val == {1: "invalid", 2: None}


@patch("inverse_cai.algorithm.voting.inverse_cai.models.get_model")
def test_get_preference_vote_for_single_text_invalid_json(mock_get_model):
    """Test preference voting when invalid JSON is returned."""
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = "invalid_json"
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
        model_name="openai/gpt-4o-mini-2024-07-18",
    )

    assert all(
        value is "invalid" for value in result.values()
    ), "Expected all votes to be None due to invalid JSON"


@patch("inverse_cai.algorithm.voting.inverse_cai.models.get_model")
def test_get_preference_vote_for_single_text_all_keys_present(mock_get_model):
    """Test that all summary keys are present in the preference voting result."""
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"0": "A", "1": "B", "2": "A"}'
    mock_get_model.return_value = mock_model

    summaries = {1: "suma", 2: "sumb", 3: "sumc"}
    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        summaries,
        config=ExpConfig(),
        model_name="openai/gpt-4o-mini-2024-07-18",
    )

    assert set(result.keys()) == set(
        summaries.keys()
    ), "Not all keys from summaries are present in the result"


def test_clean_vote_json():
    """Test cleaning and formatting of vote JSON strings."""
    vote_json = '{"1": "true", "2": "false", "3": "null", "4": "A", "5": "B"}'
    summaries = {1: "sum1", 2: "sum2", 3: "sum3", 4: "sum4", 5: "sum5"}

    cleaned_json = clean_vote_json(vote_json, len(summaries))
    expected_json = '{1:True,2:False,3:None,4:"A",5:"B"}'

    assert (
        cleaned_json == expected_json
    ), "The clean_vote_json function did not clean the JSON as expected"


@patch("inverse_cai.algorithm.voting.inverse_cai.models.get_model")
def test_get_preference_vote_for_single_text_unexpected_values(mock_get_model):
    """Test to ensure unexpected vote values are counted as invalid in preference voting."""
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"0": "Z", "1": "Y"}'  # Unexpected values
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
        model_name="openai/gpt-4o-mini-2024-07-18",
    )

    assert all(
        value is "invalid" for value in result.values()
    ), "Expected all votes to be counted as invalid due to unexpected values"


def test_parse_individual_pref_vote():
    """Test parsing of individual preference votes from JSON responses."""

    # Test valid JSON votes
    assert parse_individual_pref_vote('{"1": "A", "2": "B"}', 2) == {
        1: "A",
        2: "B",
    }, "Should correctly parse A/B votes"

    # Test invalid JSON format
    assert parse_individual_pref_vote("invalid_json", 2) == {
        0: "invalid",
        1: "invalid",
    }, "Should mark invalid JSON as invalid votes"

    # Test missing keys
    assert parse_individual_pref_vote('{"1": "A"}', 2) == {
        1: "A",
    }, "Should parse partial votes"

    # Test invalid vote values
    assert parse_individual_pref_vote('{"1": "C", "2": "D"}', 2) == {
        1: "invalid",
        2: "invalid",
    }, "Should mark invalid vote values as 'invalid'"

    # Test with None values
    assert parse_individual_pref_vote('{"1": "foo", "2": "A"}', 2) == {
        1: "invalid",
        2: "A",
    }, "Should handle null values"

    # Test with different summary lengths
    assert parse_individual_pref_vote('{"1": "A", "2": "B", "3": "A"}', 3) == {
        1: "A",
        2: "B",
        3: "A",
    }, "Should handle different numbers of summaries"

import random
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from loguru import logger

from inverse_cai.algorithm.voting import (
    get_preference_vote_for_single_text,
    clean_vote_json,
)
from inverse_cai.experiment.core import ExpConfig


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")
def test_get_preference_vote_for_single_text_flipped(
    mock_random_choice, mock_get_model
):
    # outputs are always flipped
    mock_random_choice.return_value = True
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"1": "A", "2": "B"}'
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
    )

    assert result == {1: False, 2: True}


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")
def test_get_preference_vote_for_single_text_not_flipped(
    mock_random_choice, mock_get_model
):
    mock_random_choice.return_value = False
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"1": "A", "2": "B"}'
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
    )

    assert result == {1: True, 2: False}


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")
def test_get_preference_vote_for_single_text_invalid_vote(
    mock_random_choice, mock_get_model, caplog
):
    mock_random_choice.return_value = False
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"1": "C", "2": "None"}'
    mock_get_model.return_value = mock_model

    return_val = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
    )
    assert return_val == {1: "invalid", 2: None}


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_preference_vote_for_single_text_invalid_json(mock_get_model, caplog):
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = "invalid_json"
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
    )

    assert all(
        value is "invalid" for value in result.values()
    ), "Expected all votes to be None due to invalid JSON"


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_preference_vote_for_single_text_all_keys_present(mock_get_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"1": "A", "2": "B", "3": "A"}'
    mock_get_model.return_value = mock_model

    summaries = {1: "suma", 2: "sumb", 3: "sumc"}
    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        summaries,
        config=ExpConfig(),
    )

    assert set(result.keys()) == set(
        summaries.keys()
    ), "Not all keys from summaries are present in the result"


def test_clean_vote_json():
    vote_json = '{"1": "true", "2": "false", "3": "null", "4": "A", "5": "B"}'
    summaries = {1: "sum1", 2: "sum2", 3: "sum3", 4: "sum4", 5: "sum5"}

    cleaned_json = clean_vote_json(vote_json, summaries)
    expected_json = '{1:True,2:False,3:None,4:"A",5:"B"}'

    assert (
        cleaned_json == expected_json
    ), "The clean_vote_json function did not clean the JSON as expected"


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_consistency_vote_for_single_text_valid(mock_get_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"0": "True", "1": "invalid"}'
    mock_get_model.return_value = mock_model

    result = get_consistency_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {0: "principle0", 1: "principle1"},
        config=ExpConfig(),
    )

    assert result[0] in [
        True,
        False,
        None,
    ], "Expected vote for principle 0 to be one of [True, False, None]"
    assert result[1] == "invalid", "Expected vote for principle 1 to be 'invalid'"


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_consistency_vote_for_single_text_invalid_json(mock_get_model, caplog):
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = "invalid_json_format"
    mock_get_model.return_value = mock_model

    summaries = {0: "principle0", 1: "principle1"}
    result = get_consistency_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        summaries,
        config=ExpConfig(),
    )

    for key in summaries.keys():
        assert (
            result[key] == "invalid"
        ), f"Expected vote for principle {key} to be marked as 'invalid' due to invalid JSON"


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_consistency_vote_for_single_text_all_keys_present(mock_get_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = (
        '{"0": "valid", "1": "invalid", "2": "valid"}'
    )
    mock_get_model.return_value = mock_model

    summaries = {0: "principle0", 1: "principle1", 2: "principle2"}
    result = get_consistency_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        summaries,
        config=ExpConfig(),
    )

    assert set(result.keys()) == set(
        summaries.keys()
    ), "Not all keys from summaries are present in the result"


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_preference_vote_for_single_text_unexpected_values(mock_get_model, caplog):
    """Test to ensure unexpected vote values are counted as invalid in preference voting."""
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = '{"1": "Z", "2": "Y"}'  # Unexpected values
    mock_get_model.return_value = mock_model

    result = get_preference_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {1: "suma", 2: "sumb"},
        config=ExpConfig(),
    )

    assert all(
        value is "invalid" for value in result.values()
    ), "Expected all votes to be counted as invalid due to unexpected values"


@patch("inverse_cai.algorithm.voting.icai.models.get_model")
def test_get_consistency_vote_for_single_text_unexpected_values(mock_get_model, caplog):
    """Test to ensure unexpected vote values are counted as invalid in consistency voting."""
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = (
        '{"0": "maybe", "1": "perhaps"}'  # Unexpected values
    )
    mock_get_model.return_value = mock_model

    result = get_consistency_vote_for_single_text(
        "preferred_sample",
        "rejected_sample",
        {0: "principle0", 1: "principle1"},
        config=ExpConfig(),
    )

    assert all(
        value == "invalid" for value in result.values()
    ), "Expected all votes to be marked as 'invalid' due to unexpected values"

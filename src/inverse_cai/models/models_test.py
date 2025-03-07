import pytest
import numpy as np
from unittest.mock import Mock, patch
from inverse_cai.models import get_token_probs, NULL_LOGPROB_VALUE


@patch("inverse_cai.models.logger")
def test_get_token_probs(mock_logger):
    # Mock the model's generate method
    mock_model = Mock()
    mock_model.generate.return_value = Mock(
        generations=[
            [
                Mock(
                    generation_info={
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "token1", "logprob": -0.1},
                                        {"token": "token2", "logprob": -0.2},
                                    ]
                                }
                            ]
                        }
                    }
                )
            ]
        ]
    )

    tokens = ["token1", "token2"]
    messages = ["message1", "message2"]

    # Call the function
    token_probs, errors = get_token_probs(tokens, mock_model, messages)

    # Assert the results
    sum_probs = np.exp(-0.1) + np.exp(-0.2)
    assert token_probs == {
        "token1": np.exp(-0.1) / sum_probs,
        "token2": np.exp(-0.2) / sum_probs,
    }
    assert errors == []


@patch("inverse_cai.models.logger")
def test_get_token_probs_with_missing_token(mock_logger):
    # Mock the model's generate method
    mock_model = Mock()
    mock_model.generate.return_value = Mock(
        generations=[
            [
                Mock(
                    generation_info={
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "token1", "logprob": 0.1},
                                    ]
                                }
                            ]
                        }
                    }
                )
            ]
        ]
    )

    tokens = ["token1", "token2"]
    messages = ["message1", "message2"]

    # Call the function
    token_probs, errors = get_token_probs(tokens, mock_model, messages)

    # Assert the results
    assert token_probs == {"token1": 1.0, "token2": 0.0}
    assert errors == ["token_not_found_in_top_logprobs_token2"]
    mock_logger.warning.assert_called_once_with(
        "Token token2 not found in top logprobs. Returning -1000000 logprob (close to 0 probability for token)."
    )

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
from inverse_cai.experiment.config import ExpConfig
from inverse_cai.algorithm.main import run


@pytest.fixture
def mock_feedback_df():
    return pd.DataFrame(
        {
            "text_a": ["text1", "text2"],
            "text_b": ["text3", "text4"],
            "preferred_text": ["text_a", "text_a"],
        }
    )


@pytest.fixture
def mock_config():
    return ExpConfig()


@pytest.fixture
def mock_save_path(tmp_path):
    # Create results directory in temporary path
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    return tmp_path


@patch("inverse_cai.algorithm.main.generate_principles_from_feedback")
@patch("inverse_cai.algorithm.main.cluster_principles")
@patch("inverse_cai.algorithm.main.get_cluster_summaries")
@patch("inverse_cai.algorithm.main.get_votes_for_principles")
@patch("inverse_cai.algorithm.main.filter_according_to_votes")
@patch("inverse_cai.algorithm.main.save_to_json")
@patch("inverse_cai.algorithm.main.print_clusters")
@patch("pathlib.Path.mkdir")  # Mock directory creation
@patch("pandas.DataFrame.to_csv")  # Mock CSV writing
def test_run_full_workflow(
    mock_to_csv,
    mock_mkdir,
    mock_print_clusters,
    mock_save_to_json,
    mock_filter_votes,
    mock_get_votes,
    mock_get_summaries,
    mock_cluster_principles,
    mock_generate_principles,
    mock_feedback_df,
    mock_config,
    mock_save_path,
):
    # Setup mock returns
    mock_principles = ["principle1", "principle2"]
    mock_prompt_principles = ["prompt_principle1", "prompt_principle2"]
    mock_generate_principles.return_value = (
        mock_feedback_df,
        mock_principles,
        mock_prompt_principles,
    )
    mock_feedback_df["principles"] = mock_principles
    mock_feedback_df["prompt_principles"] = mock_prompt_principles

    mock_cluster_principles.return_value = {
        "cluster1": ["principle1"],
        "cluster2": ["principle2"],
    }

    mock_get_summaries.return_value = {"cluster1": "summary1", "cluster2": "summary2"}

    mock_get_votes.return_value = (
        pd.DataFrame({"votes": [1, 2]}),
        {
            "cluster1": {
                "for": 3,
                "against": 1,
                "abstain": 0,
                "invalid": 0,
                "true": 3,
                "false": 1,
            },
            "cluster2": {
                "for": 2,
                "against": 2,
                "abstain": 1,
                "invalid": 0,
                "true": 2,
                "false": 2,
            },
        },
    )

    mock_filter_votes.return_value = ["cluster1"]

    # Run the function
    result = run(
        feedback=mock_feedback_df,
        save_path=mock_save_path,
        num_principles_per_sampling_step=2,
        num_rankings_per_sampling_step=1,
        num_clusters=2,
        random_clusters=False,
        skip_voting=False,
        require_majority_true=True,
        require_majority_relevant=False,
        require_majority_valid=True,
        require_minimum_relevance=0.1,
        order_by="for_minus_against",
        max_principles=5,
        ratio_of_max_principles_to_cluster_again=1.5,
        model_name="test-model",
        config=mock_config,
    )

    # Verify the workflow
    mock_generate_principles.assert_called_once()
    assert mock_cluster_principles.call_count == 2
    assert mock_get_summaries.call_count == 2
    assert mock_get_votes.call_count == 2
    assert mock_filter_votes.call_count == 1

    # Verify save_to_json calls
    assert mock_save_to_json.call_count >= 4

    # Check the returned dictionary structure
    expected_keys = {
        "feedback",
        "clusters",
        "prompt_clusters",
        "summaries",
        "prompt_summaries",
        "raw_votes",
        "combined_votes",
        "raw_prompt_votes",
        "combined_prompt_votes",
        "filtered_plinciples",
        "final_principles",
        "constitution",
    }
    result_keys = set(result.keys())
    assert (
        expected_keys == result_keys
    ), f"Expected keys: {expected_keys}, but got: {result_keys}."

    # Verify the constitution format
    assert isinstance(result["constitution"], str)
    assert result["constitution"].startswith("1. ")


@patch("inverse_cai.algorithm.main.generate_principles_from_feedback")
@patch("inverse_cai.algorithm.main.cluster_principles")
@patch("inverse_cai.algorithm.main.get_cluster_summaries")
@patch("inverse_cai.algorithm.main.save_to_json")
def test_run_skip_voting(
    mock_save_to_json,
    mock_get_summaries,
    mock_cluster_principles,
    mock_generate_principles,
    mock_feedback_df,
    mock_config,
    mock_save_path,
):
    # Setup mock returns
    mock_principles = ["principle1", "principle2"]
    mock_prompt_principles = ["prompt_principle1", "prompt_principle2"]
    mock_generate_principles.return_value = (
        mock_feedback_df,
        mock_principles,
        mock_prompt_principles,
    )
    mock_feedback_df["principles"] = mock_principles
    mock_feedback_df["prompt_principles"] = mock_prompt_principles

    mock_cluster_principles.return_value = {
        "cluster1": ["principle1"],
        "cluster2": ["principle2"],
    }

    mock_get_summaries.return_value = {"cluster1": "summary1", "cluster2": "summary2"}

    # Run with skip_voting=True
    result = run(
        feedback=mock_feedback_df,
        save_path=mock_save_path,
        num_principles_per_sampling_step=2,
        num_rankings_per_sampling_step=3,
        num_clusters=2,
        random_clusters=False,
        skip_voting=True,
        require_majority_true=True,
        require_majority_relevant=False,
        require_majority_valid=True,
        require_minimum_relevance=0.1,
        order_by="for_minus_against",
        max_principles=2,
        ratio_of_max_principles_to_cluster_again=1.5,
        model_name="test-model",
        config=mock_config,
    )

    # Verify workflow with skipped voting
    mock_generate_principles.assert_called_once()
    assert mock_cluster_principles.call_count == 2
    assert mock_get_summaries.call_count == 2

    # Check the returned dictionary structure
    assert result["combined_votes"] is None
    assert result["filtered_plinciples"] is None
    assert len(result["final_principles"]) == 2  # Should match max_principles


@patch("inverse_cai.models.get_model")
@patch("inverse_cai.algorithm.voting.random.choice")  # Add mock for random.choice
def test_run_integration(
    mock_random_choice, mock_get_model, mock_feedback_df, mock_config, mock_save_path
):
    # Setup mock model with different responses for different calls
    mock_model = AsyncMock()

    # Set random.choice to return False only when choosing between booleans
    def random_choice_side_effect(seq):
        return seq[0]

    mock_random_choice.side_effect = random_choice_side_effect

    async def side_effect(messages):
        message_text = str(messages)
        if "Given the data above, why do you think" in message_text:
            # For principle generation
            return Mock(
                content='{"principles": ["Select response that is more concise", "Select response that is more accurate"]}'
            )
        elif (
            "Given the data above, what features of the instruction were important to why one sample was selected over the other?"
            in message_text
        ):
            # For prompt-based feature generation
            return Mock(
                content='{"features": ["Select response that is more concise", "Select response that is more accurate"]}'
            )
        elif "check for each rule below" in message_text:
            # For voting on principles
            return Mock(content='{"0": "A", "1": "B"}')
        elif (
            "Given the prompt above, check whether each rule below is true or false"
            in message_text
        ):
            # For voting on prompt-based features
            return Mock(content="{0: true, 1: false}")
        elif "Your job is to summarize the principles" in message_text:
            # For summaries
            return Mock(content='{"summary": "Select response that is more effective"}')
        elif "Your job is to summarize the prompt" in message_text:
            # For summaries
            return Mock(content="Is the prompt...")
        else:
            # Default response for any other cases
            return Mock(content='{"default": "response"}')

    mock_model.ainvoke.side_effect = side_effect
    mock_get_model.return_value = mock_model

    # Disable parallel processing for the test
    mock_config.async_task_num = 1  # Force sequential processing

    # Run the function with minimal mocking
    result = run(
        feedback=mock_feedback_df,
        save_path=mock_save_path,
        num_principles_per_sampling_step=2,
        num_rankings_per_sampling_step=1,
        num_clusters=2,
        random_clusters=False,
        skip_voting=False,
        require_majority_true=True,
        require_majority_relevant=False,
        require_majority_valid=True,
        require_minimum_relevance=0.1,
        order_by="for_minus_against",
        max_principles=5,
        ratio_of_max_principles_to_cluster_again=1.5,
        model_name="test-model",
        config=mock_config,
    )

    # Verify the basic structure of the result
    assert isinstance(result, dict)
    assert "feedback" in result
    assert "clusters" in result
    assert "summaries" in result
    assert "combined_votes" in result
    assert "filtered_plinciples" in result
    assert "final_principles" in result
    assert "constitution" in result

    # Verify that the model was called
    assert mock_model.ainvoke.call_count > 0

    # Verify the constitution format
    assert isinstance(result["constitution"], str)
    assert result["constitution"].startswith("1. ")

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pathlib
from inverse_cai.experiment.config import ExpConfig
from inverse_cai.algorithm.main import run


@pytest.fixture
def mock_feedback_df():
    return pd.DataFrame(
        {
            "preferred_text": ["text1", "text2"],
            "rejected_text": ["text3", "text4"],
            "winner_model": ["model1", "model2"],
            "loser_model": ["model3", "model4"],
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
    mock_generate_principles.return_value = mock_feedback_df
    mock_feedback_df["principles"] = [["principle1"], ["principle2"]]

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
        num_principles_generated_per_ranking=2,
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
    mock_cluster_principles.assert_called_once()
    mock_get_summaries.assert_called_once()
    mock_get_votes.assert_called_once()
    mock_filter_votes.assert_called_once()

    # Verify save_to_json calls
    assert mock_save_to_json.call_count >= 4

    # Check the returned dictionary structure
    assert set(result.keys()) == {
        "feedback",
        "clusters",
        "summaries",
        "combined_votes",
        "filtered_plinciples",
        "final_principles",
        "constitution",
    }

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
    mock_generate_principles.return_value = mock_feedback_df
    mock_feedback_df["principles"] = [["principle1"], ["principle2"]]

    mock_cluster_principles.return_value = {
        "cluster1": ["principle1"],
        "cluster2": ["principle2"],
    }

    mock_get_summaries.return_value = {"cluster1": "summary1", "cluster2": "summary2"}

    # Run with skip_voting=True
    result = run(
        feedback=mock_feedback_df,
        save_path=mock_save_path,
        num_principles_generated_per_ranking=2,
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
    mock_cluster_principles.assert_called_once()
    mock_get_summaries.assert_called_once()

    # Check the returned dictionary structure
    assert result["combined_votes"] is None
    assert result["filtered_plinciples"] is None
    assert len(result["final_principles"]) == 2  # Should match max_principles

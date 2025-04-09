"""Tests for the annotated pairs format module."""

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from inverse_cai.data.annotated_pairs_format import (
    hash_string,
    hash_comparison,
    votes_to_annotations,
    add_annotators,
    create_annotated_pairs,
    DEFAULT_ANNOTATOR_DESCRIPTION,
    DEFAULT_ANNOTATOR_TYPE,
    DEFAULT_PREFERENCE_COLUMN,
)


def test_hash_string():
    # Test with predictable input
    assert hash_string("test") == "098f6bcd", "Hash should match expected value"

    # Test with empty string
    assert len(hash_string("")) == 8, "Hash should be 8 characters long"


def test_hash_comparison():
    # Test with two texts and no prompt
    text_a = "response A"
    text_b = "response B"
    result = hash_comparison(text_a, text_b, None)

    # The hash should be deterministic
    assert (
        hash_comparison(text_a, text_b, None) == result
    ), "Same inputs should produce same hash"

    # Test with prompt
    prompt = "What is the capital of France?"
    assert (
        hash_comparison(text_a, text_b, prompt) != result
    ), "Adding a prompt should change the hash"

    # Order matters
    assert (
        hash_comparison(text_b, text_a, None) != result
    ), "Swapping text_a and text_b should change the hash"


def test_votes_to_annotations():
    """Test the votes_to_annotations function."""
    # Setup test data
    votes = {1: True, 2: False, 3: None}
    principle_index_to_text = {1: "Be honest", 2: "Be helpful", 3: "Be concise"}
    active_principles = ["Be honest", "Be helpful", "Be concise"]
    reference_preference = "text_a"

    # Expected hashed IDs
    honest_id = hash_string("Be honest")
    helpful_id = hash_string("Be helpful")
    concise_id = hash_string("Be concise")

    # Run function
    result = votes_to_annotations(
        votes, principle_index_to_text, active_principles, reference_preference
    )

    # Verify results
    assert (
        result[honest_id]["pref"] == "text_a"
    ), "Principle with vote True should get reference_preference"
    assert (
        result[helpful_id]["pref"] == "text_b"
    ), "Principle with vote False should get opposite of reference_preference"
    assert (
        result[concise_id]["pref"] == "not_applicable"
    ), "Principle with vote None should get not_applicable"

    # Test with only some active principles
    active_principles = ["Be honest"]
    result = votes_to_annotations(
        votes, principle_index_to_text, active_principles, reference_preference
    )
    assert len(result) == 1, "Only active principles should be included"
    assert honest_id in result, "Only the active principle should be included"

    # Test with reference_preference = text_b
    reference_preference = "text_b"
    result = votes_to_annotations(
        votes, principle_index_to_text, active_principles, reference_preference
    )
    assert (
        result[honest_id]["pref"] == "text_b"
    ), "With text_b as reference, True vote should be text_b"

    # Test with unexpected vote value
    votes_with_unexpected = {1: "unexpected_value"}
    with pytest.raises(ValueError):
        votes_to_annotations(
            votes_with_unexpected,
            principle_index_to_text,
            active_principles,
            reference_preference,
        )


def test_add_annotators():
    """Test the add_annotators function."""
    # Setup test data
    output = {"annotators": {}, "metadata": {}}
    principles = {1: "Be honest", 2: "Be helpful"}
    filtered_principles = ["Be honest"]

    # Run function with filter_to_constitution=True
    add_annotators(output, principles, filtered_principles, filter_to_constitution=True)

    # Verify results
    assert len(output["annotators"]) == 2, "Should have default + 1 principle annotator"

    # Compute expected default annotator ID
    default_annotator_id = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)
    assert (
        default_annotator_id in output["annotators"]
    ), "Default annotator should be in output"
    assert (
        output["annotators"][default_annotator_id]["type"] == DEFAULT_ANNOTATOR_TYPE
    ), "Default annotator type should be set"
    assert (
        output["metadata"]["default_annotator"] == default_annotator_id
    ), "Default annotator should be set"

    # Verify principle annotator was added correctly
    principle_id = hash_string("Be honest")
    assert (
        principle_id in output["annotators"]
    ), "Principle annotator should be in output"
    assert (
        output["annotators"][principle_id]["description"] == "Be honest"
    ), "Principle description should be set"
    assert (
        output["annotators"][principle_id]["type"] == "principle"
    ), "Principle type should be set"

    # Run function with filter_to_constitution=False
    output = {"annotators": {}, "metadata": {}}
    add_annotators(
        output, principles, filtered_principles, filter_to_constitution=False
    )

    # Verify all principles are included when not filtering
    assert (
        len(output["annotators"]) == 3
    ), "Should have default + 2 principle annotators"

    # Verify both principles were added
    honest_id = hash_string("Be honest")
    helpful_id = hash_string("Be helpful")
    assert honest_id in output["annotators"], "First principle should be in output"
    assert helpful_id in output["annotators"], "Second principle should be in output"

    # Test with additional columns
    output = {"annotators": {}, "metadata": {}}
    additional_columns = ["column1", "column2"]
    add_annotators(
        output,
        principles,
        filtered_principles,
        filter_to_constitution=True,
        additional_columns=additional_columns,
    )

    # Verify column annotators were added
    column1_id = hash_string("column_column1")
    column2_id = hash_string("column_column2")
    assert column1_id in output["annotators"], "Column1 annotator should be in output"
    assert column2_id in output["annotators"], "Column2 annotator should be in output"
    assert (
        output["annotators"][column1_id]["type"] == "unknown"
    ), "Column type should be set"
    assert (
        output["annotators"][column2_id]["type"] == "unknown"
    ), "Column type should be set"


def test_create_annotated_pairs():
    """Test the create_annotated_pairs function."""
    # Setup test data
    train_df = pd.DataFrame(
        {
            "text_a": ["Response A"],
            "text_b": ["Response B"],
            "input": ["What is the capital of France?"],
            DEFAULT_PREFERENCE_COLUMN: ["text_a"],
            "model_a": ["Model X"],
            "model_b": ["Model Y"],
        }
    )

    principles = {1: "Be honest", 2: "Be helpful"}
    filtered_principles = ["Be honest"]
    comparison_votes = {0: {1: True, 2: False}}
    dataset_name = "Test Dataset"

    # Run function
    result = create_annotated_pairs(
        train_df, principles, filtered_principles, comparison_votes, dataset_name
    )

    # Verify the structure
    assert "metadata" in result, "Result should have metadata"
    assert "annotators" in result, "Result should have annotators"
    assert "comparisons" in result, "Result should have comparisons"

    # Verify metadata
    assert (
        result["metadata"]["dataset_name"] == dataset_name
    ), "Dataset name should be set"
    assert result["metadata"]["version"] == "1.0", "Version should be set"

    # Verify annotators
    default_annotator_id = None
    for annotator_id, annotator in result["annotators"].items():
        if annotator["type"] == DEFAULT_ANNOTATOR_TYPE:
            default_annotator_id = annotator_id
    assert default_annotator_id is not None, "Should have a default annotator"

    # Verify comparison
    assert len(result["comparisons"]) == 1, "Should have 1 comparison"
    comparison = result["comparisons"][0]
    assert comparison["text_a"] == "Response A", "text_a should be set"
    assert comparison["text_b"] == "Response B", "text_b should be set"
    assert (
        comparison["prompt"] == "What is the capital of France?"
    ), "prompt should be set"

    # Verify annotations
    annotations = comparison["annotations"]
    assert (
        default_annotator_id in annotations
    ), "Default annotator should be in annotations"
    assert (
        annotations[default_annotator_id]["pref"] == "text_a"
    ), "Default annotation should match preferred_text"

    honest_id = hash_string("Be honest")
    assert honest_id in annotations, "Principle annotator should be in annotations"
    assert (
        annotations[honest_id]["pref"] == "text_a"
    ), "Principle annotation should be correct"


def test_create_annotated_pairs_with_additional_columns():
    """Test the create_annotated_pairs function with additional columns."""
    # Setup test data
    train_df = pd.DataFrame(
        {
            "text_a": ["Response A"],
            "text_b": ["Response B"],
            "input": ["What is the capital of France?"],
            DEFAULT_PREFERENCE_COLUMN: ["text_a"],
            "model_a": ["Model X"],
            "model_b": ["Model Y"],
            "additional_column": ["Some value"],
        }
    )

    principles = {1: "Be honest", 2: "Be helpful"}
    filtered_principles = ["Be honest"]
    comparison_votes = {0: {1: True, 2: False}}
    dataset_name = "Test Dataset"
    additional_columns = ["additional_column"]

    # Run function
    result = create_annotated_pairs(
        train_df,
        principles,
        filtered_principles,
        comparison_votes,
        dataset_name,
        additional_columns=additional_columns,
    )

    # Verify the structure
    assert "metadata" in result, "Result should have metadata"
    assert "annotators" in result, "Result should have annotators"
    assert "comparisons" in result, "Result should have comparisons"

    # Verify metadata
    assert (
        result["metadata"]["dataset_name"] == dataset_name
    ), "Dataset name should be set"
    assert result["metadata"]["version"] == "1.0", "Version should be set"

    # Verify annotators
    default_annotator_id = None
    unknown_annotator_id = None
    for annotator_id, annotator in result["annotators"].items():
        if annotator["description"] == DEFAULT_ANNOTATOR_DESCRIPTION:
            default_annotator_id = annotator_id
        if annotator["type"] == "unknown":
            unknown_annotator_id = annotator_id
    assert default_annotator_id is not None, "Should have a default annotator"
    assert (
        unknown_annotator_id is not None
    ), "Should have a unknown annotator, has annotators " + str(
        [annotator["type"] for annotator in result["annotators"].values()]
    )

    # Verify comparison
    assert len(result["comparisons"]) == 1, "Should have 1 comparison"
    comparison = result["comparisons"][0]
    assert comparison["text_a"] == "Response A", "text_a should be set"
    assert comparison["text_b"] == "Response B", "text_b should be set"
    assert (
        comparison["prompt"] == "What is the capital of France?"
    ), "prompt should be set"

    # Verify annotations
    annotations = comparison["annotations"]
    assert (
        default_annotator_id in annotations
    ), "Default annotator should be in annotations"
    assert (
        annotations[default_annotator_id]["pref"] == "text_a"
    ), "Default annotation should match preferred_text"

    assert (
        unknown_annotator_id in annotations
    ), "Unknown annotator should be in annotations"

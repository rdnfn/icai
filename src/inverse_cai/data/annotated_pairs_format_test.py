"""Tests for the annotated pairs format module."""

import pandas as pd
import pytest

from inverse_cai.data.annotated_pairs_format import (
    hash_string,
    hash_comparison,
    votes_to_annotations,
    add_annotators,
    create_annotated_pairs,
    merge_annotated_pairs,
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
    # Test with two response dictionaries and no prompt
    response_a = {"text": "response A", "model": "Model X"}
    response_b = {"text": "response B", "model": "Model Y"}
    result = hash_comparison(response_a, response_b, None)

    # The hash should be deterministic
    assert (
        hash_comparison(response_a, response_b, None) == result
    ), "Same inputs should produce same hash"

    # Test with prompt
    prompt = "What is the capital of France?"
    assert (
        hash_comparison(response_a, response_b, prompt) != result
    ), "Adding a prompt should change the hash"

    # Order matters
    assert (
        hash_comparison(response_b, response_a, None) != result
    ), "Swapping response_a and response_b should change the hash"

    # Test that model information affects the hash
    response_a_different_model = {"text": "response A", "model": "Model Z"}
    assert (
        hash_comparison(response_a_different_model, response_b, None) != result
    ), "Changing model information should change the hash"

    # Test with additional response information
    response_a_with_extra = {
        "text": "response A",
        "model": "Model X",
        "metadata": {"timestamp": "2023-01-01"},
    }
    assert (
        hash_comparison(response_a_with_extra, response_b, None) != result
    ), "Adding extra response information should change the hash"


def test_votes_to_annotations():
    """Test the votes_to_annotations function."""
    # Setup test data
    votes = {1: True, 2: False, 3: None, 4: "invalid"}
    principle_index_to_text = {
        1: "Be honest",
        2: "Be helpful",
        3: "Be concise",
        4: "Be creative",
    }
    active_principles = ["Be honest", "Be helpful", "Be concise", "Be creative"]
    reference_preference = "a"

    # Expected hashed IDs
    honest_id = hash_string("Be honest")
    helpful_id = hash_string("Be helpful")
    concise_id = hash_string("Be concise")
    creative_id = hash_string("Be creative")

    # Run function
    result = votes_to_annotations(
        votes, principle_index_to_text, active_principles, reference_preference
    )

    # Verify results
    assert (
        result[honest_id]["pref"] == "a"
    ), "Principle with vote True should get reference_preference"
    assert (
        result[helpful_id]["pref"] == "b"
    ), "Principle with vote False should get opposite of reference_preference"
    assert (
        result[concise_id]["pref"] is None
    ), "Principle with vote 'not_applicable' should get None pref"
    assert (
        result[concise_id]["no_pref_reason"] == "not_applicable"
    ), "Principle with vote 'not_applicable' should have 'not_applicable' reason"
    assert (
        result[creative_id]["pref"] is None
    ), "Principle with vote 'invalid' should get None pref"
    assert (
        result[creative_id]["no_pref_reason"] == "invalid"
    ), "Principle with vote 'invalid' should have 'invalid' reason"

    # Test with only some active principles
    active_principles = ["Be honest"]
    result = votes_to_annotations(
        votes, principle_index_to_text, active_principles, reference_preference
    )
    assert len(result) == 1, "Only active principles should be included"
    assert honest_id in result, "Only the active principle should be included"

    # Test with reference_preference = b
    reference_preference = "b"
    result = votes_to_annotations(
        votes, principle_index_to_text, active_principles, reference_preference
    )
    assert (
        result[honest_id]["pref"] == "b"
    ), "With b as reference, True vote should be b"

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

    # Run function with filter_to_constitution=True
    add_annotators(output, principles)

    # Verify results
    assert (
        len(output["annotators"]) == 3
    ), "Should have default + 2 principle annotators"

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
    num_comparisons = 10
    train_df = pd.DataFrame(
        {
            "text_a": ["Response A"] * num_comparisons,
            "text_b": ["Response B"] * num_comparisons,
            "input": ["What is the capital of France?"] * num_comparisons,
            DEFAULT_PREFERENCE_COLUMN: ["a", "b", "text_a", "text_b", "text_b"] * 2,
            "model_a": ["Model X"] * num_comparisons,
            "model_b": ["Model Y"] * num_comparisons,
        }
    )

    principles = {1: "Be honest", 2: "Be helpful"}
    comparison_votes = {0: {1: True, 2: False}}
    dataset_name = "Test Dataset"

    # Run function
    result = create_annotated_pairs(
        df=train_df,
        principles=principles,
        comparison_votes=comparison_votes,
        dataset_name=dataset_name,
    )

    # Verify the structure
    assert "metadata" in result, "Result should have metadata"
    assert "annotators" in result, "Result should have annotators"
    assert "comparisons" in result, "Result should have comparisons"

    # Verify metadata
    assert (
        result["metadata"]["dataset_name"] == dataset_name
    ), "Dataset name should be set"
    assert result["metadata"]["version"] == "2.0", "Version should be set"

    # Verify annotators
    default_annotator_id = None
    for annotator_id, annotator in result["annotators"].items():
        if annotator["type"] == DEFAULT_ANNOTATOR_TYPE:
            default_annotator_id = annotator_id
    assert default_annotator_id is not None, "Should have a default annotator"

    # Verify comparison
    assert len(result["comparisons"]) == num_comparisons, "Should have 10 comparisons"
    comparison = result["comparisons"][0]

    # Check response_a and response_b format
    assert "response_a" in comparison, "response_a should be present"
    assert "response_b" in comparison, "response_b should be present"
    assert (
        comparison["response_a"]["text"] == "Response A"
    ), "response_a.text should be set"
    assert (
        comparison["response_b"]["text"] == "Response B"
    ), "response_b.text should be set"
    assert (
        comparison["response_a"]["model"] == "Model X"
    ), "response_a.model should be set"
    assert (
        comparison["response_b"]["model"] == "Model Y"
    ), "response_b.model should be set"
    assert (
        comparison["prompt"] == "What is the capital of France?"
    ), "prompt should be set"

    # Verify annotations
    annotations = comparison["annotations"]
    assert (
        default_annotator_id in annotations
    ), "Default annotator should be in annotations"
    assert (
        annotations[default_annotator_id]["pref"] == "a"
    ), "Default annotation should be converted to 'a'"

    honest_id = hash_string("Be honest")
    assert honest_id in annotations, "Principle annotator should be in annotations"
    assert (
        annotations[honest_id]["pref"] == "a"
    ), "Principle annotation should be correct and use 'a'"

    # check if other default preference columns are converted correctly for first five comparisons
    first_five_comparisons = result["comparisons"][:5]
    default_annotator_hash = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)
    first_five_annotations = [
        first_five_comparisons[i]["annotations"][default_annotator_hash]["pref"]
        for i in range(5)
    ]
    correct_prefs = ["a", "b", "a", "b", "b"]
    assert (
        first_five_annotations == correct_prefs
    ), f"Reference annotations should be {correct_prefs}, got {first_five_annotations} instead"


def test_create_annotated_pairs_with_additional_columns():
    """Test the create_annotated_pairs function with additional columns."""
    # Setup test data
    train_df = pd.DataFrame(
        {
            "text_a": ["Response A"],
            "text_b": ["Response B"],
            "input": ["What is the capital of France?"],
            DEFAULT_PREFERENCE_COLUMN: ["a"],
            "model_a": ["Model X"],
            "model_b": ["Model Y"],
            "additional_column": ["Some value"],
        }
    )

    principles = {1: "Be honest", 2: "Be helpful"}
    comparison_votes = {0: {1: True, 2: False}}
    dataset_name = "Test Dataset"
    additional_columns = ["additional_column"]

    # Run function
    result = create_annotated_pairs(
        df=train_df,
        principles=principles,
        comparison_votes=comparison_votes,
        dataset_name=dataset_name,
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
    assert result["metadata"]["version"] == "2.0", "Version should be set"

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

    # Check response_a and response_b format
    assert "response_a" in comparison, "response_a should be present"
    assert "response_b" in comparison, "response_b should be present"
    assert (
        comparison["response_a"]["text"] == "Response A"
    ), "response_a.text should be set"
    assert (
        comparison["response_b"]["text"] == "Response B"
    ), "response_b.text should be set"
    assert (
        comparison["prompt"] == "What is the capital of France?"
    ), "prompt should be set"

    # Verify annotations
    annotations = comparison["annotations"]
    assert (
        default_annotator_id in annotations
    ), "Default annotator should be in annotations"
    assert (
        annotations[default_annotator_id]["pref"] == "a"
    ), "Default annotation should be 'a'"

    assert (
        unknown_annotator_id in annotations
    ), "Unknown annotator should be in annotations"


def test_merge_annotated_pairs_basic():
    """Test basic merging of two datasets."""
    # Create two simple datasets with different comparisons
    dataset1 = {
        "metadata": {
            "version": "2.0",
            "dataset_name": "dataset1",
            "default_annotator": "default1",
        },
        "annotators": {"default1": {"type": "unknown", "name": "default"}},
        "comparisons": [
            {
                "id": "comp1",
                "prompt": "prompt1",
                "response_a": {"text": "A1"},
                "response_b": {"text": "B1"},
                "annotations": {"default1": {"pref": "a"}},
            }
        ],
    }

    dataset2 = {
        "metadata": {
            "version": "2.0",
            "dataset_name": "dataset2",
            "default_annotator": "default2",
        },
        "annotators": {"default2": {"type": "unknown", "name": "default"}},
        "comparisons": [
            {
                "id": "comp2",
                "prompt": "prompt2",
                "response_a": {"text": "A2"},
                "response_b": {"text": "B2"},
                "annotations": {"default2": {"pref": "b"}},
            }
        ],
    }

    result = merge_annotated_pairs([dataset1, dataset2])

    assert len(result["comparisons"]) == 2
    assert len(result["annotators"]) == 2
    assert "dataset1_dataset2" in result["metadata"]["dataset_name"]
    assert result["metadata"]["default_annotator"] == "default1"


def test_merge_annotated_pairs_same_comparison():
    """Test merging datasets with the same comparison but different annotations."""
    dataset1 = {
        "metadata": {"version": "2.0", "dataset_name": "dataset1"},
        "annotators": {
            "ann1": {"type": "human", "name": "Annotator 1"},
            "ann2": {"type": "human", "name": "Annotator 2"},
        },
        "comparisons": [
            {
                "id": "same_comp",
                "prompt": "prompt",
                "response_a": {"text": "A"},
                "response_b": {"text": "B"},
                "annotations": {"ann1": {"pref": "a"}},
                "metadata": {"source": "test1"},
            }
        ],
    }

    dataset2 = {
        "metadata": {"version": "2.0", "dataset_name": "dataset2"},
        "annotators": {
            "ann2": {"type": "human", "name": "Annotator 2"},
            "ann3": {"type": "principle", "description": "Principle"},
        },
        "comparisons": [
            {
                "id": "same_comp",
                "prompt": "prompt",
                "response_a": {"text": "A"},
                "response_b": {"text": "B"},
                "annotations": {"ann2": {"pref": "a"}, "ann3": {"pref": "b"}},
                "metadata": {"model": "gpt-4"},
            }
        ],
    }

    result = merge_annotated_pairs([dataset1, dataset2])

    assert len(result["comparisons"]) == 1
    assert len(result["annotators"]) == 3

    comparison = result["comparisons"][0]
    assert len(comparison["annotations"]) == 3
    assert comparison["annotations"]["ann1"]["pref"] == "a"
    assert comparison["annotations"]["ann2"]["pref"] == "a"
    assert comparison["annotations"]["ann3"]["pref"] == "b"
    assert comparison["metadata"]["source"] == "test1"
    assert comparison["metadata"]["model"] == "gpt-4"


def test_merge_annotated_pairs_validation():
    """Test validation during merging."""
    dataset1 = {"metadata": {"version": "2.0"}, "comparisons": [], "annotators": {}}
    dataset2 = {"metadata": {"version": "1.0"}, "comparisons": [], "annotators": {}}

    with pytest.raises(ValueError, match="same format version"):
        merge_annotated_pairs([dataset1, dataset2])

    # Test empty list
    with pytest.raises(ValueError, match="No annotated pairs"):
        merge_annotated_pairs([])

    # Test dataset with duplicate comparison IDs
    dataset_with_dupes = {
        "metadata": {"version": "2.0", "dataset_name": "dupes"},
        "annotators": {},
        "comparisons": [
            {"id": "comp1", "annotations": {}},
            {"id": "comp1", "annotations": {}},
        ],
    }

    with pytest.raises(ValueError, match="not unique"):
        merge_annotated_pairs([dataset_with_dupes])


def test_merge_annotated_pairs_conflicting_annotations():
    """Test that conflicting annotations raise an error."""
    dataset1 = {
        "metadata": {"version": "2.0"},
        "annotators": {"ann1": {}},
        "comparisons": [{"id": "comp1", "annotations": {"ann1": {"pref": "a"}}}],
    }

    dataset2 = {
        "metadata": {"version": "2.0"},
        "annotators": {"ann1": {}},
        "comparisons": [{"id": "comp1", "annotations": {"ann1": {"pref": "b"}}}],
    }

    with pytest.raises(AssertionError, match="not the same"):
        merge_annotated_pairs([dataset1, dataset2])

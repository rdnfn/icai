"""Tests for the ICAI data loader module."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from inverse_cai.data.loader.icai import (
    load_train_data,
    load_principles,
    load_filtered_principles,
    load_votes_per_comparison,
    python_dict_str_to_json_compatible,
)


def test_load_train_data():
    """Test the load_train_data function."""
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        csv_content = """index,text_a,text_b,preferred_text
0,"In the heart of a bustling city, a sleek black cat named Shadow prowled the moonlit rooftops, her eyes gleaming with curiosity and mischief. She discovered a hidden garden atop an old apartment building, where she danced under the stars, chasing fireflies that glowed like tiny lanterns. As dawn painted the sky in hues of orange and pink, Shadow found her way back home, carrying the secret of the garden in her heart.","Across the town, in a cozy neighborhood, a golden retriever named Buddy embarked on his daily adventure, tail wagging with uncontainable excitement. He found a lost toy under the bushes in the park, its colors faded and fabric worn, but to Buddy, it was a treasure untold. Returning home with his newfound prize, Buddy's joyful barks filled the air, reminding everyone in the house that happiness can be found in the simplest of things.","text_a"
"""
        csv_path = tmp_path / "000_train_data.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # Test the function
        result = load_train_data(tmp_path)

        # Verify the result has the expected structure and data
        assert len(result) == 1, "Should have 1 row"
        assert list(result.columns) == [
            "index",
            "text_a",
            "text_b",
            "preferred_text",
        ], "Columns should match"
        assert (
            result.iloc[0]["preferred_text"] == "text_a"
        ), "First row should prefer text_a"
        assert (
            "Shadow" in result.iloc[0]["text_a"]
        ), "First row text_a should contain expected content"
        assert (
            "Buddy" in result.iloc[0]["text_b"]
        ), "First row text_b should contain expected content"


def test_load_principles():
    """Test the load_principles function."""
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        test_json = """{
  "1": "Prioritize responses that provide accurate and factual information",
  "2": "Prefer responses that are helpful, informative, and directly address the user's query",
  "3": "Favor responses that demonstrate critical thinking and nuanced understanding"
}"""
        json_path = tmp_path / "030_distilled_principles_per_cluster.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(test_json)

        # Test the function
        result = load_principles(tmp_path)

        # Verify the result
        expected = {
            1: "Prioritize responses that provide accurate and factual information",
            2: "Prefer responses that are helpful, informative, and directly address the user's query",
            3: "Favor responses that demonstrate critical thinking and nuanced understanding",
        }
        assert result == expected


def test_load_filtered_principles():
    """Test the load_filtered_principles function."""
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        test_json = """[
  "Prioritize responses that provide accurate and factual information",
  "Prefer responses that are helpful, informative, and directly address the user's query"
]"""
        json_path = tmp_path / "050_filtered_principles.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(test_json)

        # Test the function
        result = load_filtered_principles(tmp_path)

        # Verify the result
        expected = [
            "Prioritize responses that provide accurate and factual information",
            "Prefer responses that are helpful, informative, and directly address the user's query",
        ]
        assert result == expected, "Should load the principles list correctly"


def test_load_comparison_votes():
    """Test the load_comparison_votes function."""
    # For this test, we need to create a CSV file with properly formatted vote data
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        csv_content = """index,votes
0,"{0: True, 1: False, 2: None}"
1,"{0: True, 1: None, 2: 'invalid'}"
"""
        csv_path = tmp_path / "040_votes_per_comparison.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # Test the function
        result = load_votes_per_comparison(tmp_path)

        # Verify the result structure
        assert 0 in result, "Index 0 should be in the result"
        assert 1 in result, "Index 1 should be in the result"

        # Verify the votes for index 0
        assert 1 in result[0], "Principle 1 should be in votes for index 0"
        assert result[0][0] is True
        assert result[0][1] is False
        assert result[0][2] is None
        assert result[1][0] is True
        assert result[1][1] is None
        assert result[1][2] == "invalid"


def test_python_dict_str_to_json_compatible():
    """Test the python_dict_str_to_json_compatible utility function."""
    # Test basic conversion of Python values to JSON
    python_str = "{1: True, 2: False, 3: None, 4: 'invalid'}"
    json_str = python_dict_str_to_json_compatible(python_str)

    # Verify conversion
    assert '"1"' in json_str, "Keys should be quoted"
    assert '"2"' in json_str, "Keys should be quoted"
    assert '"3"' in json_str, "Keys should be quoted"
    assert "true" in json_str, "True should be converted to true"
    assert "false" in json_str, "False should be converted to false"
    assert "null" in json_str, "None should be converted to null"
    assert '"invalid"' in json_str, "Single quotes should be converted to double quotes"

    # Test that the result can be parsed by json.loads
    result = json.loads(json_str)
    assert result == {"1": True, "2": False, "3": None, "4": "invalid"}

    # Test with more complex dictionary
    complex_str = "{1: True, 20: False, 300: None, 4000: True}"
    json_str = python_dict_str_to_json_compatible(complex_str)

    # Verify all keys are properly quoted
    result = json.loads(json_str)
    assert result == {"1": True, "20": False, "300": None, "4000": True}

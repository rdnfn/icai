"""Functions for loading ICAI experiment results."""

import json
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import regex as re

TRAIN_DATA_FILENAME = "000_train_data.csv"
PRINCIPLES_FILENAME = "030_distilled_principles_per_cluster.json"
COMPARISON_VOTES_FILENAME = "040_votes_per_comparison.csv"
FILTERED_PRINCIPLES_FILENAME = "050_filtered_principles.json"


def python_dict_str_to_json_compatible(dict_str: str) -> str:
    """Convert a Python dictionary string to JSON-compatible format.

    Useful for loading dictionaries that are stored as strings without having to
    use eval.

    Args:
        dict_str: A string representation of a Python dictionary

    Returns:
        A JSON-compatible string that can be parsed with json.loads
    """
    # Replace Python-style booleans and None with JSON-compatible values
    dict_str = re.sub(r"(?<=: )None", "null", dict_str)
    dict_str = re.sub(r"(?<=: )True", "true", dict_str)
    dict_str = re.sub(r"(?<=: )False", "false", dict_str)

    # Add double quotes around keys
    dict_str = re.sub(r"({|, )(\d+):", r'\1"\2":', dict_str)

    # Replace single quotes with double quotes for string values
    dict_str = re.sub(r"(?<=: )'(.*?)'(?=,|})", r'"\1"', dict_str)

    return dict_str


def load_train_data(results_path: Path) -> pd.DataFrame:
    """Load training data from the results directory.

    Args:
        results_path: Path to the results directory.

    Returns:
        A pandas DataFrame containing the training data.
    """
    train_data_path = results_path / TRAIN_DATA_FILENAME
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_DATA_FILENAME} in {results_path}"
        )
    return pd.read_csv(train_data_path)


def load_principles(results_path: Path) -> Dict[int, str]:
    """Load principle data from the results directory.

    Args:
        results_path: Path to the results directory.

    Returns:
        A dictionary mapping principle IDs to their corresponding principles.
    """
    principles = {}
    distilled_principles_path = results_path / PRINCIPLES_FILENAME
    if not distilled_principles_path.exists():
        raise FileNotFoundError(
            f"Could not find {PRINCIPLES_FILENAME} in {results_path}"
        )
    with open(distilled_principles_path, "r", encoding="utf-8") as f:
        principles_dict = json.load(f)
        # Convert keys to integers
        principles_dict = {int(k): v for k, v in principles_dict.items()}
        principles = principles_dict
    return principles


def load_votes_per_comparison(
    results_path: Path,
) -> Dict[int, Dict[int, Union[bool, str, None]]]:
    """Load comparison votes from the results directory.

    Args:
        results_path: Path to the results directory.

    Returns:
        A dictionary mapping comparison IDs (matching the ones from the training
        dataset) to a mapping from principle id (as returned by load_principles) to
        a boolean indicating whether the comparison agrees or disagrees with the
        principle ("the comparison votes in favor of the principle"), None for
        not applicable votes, or the string value "invalid".
    """
    comparison_votes = {}
    votes_per_comparison_path = results_path / COMPARISON_VOTES_FILENAME
    if not votes_per_comparison_path.exists():
        raise FileNotFoundError(
            f"Could not find {COMPARISON_VOTES_FILENAME} in {results_path}"
        )
    votes_df = pd.read_csv(votes_per_comparison_path)
    return parse_raw_votes(votes_df)


def parse_raw_votes(
    raw_votes: pd.Series | pd.DataFrame,
) -> Dict[int, Dict[int, Union[bool, str, None]]]:
    """Parse raw votes from a pandas Series or DataFrame.

    For parsing raw votes which either were loaded from a json or
    directly passed as raw votes inside the ICAI pipeline.

    Args:
        votes_series: A pandas Series or DataFrame containing the raw votes.

    Returns:
        A dictionary mapping comparison IDs to a mapping from principle id to vote value.
    """
    comparison_votes = {}
    if isinstance(raw_votes, pd.Series):
        raw_votes = raw_votes.to_frame(name="votes")
        raw_votes["index"] = raw_votes.index

    assert "index" in raw_votes.columns, "Index column must be present"

    for _, row in raw_votes.iterrows():
        idx = row["index"]
        votes_value = row["votes"]
        if isinstance(votes_value, str):
            votes_dict = json.loads(python_dict_str_to_json_compatible(votes_value))
        elif isinstance(votes_value, dict):
            votes_dict = votes_value
        else:
            raise ValueError(
                f"Unexpected vote value: {votes_value} (type: {type(votes_value)})"
            )
        comparison_votes[idx] = {int(k): v for k, v in votes_dict.items()}
    return comparison_votes


def load_filtered_principles(results_path: Path) -> List[str]:
    """Load filtered principles from the results directory.

    Args:
        results_path: Path to the results directory.

    Returns:
        A list of the principles remaining after filtering. The list contains
        strings that match a subset of the principles returned by
        load_principles.
    """
    filtered_principles = []
    filtered_principles_path = results_path / FILTERED_PRINCIPLES_FILENAME
    if not filtered_principles_path.exists():
        raise FileNotFoundError(
            f"Could not find {FILTERED_PRINCIPLES_FILENAME} in {results_path}"
        )
    with open(filtered_principles_path, "r", encoding="utf-8") as f:
        filtered_principles = json.load(f)
    return filtered_principles

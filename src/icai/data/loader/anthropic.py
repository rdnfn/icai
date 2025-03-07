"""Loading anthropic preference data."""

import pandas as pd
import json


def load_original_jsonl_file(
    file_path: str,
    switch_labels: bool = False,
    max_text_length: int = None,
) -> pd.DataFrame:
    """
    Load the anthropic preference data from a JSON file.

    Note: there are no ties in anthropic preference data.
    Thus, no removing ties necessary.

    Args:
        file_path: The file path to the JSON file.
        switch_labels: Whether to switch the labels.
        max_text_length: The maximum text length to allow.

    Returns:
        The anthropic preference data as a pandas DataFrame.
    """

    # List to hold all the loaded JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, "r") as file:
        for line in file:
            # Parse the JSON object in the current line and add it to the list
            data.append(json.loads(line))

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # switch original labels
    if switch_labels:
        df = switch_labels_in_df(df)

    # Truncate the data if necessary
    if max_text_length:
        df = limit_text_length(df, max_text_length)

    standard_df = get_standard_df(df)

    return standard_df


def load(
    path: str,
    switch_labels: bool = False,
    limit_text_length: int = None,
) -> pd.DataFrame:
    """
    Load the anthropic preference data from a CSV file.

    Args:
        path (str): The path to the CSV file.
        switch_labels (bool): Whether to switch the labels.
        limit_text_length (int): The maximum text length to allow.
    """

    # Load the CSV file
    df = pd.read_csv(path)

    # Limit text length
    if limit_text_length:
        df = limit_text_length(df, limit_text_length)

    # switch original labels
    if switch_labels:
        df = switch_labels_in_df(df)

    return df


def get_standard_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the original df to df with standard columns.
    """
    standard_df = pd.DataFrame()
    standard_df.index = df.index
    standard_df["text_a"] = df["chosen"]
    standard_df["text_b"] = df["rejected"]
    standard_df["preferred_text"] = "text_a"
    return standard_df


def limit_text_length(df: pd.DataFrame, max_text_length: int) -> pd.DataFrame:
    """
    Limits the text length in the data frame to the given maximum text length.
    """
    return df[df.applymap(str).applymap(len).lt(max_text_length).all(axis=1)]


def switch_labels_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Switches the chosen and rejected column in data frame
    """
    df["chosen"], df["rejected"] = df["rejected"], df["chosen"]
    return df

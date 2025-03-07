"""Functions for loading standard preference data."""

import pandas as pd


def switch_pref_labels_in_df(df):
    """
    Switches model_a and model_b labels.
    """

    def switch_preferred(row):
        win_col = "preferred_text"

        if row[win_col] == "text_a":
            return "text_b"
        elif row[win_col] == "text_b":
            return "text_a"
        else:
            return row[win_col]

    df["preferred_text"] = df.apply(switch_preferred, axis=1)
    return df


def load(path: str, switch_labels: bool = False) -> pd.DataFrame:
    """
    Load the standard preference data from a CSV file.

    Args:
        path (str): The path to the CSV file.
    """

    # Load the CSV file
    df = pd.read_csv(path)

    # switch original labels
    if switch_labels:
        df = switch_pref_labels_in_df(df)

    return df

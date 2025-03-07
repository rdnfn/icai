"""Parse and process data for inverse_cai."""

import ast
import pandas as pd
import pathlib
from typing import Optional


def get_standard_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the original df to df with standard columns.
    """
    standard_df = pd.DataFrame()
    standard_df.index = df.index

    def generate_conversation_string(messages):
        conversation = ""
        for message in messages:
            conversation += f"### {message['role']}:\n{message['content']}\n\n"
        return conversation

    standard_df["text_a"] = df["conversation_a"].apply(generate_conversation_string)
    standard_df["text_b"] = df["conversation_b"].apply(generate_conversation_string)

    def get_winner(row):
        if row["winner"] == "model_a":
            return "text_a"
        elif row["winner"] == "model_b":
            return "text_b"
        elif row["winner"] == "tie":
            return "tie"
        elif row["winner"] == "tie (bothbad)":
            return "tie (bothbad)"
        else:
            raise ValueError(f"Invalid winner value: {row['winner']}")

    standard_df["preferred_text"] = df.apply(get_winner, axis=1)

    return standard_df


def switch_labels_in_df(df):
    """
    Switches model_a and model_b labels.
    """

    def switch_winner(row):
        if row["winner"] == "model_a":
            return "model_b"
        elif row["winner"] == "model_b":
            return "model_a"
        else:
            return row["winner"]

    df["winner"] = df.apply(switch_winner, axis=1)
    return df


def load_raw(
    path: pathlib.Path,
    switch_labels: bool = False,
    remove_ties: bool = True,
    change_to_standard_df: bool = True,
    filter_by_user: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the raw chatbot arena data from a CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(path)

    list_of_row_dicts = [ast.literal_eval(x) for x in df["train"].tolist()]
    df = pd.DataFrame(list_of_row_dicts)

    if filter_by_user is not None:
        df = df[df["judge"] == filter_by_user]

    # switch original labels
    if switch_labels:
        df = switch_labels_in_df(df)

    if change_to_standard_df:
        standard_df = get_standard_df(df)
    else:
        standard_df = df

    if remove_ties:
        if not change_to_standard_df:
            raise ValueError("remove_ties=True only works when get_standard_df=True")
        standard_df = standard_df[
            (standard_df["preferred_text"] != "tie")
            & (standard_df["preferred_text"] != "tie (bothbad)")
        ]

    return standard_df

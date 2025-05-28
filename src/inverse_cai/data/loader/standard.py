"""Functions for loading standard preference data."""

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = ["text_a", "text_b"]


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


def load(
    path: str, switch_labels: bool = False, merge_prompts: bool = True
) -> pd.DataFrame:
    """
    Load the standard preference data from a CSV file.

    Args:
        path (str): The path to the CSV file.
    """

    # Load the CSV file
    df = pd.read_csv(path)

    # Check that the required columns are present
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {path} is missing required columns: {missing_columns}"
        )

    if "prompt" in df.columns and merge_prompts:
        df = _add_prompt_to_texts(df)

    # switch original labels
    if switch_labels:
        df = switch_pref_labels_in_df(df)

    return df


def _add_prompt_to_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the prompt to the texts.
    """

    # sanity check to see if the prompt is always in the response
    # using 100 first rows
    prompt_in_response = df.iloc[:100].apply(
        lambda row: str(row["prompt"]) in str(row["text_a"])
        and str(row["prompt"]) in str(row["text_b"]),
        axis=1,
    )
    if prompt_in_response.all():
        logger.warning(
            "Prompt is always in the response. This is unexpected. "
            "This is likely because the prompt is already included in the text columns."
            "Skipping merging the prompt and text columns."
        )
        return df

    logger.info("Combining prompts and texts since separate prompt column provided.")

    df["_og_text_a"] = df["text_a"]
    df["_og_text_b"] = df["text_b"]

    # Parse the prompt and text together into
    # a chatbot conversation.
    for text_name in ["text_a", "text_b"]:
        df[text_name] = df[["prompt", text_name]].apply(
            lambda x: str(
                [
                    {"role": "user", "content": x["prompt"]},
                    {"role": "assistant", "content": x[text_name]},
                ]
            ),
            axis=1,
        )

    return df

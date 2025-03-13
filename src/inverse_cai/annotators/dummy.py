import random
import pandas as pd


def annotate_perfectly(data: pd.DataFrame, icai_results_dict: dict) -> pd.DataFrame:
    """Annotate the data perfectly."""
    data["annotation"] = data["preferred_text"]
    return data


def annotate_randomly(data: pd.DataFrame, icai_results_dict: dict) -> pd.DataFrame:
    """Annotate the data randomly."""
    data["annotation"] = data["preferred_text"].apply(
        lambda x: random.choice(["text_a", "text_b"])
    )
    return data

import pathlib
import json
import ast
import pandas as pd
from loguru import logger


def load_json_file(path: str):
    with open(path, "r") as f:
        content = json.load(f)

    return content


def convert_vote_to_string(vote: bool | None) -> str:
    if vote is True:
        return "Agree"
    elif vote is False:
        return "Disagree"
    elif vote is None:
        return "Not applicable"
    elif vote == "invalid":
        return "Invalid"
    else:
        raise ValueError(f"Completely invalid vote value: {vote}")


def get_votes_df(results_dir: pathlib.Path, cache: dict) -> pd.DataFrame:
    """
    Get the votes dataframe for a given results directory.
    If the dataframe is already in the cache, return it.
    Otherwise, create it, add it to the cache, and return it.
    """

    if "votes_df" in cache and results_dir in cache["votes_df"]:
        return cache["votes_df"][results_dir]
    else:
        votes_df = create_votes_df(results_dir)

        if "votes_df" not in cache:
            cache["votes_df"] = {}
        cache["votes_df"][results_dir] = votes_df

        return votes_df


def create_votes_df(results_dir: pathlib.Path) -> list[dict]:

    # load relevant data from experiment logs
    votes_per_comparison = pd.read_csv(
        results_dir / "040_votes_per_comparison.csv", index_col="index"
    )
    principles_by_id: dict = load_json_file(
        results_dir / "030_distilled_principles_per_cluster.json",
    )
    comparison_df = pd.read_csv(results_dir / "000_train_data.csv", index_col="index")

    # merge original comparison data with votes per comparison
    full_df = comparison_df.merge(
        votes_per_comparison, left_index=True, right_index=True
    )
    full_df["comparison_id"] = full_df.index

    # add vote data column
    full_df["votes_dicts"] = full_df["votes"].apply(ast.literal_eval)

    def get_voting_data_columns(row):
        # votes are saved as dict with principle_id as key
        # and True, False, or None as value
        vote_dict = row["votes_dicts"]
        vote_principle_ids = list(vote_dict.keys())
        vote_principles = [principles_by_id[str(id)] for id in vote_principle_ids]
        vote_values = list(vote_dict.values())

        return vote_principle_ids, vote_principles, vote_values

    (
        full_df["principle_id"],
        full_df["principle"],
        full_df["vote"],
    ) = zip(*full_df.apply(get_voting_data_columns, axis=1))

    # explode into multiple rows
    # such that each row contains one principle and vote
    # (rather than one principle and a dict of multiple votes)
    full_df = full_df.explode(["principle_id", "principle", "vote"])

    # sanity check: make sure our length is correct
    if len(full_df) != len(comparison_df) * len(principles_by_id):
        votes_per_comparison = full_df.value_counts("comparison_id")
        max_votes = full_df.groupby("principle_id").size().max()
        min_votes = full_df.groupby("principle_id").size().min()
        logger.info(
            f"Note: not all principles have same number of votes (max: {max_votes} min: {min_votes}). This observation is not necessarily a problem, just indicating that ~{1 - min_votes/max_votes:.3f}% of votes may have been faulty."
        )

    # convert votes from True/False/None to strings
    full_df["vote"] = full_df["vote"].apply(convert_vote_to_string)

    # add a weight column
    full_df["weight"] = 1

    return full_df

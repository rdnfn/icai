"""Manager for intermediate files created during the inverse cai algorithm."""

import pathlib
import os
import pandas as pd

from loguru import logger

# file names
S1_PRINCIPLES = "s010_principles.txt"
S2_CLUSTERS = "s020_clusters.txt"
S3_VOTES = "s030_votes.txt"


def save_files(data: dict, path: str):
    """Save files to a directory."""

    folder = pathlib.Path(path)

    # check if the folder exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} not found.")

    # save the files
    s1_principles = folder / S1_PRINCIPLES
    s2_clusters = folder / S2_CLUSTERS
    s3_votes = folder / S3_VOTES

    data["s1_principles"].to_csv(s1_principles, index=False)
    data["s2_clusters"].to_csv(s2_clusters, index=False)
    data["s3_votes"].to_csv(s3_votes, index=False)

    logger.info(f"Saved files at {folder}.")


def load_files(path: str):
    """Load files from a directory."""

    folder = pathlib.Path(path)

    # check if the folder exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} not found.")

    # see if files exist
    s1_principles = folder / S1_PRINCIPLES
    s2_clusters = folder / S2_CLUSTERS
    s3_votes = folder / S3_VOTES

    if s1_principles.exists():
        s1_principles = pd.read_csv(s1_principles)
        logger.info(f"Loaded prior experiment data {S1_PRINCIPLES} at {s1_principles}.")

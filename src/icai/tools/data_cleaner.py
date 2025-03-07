"""Tool for cleaning data for publication

This CLI tool helps avoid accidentally leaking data that may lead
to training data contamination with test sets."""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import pandas as pd
from loguru import logger

CSVS_TO_CLEAN = ["000_train_data.csv"]
FILES_TO_COPY = [
    "030_distilled_principles_per_cluster.json",
    "040_votes_per_comparison.csv",
    "041_votes_per_cluster.json",
    "050_filtered_principles.json",
    "060_constitution.json",
]
COLS_TO_DELETE = []
COLS_TO_LIMIT = ["text_a", "text_b"]
NUM_SAMPLES = 10


def create_clean_csv(
    data_path: str,
    output_path: str,
    cols_to_delete: list[str],
    cols_to_limit: list[str],
    num_samples: int,
) -> None:
    """Create a clean CSV by deleting specified columns and limiting the number of samples.

    All other samples are set to "hidden" string.

    Args:
        data_path (str): The path to the input data CSV file.
        output_path (str): The path to save the cleaned CSV file.
        cols_to_delete (list[str]): The columns to delete.
        cols_to_limit (list[str]): The columns to limit.
        num_samples (int): The number of samples to limit to.
    """
    logger.info(f"Cleaning data from {data_path} and saving to {output_path}")
    logger.info(f"Deleting columns: {cols_to_delete}")
    logger.info(f"Limiting columns: {cols_to_limit} to {num_samples} samples")

    df = pd.read_csv(data_path)

    # Delete specified columns
    df = df.drop(columns=cols_to_delete)

    # Set the samples beyond the limit to "hidden" string
    for col in cols_to_limit:
        df.loc[df.index >= num_samples, col] = "hidden"

    # Save the cleaned CSV
    df.to_csv(output_path, index=False)


def create_cleaned_results_dir(
    input_dir: str,
    output_dir: str,
    num_samples: int,
    cols_to_delete: list[str] = COLS_TO_DELETE,
    cols_to_limit: list[str] = COLS_TO_LIMIT,
) -> None:
    """Create a cleaned results directory by copying all files from the results directory
    and setting all samples beyond the limit to "hidden" string.
    """

    logger.info(f"🛀 Cleaning data from {input_dir} and saving to {output_dir}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # subdir with result files
    input_results_dir = input_dir / "results"
    output_results_dir = output_dir / "results"

    # create output directory
    logger.info(f"Creating cleaned results directory in {output_results_dir}")
    if output_results_dir.parent.exists():
        logger.warning(
            f"Output directory {output_results_dir} already exists, aborting."
        )
        return

    output_results_dir.mkdir(parents=True, exist_ok=False)

    # copy files
    for file in FILES_TO_COPY:
        logger.info(f"Copying file {file} to {output_results_dir}")
        # check if file exists
        input_path = input_results_dir / file
        if not input_path.exists():
            logger.warning(f"File {file} does not exist in {input_results_dir}")
            continue

        # copy file
        output_path = output_results_dir / file
        shutil.copy(input_path, output_path)

    # clean CSVs
    for csv in CSVS_TO_CLEAN:
        create_clean_csv(
            data_path=str(input_results_dir / csv),
            output_path=str(output_results_dir / csv),
            cols_to_delete=cols_to_delete,
            cols_to_limit=cols_to_limit,
            num_samples=num_samples,
        )

    # copy hydra config
    hydra_config_path = input_dir / ".hydra" / "config.yaml"
    if hydra_config_path.exists():
        logger.info(f"Copying hydra config from {hydra_config_path} to {output_dir}")
        output_hydra_dir = output_dir / ".hydra"
        output_hydra_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(hydra_config_path, output_hydra_dir / "config.yaml")

    logger.info(f"Cleaned results directory saved to {output_dir}")
    logger.info(f"👋 Done!")


def run():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "-d",
        "--cols-to-delete",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-l",
        "--cols-to-limit",
        type=str,
        nargs="+",
        default=[],
    )
    args = parser.parse_args()
    create_cleaned_results_dir(**vars(args))

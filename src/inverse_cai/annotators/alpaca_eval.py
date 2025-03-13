"""Functions to annotate a pairwise output dataset using LLM annotators.

Based on AlpacaEval.
"""

import pathlib
import shutil
from importlib import resources
from loguru import logger
import pandas as pd
import alpaca_eval.main

from inverse_cai.experiment.config.main import ExpConfig

# get default annotators from alpacaeval_annotator_configs assets
DEFAULT_ANNOTATORS = {
    path.name: path
    for path in resources.files(
        "inverse_cai.assets.alpacaeval_annotator_configs"
    ).iterdir()
    if path.is_dir()
}


def generate_constitutional_annotator_configs(
    config: ExpConfig, tmp_files_path: str, constitution: str
):
    """Generate the constitutional annotator config."""

    base_const_configs = (
        config.annotator.alpaca_eval.base_constitutional_annotator_configs
    )

    tmp_config_paths = []

    for base_config in base_const_configs:
        tmp_base_config_path = generate_tmp_annotator_config(
            annotator_config_path=base_config,
            tmp_files_path=tmp_files_path,
            constitution=constitution,
        )

        if tmp_base_config_path in tmp_config_paths:
            raise ValueError(
                f"Config {tmp_base_config_path} already in tmp_config_paths"
            )

        tmp_config_paths.append(tmp_base_config_path)

    return tmp_config_paths


def generate_other_annotator_configs(config: ExpConfig, tmp_files_path: str):
    """Generate the other annotator configs.

    Created as separate tmp annotator configs (to avoid using old cached data) if
    the config is set to create tmp files,
    via config `config.annotator.create_other_annotator_tmp_configs`.
    """

    annotator_configs = config.annotator.alpaca_eval.other_annotator_configs

    tmp_config_paths = []

    if config.annotator.alpaca_eval.create_other_annotator_tmp_configs:
        for annotator_config in annotator_configs:
            tmp_annotator_config = generate_tmp_annotator_config(
                annotator_config_path=annotator_config,
                tmp_files_path=tmp_files_path,
            )
            tmp_config_paths.append(tmp_annotator_config)
        return tmp_config_paths
    else:
        return annotator_configs


def generate_tmp_annotator_config(
    annotator_config_path, tmp_files_path, constitution=None
) -> pathlib.Path:
    # check for error with path
    if annotator_config_path is None:
        raise ValueError("No config provided for the annotator, given None value.")
    annotator_config_path = pathlib.Path(annotator_config_path)
    if not annotator_config_path.exists():
        if str(annotator_config_path) == annotator_config_path.name:
            if annotator_config_path.name in DEFAULT_ANNOTATORS:
                annotator_config_path = DEFAULT_ANNOTATORS[annotator_config_path.name]
                logger.info(f"Using default annotator '{annotator_config_path.name}'")
            else:
                raise FileNotFoundError(
                    f"No default annotator with name '{annotator_config_path.name}'"
                    f" found in available default annotators: {list(DEFAULT_ANNOTATORS.keys())}."
                )

        else:
            raise FileNotFoundError(
                f"Annotator config path '{annotator_config_path}' does not exist. "
                f"To use default annotators just put their name (available configs: {list(DEFAULT_ANNOTATORS.keys())})"
            )
    if not annotator_config_path.is_dir():
        raise NotADirectoryError(
            f"Annotator config '{annotator_config_path}' is not a directory"
        )

    tmp_annotator_config_path = (
        pathlib.Path(tmp_files_path) / "annotator_configs" / annotator_config_path.name
    )
    if tmp_annotator_config_path.exists():
        raise FileExistsError(
            f"Annotator config {tmp_annotator_config_path} already exists"
        )
    # copy all files that are not json (completions)
    shutil.copytree(
        annotator_config_path,
        tmp_annotator_config_path,
        ignore=shutil.ignore_patterns("*.json"),
    )

    # add constitution to the prompt file
    if constitution is not None:
        prompt_file_path = tmp_annotator_config_path / "constitutional_prompt.txt"
        with open(prompt_file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()
        file_contents = file_contents.replace("{constitution}", constitution)
        with open(prompt_file_path, "w", encoding="utf-8") as file:
            file.write(file_contents)
        logger.info(
            f"Copied config and added constitution to {tmp_annotator_config_path}."
        )
    else:
        logger.info(
            f"Copied config (without constitution) to {tmp_annotator_config_path}."
        )
    return str(tmp_annotator_config_path.absolute())


def create_tmp_data_file(data: pd.DataFrame, tmp_files_path: str):
    """Create temporary data file of output data to annotate in AlpacaEval format."""

    # ensure tmp_files_path exists
    pathlib.Path(tmp_files_path).mkdir(parents=True, exist_ok=True)

    tmp_data_path = pathlib.Path(tmp_files_path) / "alpaca_style_data.json"

    # generating a data frame with columns: "instruction", "output_1",
    # "output_2", "preference" (one of 1 or 2)
    # from columns "text_a", "text_b", "preferred_text"

    alpaca_eval_data = pd.DataFrame(
        {
            "output_1": data["text_a"],
            "output_2": data["text_b"],
            "preference": data["preferred_text"],
        }
    )

    # add additional columns required by AlpacaEval
    alpaca_eval_data["instruction"] = ""
    alpaca_eval_data["annotator_index"] = 0
    alpaca_eval_data["dataset"] = "custom_dataset"
    alpaca_eval_data["datasplit"] = "eval"
    alpaca_eval_data["generator"] = None

    # update the preference column to be 1 or 2 instead of
    # the original labels (e.g. "text_a" or "text_b")
    alpaca_eval_data["preference"] = alpaca_eval_data["preference"].apply(
        lambda x: 1 if x == "text_a" else 2
    )

    # add random generator name: f"dummy_model_01" if odd, and f"dummy_model_02" if even
    alpaca_eval_data["generator"] = alpaca_eval_data.index.map(
        lambda x: f"dummy_model_01" if x % 2 == 1 else f"dummy_model_02"
    )

    # repeat each row 4 times to simulate 4 annotators
    alpaca_eval_data = alpaca_eval_data.loc[
        alpaca_eval_data.index.repeat(4)
    ].reset_index(drop=True)

    alpaca_eval_data.to_json(tmp_data_path, orient="records", indent=4)

    return tmp_data_path


def annotate(
    config: ExpConfig,
    data: pd.DataFrame,
    constitution: str,
    tmp_files_path: str,
    is_single_annotator: bool,
):
    """Annotate a dataset using LLM annotators."""

    annotator_configs = generate_other_annotator_configs(
        config=config, tmp_files_path=tmp_files_path
    )

    if constitution is not None:
        if constitution == "None":
            raise ValueError(
                "Constitution is set to string 'None'. This seems like a mistake?. "
                "If you want no constitution set, "
                "use null in command-line and yaml instead of Python's None."
            )
        elif constitution == "":
            logger.error("Constitution is an empty string. This seems like a mistake?")
        const_configs = generate_constitutional_annotator_configs(
            config=config,
            tmp_files_path=tmp_files_path,
            constitution=constitution,
        )
        logger.info(
            f"{len(const_configs)} constitutional annotator configs generated ({const_configs})"
        )
        annotator_configs.extend(const_configs)

    # transform data to AlpacaEval format
    tmp_data_path = create_tmp_data_file(data, tmp_files_path)
    logger.info(f"Data transformed to AlpacaEval format and saved to '{tmp_data_path}'")

    tmp_leaderboard_path = pathlib.Path(tmp_files_path) / "leaderboard.csv"

    evaluator_results = pd.DataFrame()
    for annotator_config in annotator_configs:
        logger.info(f"Annotating with annotator '{annotator_config}'")

        # ensure absolute path of annotator config is passed
        if "/" in annotator_config:
            annotator_config = str(pathlib.Path(annotator_config).absolute())
            default_annotator = False
        else:
            default_annotator = True

        annotator_name = (
            annotator_config.split("/")[-1]
            if not default_annotator
            else annotator_config + " (default)"
        )
        evaluator_leaderboard, all_crossannotations = (
            alpaca_eval.main.analyze_evaluators(
                annotators_config=annotator_config,
                is_return_instead_of_print=True,
                precomputed_leaderboard=tmp_leaderboard_path,
                is_single_annotator=is_single_annotator,
                analyzer_kwargs={
                    "gold_crossannotations": tmp_data_path.absolute(),
                    # if below is not set to None, default data used
                    "gold_annotations": None,
                },
                is_overwrite_leaderboard=True,
            )
        )
        evaluator_leaderboard.index = [annotator_name]
        evaluator_results = pd.concat([evaluator_results, evaluator_leaderboard])

    logger.info("All annotations completed.")

    # create standard annotator df from evaluator_results
    standard_results = evaluator_results.copy()
    standard_results.rename(columns={"Human agreement": "agreement"}, inplace=True)
    standard_results["annotator"] = standard_results.index
    standard_results = standard_results[["annotator", "agreement"]]
    standard_results.reset_index(drop=True, inplace=True)

    return standard_results, evaluator_results

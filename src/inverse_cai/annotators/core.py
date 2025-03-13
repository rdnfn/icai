"""Main module for annotators."""

import pathlib
import pandas as pd
from loguru import logger

import inverse_cai.annotators.alpaca_eval
from inverse_cai.experiment.config.main import ExpConfig


def annotate(
    cfg: ExpConfig,
    data: pd.DataFrame,
    constitution: list[str],
    tmp_path: pathlib.Path,
    test_data: pd.DataFrame | None,
    results_path: pathlib.Path,
):
    """Annotate the data using the annotator specified in the config.

    Args:
        cfg (ExpConfig): The experiment configuration.
        data (pd.DataFrame): The data to annotate.
        constitution (list[str]): The constitution to use for the annotation.
        tmp_path (pathlib.Path): The path to the temporary files.
        test_data (pd.DataFrame | None): The test data to annotate.
        results_path (pathlib.Path): The path to the results.
    """
    if not cfg.annotator.alpaca_eval.test_data_only:
        annotation_results = inverse_cai.annotators.alpaca_eval.annotate(
            config=cfg,
            data=data,
            constitution=constitution,
            is_single_annotator=cfg.annotator.alpaca_eval.is_single_annotator,
            tmp_files_path=tmp_path / "trainset",
        )

        logger.info(f"Results table (training data):\n{annotation_results}")
        annotation_results.to_csv(results_path / "092_results_training.csv")
    if test_data is not None:
        if isinstance(test_data, list):
            for i, test_data_single in enumerate(test_data):
                logger.info(f"Running LLM annotation on test data {i}/{len(test_data)}")
                test_annotation_results = inverse_cai.annotators.alpaca_eval.annotate(
                    config=cfg,
                    data=test_data_single,
                    constitution=constitution,
                    is_single_annotator=cfg.annotator.alpaca_eval.is_single_annotator,
                    tmp_files_path=tmp_path / f"testset_{i}",
                )
                logger.info(
                    f"Results table (test data {i}/{len(test_data)}):\n{test_annotation_results}"
                )
                test_annotation_results.to_csv(
                    results_path / f"093_results_testset_{i}.csv"
                )
        else:
            logger.info("Running LLM annotation on test data")
            test_annotation_results = inverse_cai.annotators.alpaca_eval.annotate(
                config=cfg,
                data=test_data,
                constitution=constitution,
                is_single_annotator=cfg.annotator.alpaca_eval.is_single_annotator,
                tmp_files_path=tmp_path / "testset",
            )
            logger.info(f"Results table (test data):\n{test_annotation_results}")
            test_annotation_results.to_csv(results_path / "093_results_testset.csv")
    else:
        if cfg.annotator.alpaca_eval.test_data_only:
            logger.warning(
                "No test data provided, but `test_data_only` is set to True. "
                "No test data will be annotated."
            )

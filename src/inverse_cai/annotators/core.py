"""Main module for annotators."""

import pathlib
import importlib
import pandas as pd
from loguru import logger

import inverse_cai.annotators.alpaca_eval
from inverse_cai.experiment.config.main import ExpConfig, FunctionAnnotatorConfig


def _import_annotator_function(annotator: FunctionAnnotatorConfig):
    """Helper function to import an annotator function based on config.

    Args:
        annotator: The annotator configuration object.

    Returns:
        The imported annotator function.

    Raises:
        ValueError: If the annotator function cannot be imported.
    """
    # Case 1: Explicit module is provided
    if annotator.function_module_to_import is not None:
        module = importlib.import_module(annotator.function_module_to_import)
        return getattr(module, annotator.function.split(".")[-1])

    # Case 2: Function path with module (contains dots)
    function_path = annotator.function
    if "." in function_path:
        module_path, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)

    # Case 3: No module info available
    raise ValueError(
        f"Cannot import annotator function: {function_path}. "
        f"Please specify function_module_to_import or use a fully qualified function name."
    )


def _evaluate_annotations(annotated_data: pd.DataFrame):
    """Evaluate the annotations, checking agreement between "preferred_text" and "annotation"."""
    agreement = (
        annotated_data["preferred_text"] == annotated_data["annotation"]
    ).mean()
    return agreement


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
        if not cfg.annotator.alpaca_eval.skip:
            annotation_results, alpaca_eval_results = (
                inverse_cai.annotators.alpaca_eval.annotate(
                    config=cfg,
                    data=data,
                    constitution=constitution,
                    is_single_annotator=cfg.annotator.alpaca_eval.is_single_annotator,
                    tmp_files_path=tmp_path / "trainset",
                )
            )

            logger.info(f"Results table (training data):\n{annotation_results}")
            alpaca_eval_results.to_csv(
                results_path / "092_full_alpacaeval_results_training.csv"
            )
        else:
            logger.info("Skipping AlpacaEval annotators")
            annotation_results = pd.DataFrame()

        if not cfg.annotator.fn_annotators:
            logger.info("No function annotators provided, skipping")
        else:
            num_fn_annotators = len(cfg.annotator.fn_annotators)
            logger.info(f"Running {num_fn_annotators} function annotators")
            for i, annotator in enumerate(cfg.annotator.fn_annotators):
                logger.info(
                    f"Running function annotator {i+1}/{num_fn_annotators}: {annotator}"
                )

                try:
                    annotator_fn = _import_annotator_function(annotator)

                    # Create results dictionary and call the annotator function
                    icai_results_dict = {"annotation_results": annotation_results}
                    annotated_data = annotator_fn(
                        data=data, icai_results_dict=icai_results_dict
                    )
                    agreement = _evaluate_annotations(annotated_data)
                    logger.info(f"Agreement for {annotator}: {agreement}")

                    # add row with annotator name and agreement to annotation results
                    new_row = pd.DataFrame(
                        {"annotator": [annotator], "agreement": [agreement]}
                    )
                    annotation_results = pd.concat(
                        [annotation_results, new_row], ignore_index=True
                    )
                except Exception as e:
                    logger.error(f"Failed to run annotator {annotator}: {e}")

        annotation_results.to_csv(results_path / "094_results_training.csv")
    if test_data is not None:
        # ensure we can iterate over the test data, even if it's a single dataframe
        if not isinstance(test_data, list):
            test_data = [test_data]

        for i, test_data_single in enumerate(test_data):
            logger.info(f"Running LLM annotation on test data {i}/{len(test_data)}")

            if not cfg.annotator.alpaca_eval.skip:
                test_annotation_results, alpaca_eval_results = (
                    inverse_cai.annotators.alpaca_eval.annotate(
                        config=cfg,
                        data=test_data_single,
                        constitution=constitution,
                        is_single_annotator=cfg.annotator.alpaca_eval.is_single_annotator,
                        tmp_files_path=tmp_path / f"testset_{i}",
                    )
                )
                logger.info(
                    f"Results table (test data {i}/{len(test_data)}):\n{test_annotation_results}"
                )
                alpaca_eval_results.to_csv(
                    results_path / f"093_full_alpacaeval_results_testset_{i}.csv"
                )
            else:
                logger.info("Skipping AlpacaEval annotators for test data")
                test_annotation_results = pd.DataFrame()

            test_annotation_results.to_csv(
                results_path / f"095_results_testset_{i}.csv"
            )
    else:
        if cfg.annotator.alpaca_eval.test_data_only:
            logger.warning(
                "No test data provided, but `test_data_only` is set to True. "
                "No test data will be annotated."
            )

"""Main module for annotators."""

import pathlib
import importlib
import pandas as pd
from loguru import logger

import inverse_cai.annotators.alpaca_eval
from inverse_cai.experiment.config.main import ExpConfig, FunctionAnnotatorConfig
from inverse_cai.data.annotated_pairs_format import (
    create_annotated_pairs,
    save_annotated_pairs_to_file,
    load_annotated_pairs_from_file,
    merge_annotated_pairs,
    DEFAULT_PREFERENCE_COLUMN,
)


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
    return agreement * 100


def _run_annotation_pipeline(
    cfg: ExpConfig,
    data: pd.DataFrame,
    ap_data: dict,
    icai_results_dict: dict,
    constitution: list[str],
    tmp_files_path: pathlib.Path,
    alpaca_results_csv_path: pathlib.Path,
    annotations_ap_path: pathlib.Path,
    final_results_csv_path: pathlib.Path,
    dataset_name: str,
    results_path: pathlib.Path,
):
    """Run the annotation pipeline on a dataset.

    Args:
        cfg: The experiment configuration.
        data: The data to annotate.
        constitution: The constitution to use for annotation.
        tmp_files_path: Path for temporary files.
        alpaca_results_csv_path: Path to save AlpacaEval results.
        annotations_ap_path: Path to save annotations in AnnotatedPairs json format.
        final_results_csv_path: Path to save final annotation results.
        dataset_name: Name of the dataset for logging purposes.

    Returns:
        DataFrame with annotation results.
    """
    # Initialize annotation results
    individual_annotations = pd.DataFrame()
    annotation_results = pd.DataFrame()

    # Run AlpacaEval if not skipped
    if not cfg.annotator.alpaca_eval.skip:
        annotation_results, alpaca_eval_results = (
            inverse_cai.annotators.alpaca_eval.annotate(
                config=cfg,
                data=data,
                constitution=constitution,
                is_single_annotator=cfg.annotator.alpaca_eval.is_single_annotator,
                tmp_files_path=tmp_files_path,
            )
        )
        annotation_results["annotator_type"] = "alpaca_eval"
        annotation_results["annotator_short_name"] = annotation_results["annotator"]
        logger.info(f"Results table ({dataset_name}):\n{annotation_results}")
        alpaca_eval_results.to_csv(alpaca_results_csv_path)
    else:
        logger.info(f"Skipping AlpacaEval annotators for {dataset_name}")

    # Run function annotators
    if not cfg.annotator.fn_annotators:
        logger.info(f"No function annotators provided for {dataset_name}, skipping")
    else:
        num_fn_annotators = len(cfg.annotator.fn_annotators)
        logger.info(
            f"Running {num_fn_annotators} function annotators on {dataset_name}"
        )

        for i, annotator in enumerate(cfg.annotator.fn_annotators):
            logger.info(
                f"Running function annotator {i+1}/{num_fn_annotators}: {annotator}"
            )

            try:
                annotator_fn = _import_annotator_function(annotator)
                annotator_kwargs = annotator.function_kwargs
                annotated_data = annotator_fn(
                    data=data,
                    ap_data=ap_data,
                    icai_results_dict=icai_results_dict,
                    results_path=results_path,
                    **annotator_kwargs,
                )
                annotator_short_name = annotator.function.split(".")[-1]
                agreement = _evaluate_annotations(annotated_data)

                # Add row with annotator name and agreement to annotation results
                new_row = pd.DataFrame(
                    {
                        "annotator": [annotator],
                        "agreement": [agreement],
                        "annotator_type": ["function"],
                        "annotator_short_name": [annotator_short_name],
                    }
                )
                annotation_results = pd.concat(
                    [annotation_results, new_row], ignore_index=True
                )

                if "text_a" not in individual_annotations.columns:
                    individual_annotations["text_a"] = data["text_a"]
                if "text_b" not in individual_annotations.columns:
                    individual_annotations["text_b"] = data["text_b"]
                if DEFAULT_PREFERENCE_COLUMN not in individual_annotations.columns:
                    individual_annotations[DEFAULT_PREFERENCE_COLUMN] = data[
                        DEFAULT_PREFERENCE_COLUMN
                    ]
                individual_annotations[annotator_short_name] = annotated_data[
                    "annotation"
                ]
            except Exception as e:
                logger.error(
                    f"Failed to run annotator {annotator} on {dataset_name}: {e}",
                    exc_info=True,
                )

    logger.info(
        f"Annotation results for dataset '{dataset_name}':\n{annotation_results}"
    )

    # save annotations of function annotators to AnnotatedPairs format
    # TODO: potentially support alpacaeval annotations here as well
    annotated_pairs = create_annotated_pairs(
        df=individual_annotations,
        dataset_name=f"Dataset with annotations by evaluated annotators - {dataset_name}",
        auto_detect_annotators=True,
    )
    if annotations_ap_path.exists():
        # merge with existing annotated pairs
        existing_annotated_pairs = load_annotated_pairs_from_file(annotations_ap_path)
        merged_metadata = existing_annotated_pairs["metadata"].copy()
        merged_metadata["description"] = (
            merged_metadata["description"]
            + "\n\nIncluding annotations by function annotators."
        )
        annotated_pairs = merge_annotated_pairs(
            [existing_annotated_pairs, annotated_pairs],
            merged_metadata=merged_metadata,
        )
    save_annotated_pairs_to_file(
        annotated_pairs,
        annotations_ap_path,
    )
    annotation_results.to_csv(final_results_csv_path, index=False)
    return annotation_results


def annotate(
    cfg: ExpConfig,
    data: pd.DataFrame,
    ap_data: dict,
    icai_results_dict: dict,
    constitution: list[str],
    tmp_path: pathlib.Path,
    test_data: list[pd.DataFrame] | None,
    test_ap_data: list[dict] | None,
    results_path: pathlib.Path,
):
    """Annotate the data using the annotator specified in the config.

    Args:
        cfg (ExpConfig): The experiment configuration.
        data (pd.DataFrame): The data to annotate.
        ap_data (list[dict]): Data to annotate in AnnotatedPairs format.
            This can be used to access per-principle votes by annotator.
        constitution (list[str]): The constitution to use for the annotation.
        tmp_path (pathlib.Path): The path to the temporary files.
        test_data (list[pd.DataFrame] | None): The test data to annotate.
        test_ap_data (list[dict] | None): Data to annotate in AnnotatedPairs format.
        results_path (pathlib.Path): The path to the results.
    """
    # Process training data if not skipped
    if not cfg.annotator.test_data_only:
        _run_annotation_pipeline(
            cfg=cfg,
            data=data,
            ap_data=ap_data,
            icai_results_dict=icai_results_dict,
            constitution=constitution,
            tmp_files_path=tmp_path / "trainset",
            alpaca_results_csv_path=results_path
            / "092_full_alpacaeval_results_training.csv",
            annotations_ap_path=results_path / "070_annotations_train_ap.json",
            final_results_csv_path=results_path / "094_results_training.csv",
            dataset_name="training data",
            results_path=results_path,
        )

    # Process test data if provided
    if test_data:
        if len(test_ap_data) == 0:
            logger.warning(
                "No AnnotatedPairs data available for test sets. Annotating without AP data."
            )
            test_ap_data = [None] * len(test_data)

        test_data_forms = zip(test_data, test_ap_data)

        for i, (test_df_single, test_ap_single) in enumerate(test_data_forms):
            _run_annotation_pipeline(
                cfg=cfg,
                data=test_df_single,
                ap_data=test_ap_single,
                icai_results_dict=icai_results_dict,
                constitution=constitution,
                tmp_files_path=tmp_path / f"testset-{i}",
                alpaca_results_csv_path=results_path
                / f"093_full_alpacaeval_results_testset-{i}.csv",
                annotations_ap_path=results_path
                / f"071_annotations_testset-{i}_ap.json",
                final_results_csv_path=results_path / f"095_results_testset-{i}.csv",
                dataset_name=f"test data {i+1}/{len(test_data)}",
                results_path=results_path,
            )
    elif cfg.annotator.test_data_only:
        logger.warning(
            "No test data provided, but `test_data_only` is set to True. "
            "No test data will be annotated."
        )

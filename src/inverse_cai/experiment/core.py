"""Script to run the Inverse Constitutional AI reconstruction experiment."""

from typing import Optional
import os
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from loguru import logger
import dotenv
import pathlib
import numpy as np

import inverse_cai
from inverse_cai.experiment.config.main import ExpConfig
import inverse_cai.annotators
from inverse_cai.data.annotated_pairs_format import (
    save_annotated_pairs_to_file,
    create_annotated_pairs,
)
import inverse_cai.data.loader.icai as icai_loader
from inverse_cai.algorithm.voting import get_votes_for_principles

cs = ConfigStore.instance()
cs.store(name="config", node=ExpConfig)


def setup_train_data(cfg: ExpConfig) -> pd.DataFrame:
    logger.info("Setting up training data")
    return setup_data(
        data_path=cfg.data_path,
        invert_labels=cfg.data_invert_labels,
        data_len=cfg.data_len,
        data_start_index=cfg.data_start_index,
        merge_prompts=cfg.data_merge_prompts,
    )


def setup_test_data(cfg: ExpConfig) -> pd.DataFrame:
    logger.info("Setting up test data")
    if cfg.test_data_path is None:
        logger.info(
            "No test data path specified. Only using training data for testing."
        )
        return []
    else:
        if isinstance(cfg.test_data_path, list):
            # Handle None values for test dataset settings
            test_data_len = cfg.test_data_len
            if test_data_len is None:
                test_data_len = [None] * len(cfg.test_data_path)
            elif not isinstance(test_data_len, list):
                test_data_len = [test_data_len] * len(cfg.test_data_path)

            test_data_invert_labels = cfg.test_data_invert_labels
            if test_data_invert_labels is None:
                test_data_invert_labels = [False] * len(cfg.test_data_path)
            elif not isinstance(test_data_invert_labels, list):
                test_data_invert_labels = [test_data_invert_labels] * len(
                    cfg.test_data_path
                )

            test_data_start_index = cfg.test_data_start_index
            if test_data_start_index is None:
                test_data_start_index = [0] * len(cfg.test_data_path)
            elif not isinstance(test_data_start_index, list):
                test_data_start_index = [test_data_start_index] * len(
                    cfg.test_data_path
                )

            return [
                setup_data(
                    data_path=path,
                    invert_labels=invert_labels,
                    data_len=data_len,
                    data_start_index=data_start_index,
                    merge_prompts=cfg.test_data_merge_prompts,
                )
                for path, data_len, invert_labels, data_start_index in zip(
                    cfg.test_data_path,
                    test_data_len,
                    test_data_invert_labels,
                    test_data_start_index,
                )
            ]
        elif isinstance(cfg.test_data_path, str):
            return [
                setup_data(
                    data_path=cfg.test_data_path,
                    invert_labels=(
                        cfg.test_data_invert_labels
                        if cfg.test_data_invert_labels is not None
                        else False
                    ),
                    data_len=cfg.test_data_len,
                    data_start_index=cfg.test_data_start_index,
                    merge_prompts=cfg.test_data_merge_prompts,
                )
            ]
        else:
            raise ValueError(
                f"test_data_path must be a string or a list of strings (given '{cfg.test_data_path}')"
            )


def setup_data(
    data_path: str,
    invert_labels: bool,
    data_len: Optional[int],
    data_start_index: Optional[int],
    merge_prompts: bool,
) -> pd.DataFrame:

    data = inverse_cai.data.loader.standard.load(
        data_path, switch_labels=invert_labels, merge_prompts=merge_prompts
    )

    # Limit the number of samples
    if data_len is None:
        logger.warning(f"No data_len specified. Using all data.")
        data_len = len(data)
    if data_len > len(data):
        raise ValueError(
            f"Requested data length {data_len} is "
            f"greater than the length of the data {len(data)}."
        )
    logger.info(f"Overall data length: {len(data)}")
    logger.info(f"Using data length: {data_len}")

    # Set default data_start_index if None
    if data_start_index is None:
        data_start_index = 0

    data = data.iloc[data_start_index : data_start_index + data_len]

    return data


def assert_no_identical_rows(df1: pd.DataFrame, df2: pd.DataFrame | list[pd.DataFrame]):

    if isinstance(df2, list):
        concatenated_df = pd.concat([df1, *df2])
    else:
        concatenated_df = pd.concat([df1, df2])
    unique_df = concatenated_df.drop_duplicates()

    # Check if the lengths are the same
    if len(concatenated_df) != len(unique_df):
        raise ValueError(
            "Identical rows found between the two test and train DataFrames."
        )
    else:
        logger.info("All good. No identical rows found between test and train splits.")


def add_loguru_to_hydra():
    # add logger
    # From https://github.com/facebookresearch/hydra/issues/2735#issuecomment-1774523324
    hydra_path = HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "main.log"))

    def api_only(record):
        return record["level"].name == "LM_API_CALL"

    log_format = '{{"time":"{time:YYYY-MM-DD HH:mm:ss}", "message":{message}}}'
    logger.add(
        os.path.join(
            hydra_path,
            "api_calls.jsonl",
        ),
        format=log_format,
        filter=api_only,
    )


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: DictConfig):
    """Run Inverse Constitutional AI reconstruction experiment."""

    add_loguru_to_hydra()
    hydra_out_path = HydraConfig.get().runtime.output_dir

    # setting up results dir inside hydra output dir
    results_path = pathlib.Path(hydra_out_path) / "results"
    tmp_path = pathlib.Path(hydra_out_path) / "tmp"
    for path in [results_path, tmp_path]:
        path.mkdir(parents=True, exist_ok=True)

    ap_paths = []

    # Enable running of cfg checks defined in __post_init__ method above.
    # (from https://github.com/facebookresearch/hydra/issues/981)
    cfg: ExpConfig = OmegaConf.to_object(cfg)

    # set up wandb logging
    if cfg.wandb_project is not None:
        if cfg.wandb_silent:
            os.environ["WANDB_SILENT"] = "true"
        logger.warning(
            "Logging to wandb is deprecated, but cfg.wandb_project argument passed. "
            "Please remove the argument."
        )

    if cfg.alg_model_cache:
        logger.warning(
            "Using cache for LLM in algorithm. Use `alg_model_cache=False` to disable. "
            "This makes the entire algorithm deterministic (as far as possible)."
        )
        langchain.globals.set_llm_cache(
            langchain.cache.SQLiteCache(database_path=".langchain.db")
        )
        np.random.seed(123)

    # Handle OpenRouter flag
    if cfg.alg_use_openrouter:
        # Only modify if not already an OpenRouter model
        if not cfg.alg_model.startswith("openrouter/"):
            orig_model = cfg.alg_model
            cfg.alg_model = f"openrouter/{orig_model}"
            logger.info(
                f"Using OpenRouter: Changed model from '{orig_model}' to '{cfg.alg_model}'"
            )

    logger.info("Starting experiment with config: \n" + OmegaConf.to_yaml(cfg))

    if cfg.annotator.alpaca_eval.constitution is not None and cfg.generate_constitution:
        raise ValueError(
            "A constitution was provided via `annotator.alpaca_eval.constitution`, "
            "but also `generate_constitution` is set to True. "
            "Please only set one of them."
        )

    # load secrets
    dotenv.load_dotenv(cfg.secrets_path, verbose=True)

    data = setup_train_data(cfg)
    has_preferred_text = "preferred_text" in data.columns
    added_pseudo_preferred_text = False
    if not has_preferred_text:
        logger.warning(
            "Data has no preferred_text column, this may cause issues for some functionalities. Adding pseudo preferred_text column that always selects the first text (text_a)."
        )
        data["preferred_text"] = "text_a"
        added_pseudo_preferred_text = True
    data.to_csv(results_path / "000_train_data.csv", index=True, index_label="index")
    test_data = setup_test_data(cfg)
    assert_no_identical_rows(data, test_data)
    assert isinstance(test_data, list)

    if test_data:
        for i, test_df in enumerate(test_data):
            test_df.to_csv(
                results_path / f"001_test_data_{i}.csv", index=True, index_label="index"
            )

    # TODO: remove this in a future version once s1_num_principles_per_instance is removed
    if cfg.s1_num_principles_per_instance is not None:
        logger.warning(
            "`s1_num_principles_per_instance` is set. This is deprecated and will be removed in a future version. "
            "Please use `s1_num_principles_per_sampling_step` instead. Overwriting `s1_num_principles_per_sampling_step`."
        )
        num_principles_per_sampling_step = cfg.s1_num_principles_per_instance
    else:
        num_principles_per_sampling_step = cfg.s1_num_principles_per_sampling_step

    if cfg.generate_constitution:
        results = inverse_cai.algorithm.run(
            save_path=results_path,
            feedback=data,
            num_principles_per_sampling_step=num_principles_per_sampling_step,
            num_rankings_per_sampling_step=cfg.s1_num_rankings_per_sampling_step,
            num_clusters=cfg.s2_num_clusters,
            random_clusters=cfg.s2_random_clusters,
            skip_voting=cfg.s3_skip_voting_entirely,
            require_majority_true=cfg.s3_filter_majority_true,
            require_majority_relevant=cfg.s3_filter_majority_relevant,
            require_majority_valid=cfg.s3_filter_majority_valid,
            require_minimum_relevance=cfg.s3_filter_min_relevance,
            ratio_of_max_principles_to_cluster_again=cfg.s3_ratio_of_max_principles_to_cluster_again,
            order_by=cfg.s3_order_by,
            max_principles=cfg.s3_max_principles,
            model_name=cfg.alg_model,
            config=cfg,
        )
        constitution = results["constitution"]

        # Generate annotated pairs format
        if results.get("raw_votes") is not None:
            ap_output_file = results_path / "070_annotations_train_ap.json"
            parsed_votes = icai_loader.parse_raw_votes(results["raw_votes"])
            parsed_prompt_votes = icai_loader.parse_raw_votes(
                results["raw_prompt_votes"]
            )
            train_annotated_pairs = create_annotated_pairs(
                df=data,
                principle_index_to_text=results["summaries"],
                comparison_votes=parsed_votes,
                nonpref_principle_index_to_text=results["prompt_summaries"],
                nonpref_comparison_votes=parsed_prompt_votes,
                dataset_name=f"ICAI Training Dataset - {pathlib.Path(hydra_out_path).name}",
                auto_detect_annotators=True,
            )
            save_annotated_pairs_to_file(train_annotated_pairs, ap_output_file)
            ap_paths.append(ap_output_file)
            logger.info(f"Generated annotated pairs format at {ap_output_file}")
    else:
        logger.warning(
            "Running LLM annotation on dataset without generating a new constitution"
        )
        if cfg.annotator.alpaca_eval.constitution is None:
            logger.error(
                "No constitution provided and `generate_constitution` is set to False. "
                "No constitution will be used in annotation. "
                "You may want to provide a constitution via `annotator.constitution`."
            )
        constitution = cfg.annotator.alpaca_eval.constitution

    test_ap_data = []
    if cfg.test_data_annotate_with_principles:
        for i, test_df in enumerate(test_data):
            test_annotation_cache_path = (
                results_path / "cache" / "02_principle_votes_testset"
            )

            logger.info("Annotating test data by principle-following annotators")
            raw_votes, _ = get_votes_for_principles(
                feedback_df=test_df,
                summaries=results["summaries"],
                max_votes_in_single_prompt=cfg.s3_filter_max_votes_in_single_prompt,
                model_name=cfg.alg_model,
                cache_path=test_annotation_cache_path,
                config=cfg,
                max_concurrent_tasks=cfg.async_task_num,
                num_seeds=cfg.s3_num_seeds_to_reannotate_with,
                voting_method_cross_seed=cfg.s3_voting_method_cross_seed,
            )
            raw_votes.to_csv(
                test_annotation_cache_path, index=True, index_label="index"
            )

            test_prompt_annotation_cache_path = (
                results_path / "048_prompt_votes_per_comparison_testset.csv"
            )

            logger.info("Annotating test data by prompt-principle-following annotators")
            raw_prompt_votes, _ = get_votes_for_principles(
                feedback_df=test_df,
                summaries=results["prompt_summaries"],
                max_votes_in_single_prompt=cfg.s3_filter_max_votes_in_single_prompt,
                model_name=cfg.alg_model,
                cache_path=test_annotation_cache_path,
                config=cfg,
                is_prompt_principles=True,
                max_concurrent_tasks=cfg.async_task_num,
                num_seeds=cfg.s3_num_seeds_to_reannotate_with,
                voting_method_cross_seed=cfg.s3_voting_method_cross_seed,
            )
            raw_prompt_votes.to_csv(
                test_prompt_annotation_cache_path, index=True, index_label="index"
            )

            parsed_votes = icai_loader.parse_raw_votes(raw_votes)
            parsed_prompt_votes = icai_loader.parse_raw_votes(raw_prompt_votes)
            test_annotated_pairs = create_annotated_pairs(
                df=test_df,
                principle_index_to_text=results["summaries"],
                comparison_votes=parsed_votes,
                nonpref_principle_index_to_text=results["prompt_summaries"],
                nonpref_comparison_votes=parsed_prompt_votes,
                dataset_name=f"ICAI Test Dataset - {pathlib.Path(hydra_out_path).name}",
                auto_detect_annotators=True,
            )
            ap_path = results_path / f"071_annotations_testset-{i}_ap.json"
            save_annotated_pairs_to_file(test_annotated_pairs, ap_path)
            ap_paths.append(ap_path)
            logger.info(f"Generated annotated pairs format at {ap_path}")
            test_ap_data.append(test_annotated_pairs)

    if cfg.annotator.skip:
        logger.warning("Skipping LLM annotation stage")
        if not cfg.generate_constitution:
            logger.error(
                "You have just done nothing. Neither a "
                "constitution was generated nor was the data annotated. "
                "Set `generate_constitution` to True or `annotator.skip` to False to do something."
            )
    else:
        logger.info("Running LLM annotation stage")

        inverse_cai.annotators.annotate(
            cfg=cfg,
            data=data,
            ap_data=train_annotated_pairs,
            icai_results_dict=results,
            test_data=test_data,
            test_ap_data=test_ap_data,
            constitution=constitution,
            tmp_path=tmp_path,
            results_path=results_path,
        )

    logger.warning(
        "Usage guidance: ICAI can only provide information about specific preference "
        "annotation datasets rather than annotators' reasoning processes more broadly. "
        "We recommend caution to avoid overinterpreting the results. Further, we "
        "recommend to manually inspect ICAI's interpretable constitutions before using "
        "them for downstream tasks to avoid accidentally amplifying harmful biases."
    )

    if added_pseudo_preferred_text:
        logger.warning(
            "Used synthetic preferred_text column. Since no ground-truth reference provided, "
            "the results should be interpreted with caution, e.g. constitutions and "
            "per-principle approval votes are based on synthetic data. Per-principle votes "
            "still make sense, as long as they are considered relative to the synthetic "
            "preferred_text column (always preferring text_a)."
        )

    logger.info(f"Experiment finished. Find results at {results_path}")
    ap_commands = []
    for ap_path in ap_paths:
        ap_commands.append(f"feedback-forensics -d {ap_path}")
    logger.info(
        "🔍 You can use Feedback Forensics to inspect the results "
        f"for the different datasets via the following commands: \n\n"
        + "\n\n".join(ap_commands)
        + "\n\nFollow the instructions in the Feedback Forensics repo to install it (https://github.com/rdnfn/feedback-forensics)."
    )
    logger.info("All done! ✨")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

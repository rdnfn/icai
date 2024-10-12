"""Script to run the Inverse Constitutional AI reconstruction experiment."""

from dataclasses import dataclass, MISSING, field, asdict
from typing import Optional, Dict, Any, Union
import os
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig, open_dict
import pandas as pd
from loguru import logger
import dotenv
import pathlib
import wandb
import numpy as np
import copy
import langchain.cache
import langchain.globals

import inverse_cai
from inverse_cai.experiment.config.main import ExpConfig


cs = ConfigStore.instance()
cs.store(name="config", node=ExpConfig)


def setup_train_data(cfg: ExpConfig) -> pd.DataFrame:
    logger.info("Setting up training data")
    return setup_data(
        data_path=cfg.data_path,
        invert_labels=cfg.data_invert_labels,
        data_len=cfg.data_len,
        data_start_index=cfg.data_start_index,
    )


def setup_test_data(cfg: ExpConfig) -> pd.DataFrame:
    logger.info("Setting up test data")
    if cfg.test_data_path is None:
        logger.info(
            "No test data path specified. Only using training data for testing."
        )
        return None
    else:
        return setup_data(
            data_path=cfg.test_data_path,
            invert_labels=cfg.test_data_invert_labels,
            data_len=cfg.test_data_len,
            data_start_index=cfg.test_data_start_index,
        )


def setup_data(
    data_path: str,
    invert_labels: bool,
    data_len: Optional[int],
    data_start_index: int,
) -> pd.DataFrame:

    data = inverse_cai.data.loader.standard.load(data_path, switch_labels=invert_labels)

    # Limit the number of samples
    if data_len is None:
        if len(data) < 100:
            logger.warning(
                "No data_len specified and the data is less than 100 samples. "
                "Using all data."
            )
            data_len = len(data)
        else:
            logger.warning(
                "No data_len specified, using the first 100 samples of the data."
            )
            data_len = 100
    if data_len > len(data):
        raise ValueError(
            f"Requested data length {data_len} is "
            f"greater than the length of the data {len(data)}."
        )
    logger.info(f"Overall data length: {len(data)}")
    logger.info(f"Using data length: {data_len}")
    data = data.iloc[data_start_index : data_start_index + data_len]

    return data


def assert_no_identical_rows(df1, df2):

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

    # Enable running of cfg checks defined in __post_init__ method above.
    # (from https://github.com/facebookresearch/hydra/issues/981)
    cfg: ExpConfig = OmegaConf.to_object(cfg)

    # set up wandb logging
    if cfg.wandb_project is not None:
        if cfg.wandb_silent:
            os.environ["WANDB_SILENT"] = "true"

        wandb_logging = True
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
        )
        logger.info("Logging to wandb")
        logger.info(f"Wandb run: {wandb.run.id}")
        logger.info(f"Wandb run page: {wandb.run.get_url()}")
    else:
        wandb_logging = False

    if cfg.alg_model_cache:
        logger.warning(
            "Using cache for LLM in algorithm. Use `alg_model_cache=False` to disable. "
            "This makes the entire algorithm deterministic (as far as possible)."
        )
        langchain.globals.set_llm_cache(
            langchain.cache.SQLiteCache(database_path=".langchain.db")
        )
        np.random.seed(123)

    logger.info("Starting experiment with config: \n" + OmegaConf.to_yaml(cfg))

    if cfg.annotator.constitution is not None and cfg.generate_constitution:
        raise ValueError(
            "A constitution was provided via `annotator.constitution`, "
            "but also `generate_constitution` is set to True. "
            "Please only set one of them."
        )

    # load secrets
    dotenv.load_dotenv(cfg.secrets_path, verbose=True)

    data = setup_train_data(cfg)
    data.to_csv(results_path / "000_train_data.csv", index=True, index_label="index")
    test_data = setup_test_data(cfg)
    assert_no_identical_rows(data, test_data)

    if cfg.generate_constitution:
        results = inverse_cai.algorithm.run(
            save_path=results_path,
            feedback=data,
            num_principles_generated_per_ranking=cfg.s1_num_principles_per_instance,
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
    else:
        logger.warning(
            "Running LLM annotation on dataset without generating a new constitution"
        )
        if cfg.annotator.constitution is None:
            logger.error(
                "No constitution provided and `generate_constitution` is set to False. "
                "No constitution will be used in annotation. "
                "You may want to provide a constitution via `annotator.constitution`."
            )
        constitution = cfg.annotator.constitution

    if cfg.annotator.skip:
        logger.warning("Skipping LLM annotation stage")
        annotation_results = pd.DataFrame()
        if not cfg.generate_constitution:
            logger.error(
                "You have just done nothing. Neither a "
                "constitution was generated nor was the data annotated. "
                "Set `generate_constitution` to True or `annotator.skip` to False to do something."
            )
    else:
        logger.info("Running LLM annotation stage")

        if not cfg.annotator.test_data_only:
            annotation_results = inverse_cai.annotator.annotate(
                config=cfg,
                data=data,
                constitution=constitution,
                is_single_annotator=cfg.annotator.is_single_annotator,
                tmp_files_path=tmp_path / "trainset",
            )

            logger.info(f"Results table (training data):\n{annotation_results}")
            annotation_results.to_csv(results_path / "092_results_training.csv")
        if test_data is not None:
            logger.info("Running LLM annotation on test data")
            test_annotation_results = inverse_cai.annotator.annotate(
                config=cfg,
                data=test_data,
                constitution=constitution,
                is_single_annotator=cfg.annotator.is_single_annotator,
                tmp_files_path=tmp_path / "testset",
            )
            logger.info(f"Results table (test data):\n{test_annotation_results}")
            test_annotation_results.to_csv(results_path / "093_results_testset.csv")
        else:
            if cfg.annotator.test_data_only:
                logger.warning(
                    "No test data provided, but `test_data_only` is set to True. "
                    "No test data will be annotated."
                )

    # TODO: add wandb logging of results

    logger.info(f"Experiment finished. Find results at {results_path}")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

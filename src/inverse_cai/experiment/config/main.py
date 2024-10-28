from dataclasses import dataclass, field
from typing import Optional, Any, Union
import os
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from loguru import logger

import inverse_cai
from inverse_cai.experiment.config.prompts import PromptConfig

UPDATED_HYDRA_DEFAULTS = {
    "job_logging": {
        # this prevents non-loguru loggers from showing up
        # (e.g. OpenAI API logs, which are very verbose and not needed for debugging)
        "disable_existing_loggers": True,
    },
    "run": {"dir": "exp/outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}"},
}


@dataclass
class AnnotatorConfig:
    """Configuration for the AI annotator."""

    skip: bool = False  # whether to skip the AI judgment stage
    create_other_annotator_tmp_configs: bool = (
        True  # whether to create tmp annotator configs
        # (to avoid accidentally using old cached data)
    )
    constitution: Optional[Union[str, bool]] = (
        None  # constitution to use for AI judgment
    )
    is_single_annotator: bool = False  # whether to use a single annotator (or four)
    base_constitutional_annotator_configs: list[str] = field(
        default_factory=lambda: [
            "data/annotator_configs/chatgpt_fn_constitutional_base_neutral_v1"
            # base annotator config to add constitution to
        ]
    )
    other_annotator_configs: list[str] = field(
        default_factory=lambda: [
            "data/annotator_configs/chatgpt_fn_noinstruction",  # annotators to test against
        ]
    )
    test_data_only: bool = False  # whether to only annotate the test data


@dataclass
class ExpConfig:
    """Main inverse constitutional AI experiment configuration."""

    # General config
    secrets_path: str = "./secrets.toml"  # Path to the secrets file
    parallel_workers: int = (
        -1
    )  # Number of parallel workers to use, -1 for all avaliable

    # logging config via wandb
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_silent: bool = True

    # Data config
    data_path: str = "./data/processed/example/example.csv"  # Path to the data file
    data_len: Optional[int] = None  # Number of samples to use for the experiment
    data_start_index: int = 0  # Index of the first sample to use
    data_invert_labels: bool = False  # Whether to invert the labels of the data

    # Test data config
    test_data_path: Optional[str] = None
    test_data_len: Optional[int] = None
    test_data_start_index: int = 0
    test_data_invert_labels: bool = False

    ## Algorithm config
    # general
    alg_model: str = "openai/gpt-3.5-turbo-0125"  # model to use for the algorithm
    alg_model_cache: bool = False  # whether to use the cache for the model
    generate_constitution: bool = (
        True  # whether to generate a constitution using the algorithm
    )

    # Stage 1: principle generation
    s1_num_principles_per_instance: int = 3

    # Stage 2: principle clustering and de-duplication
    s2_num_clusters: int = 3
    s2_random_clusters: bool = False

    # Stage 3: principle approximate voting
    s3_skip_voting_entirely: bool = False
    s3_filter_max_votes_in_single_prompt: int = 40
    s3_filter_majority_true: bool = True
    s3_filter_majority_relevant: bool = False
    s3_filter_majority_valid: bool = True
    s3_filter_min_relevance: Optional[float] = (
        0.1  # minimum proportional relevance (positive votes) to keep a principle
    )
    s3_order_by: str = "for_minus_against"
    s3_max_principles: Optional[int] = None
    s3_ratio_of_max_principles_to_cluster_again: float = (
        1.5  # proportion of max_principles to sample from filtered principle to then cluster again (until finally getting max principles)
    )

    # Stage 9: AI judgment
    annotator: Optional[AnnotatorConfig] = field(default_factory=AnnotatorConfig)

    # Prompts config
    alg_prompts: PromptConfig = field(default_factory=PromptConfig)

    # updateding Hydra defaults
    hydra: Any = field(default_factory=lambda: UPDATED_HYDRA_DEFAULTS)

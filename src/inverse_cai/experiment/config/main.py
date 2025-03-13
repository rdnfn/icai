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
class AlpacaEvalAnnotatorConfig:
    """Configuration for the AI annotator."""

    skip: bool = False  # whether to skip AlpacaEval annotators
    create_other_annotator_tmp_configs: bool = (
        True  # whether to create tmp annotator configs
        # (to avoid accidentally using old cached data)
    )
    constitution: Optional[Union[str, bool]] = (
        None  # constitution to use for AI judgment
    )
    is_single_annotator: bool = False  # whether to use a single annotator (or four)
    # for available default annotators, see inverse_cai/assets/alpacaeval_annotator_configs,
    # for these you can just use the name of the annotator, e.g.
    # "gpt4omini_fn_constitutional_base_neutral_v2"
    # for custom annotators, you can use the path to the annotator config file, e.g.
    # "data/annotator_configs/my_custom_annotator"
    base_constitutional_annotator_configs: list[str] = field(
        default_factory=lambda: [
            "gpt4omini_fn_constitutional_base_neutral_v2"
            # base annotator config to add constitution to
        ]
    )
    other_annotator_configs: list[str] = field(
        default_factory=lambda: [
            "alpaca_eval_gpt4omini_fn_noinstruction",
            # non-constitutionalannotators to test against
        ]
    )


@dataclass
class FunctionAnnotatorConfig:
    """Configuration for a annotator based on a Python function."""

    function: Optional[str] = (
        None  # string of the function path, e.g. "package.my_function"
    )
    function_module_to_import: Optional[str] = (
        None  # string of the module to import, e.g. "package", if None will use the function path as the module
    )
    function_kwargs: dict = field(default_factory=dict)


@dataclass
class AnnotatorConfig:
    """Configuration for the AI annotator."""

    alpaca_eval: AlpacaEvalAnnotatorConfig = field(
        default_factory=AlpacaEvalAnnotatorConfig
    )
    fn_annotators: list[FunctionAnnotatorConfig] = field(default_factory=list)
    skip: bool = False  # whether to skip the AI judgment stage entirely
    test_data_only: bool = False  # whether to only annotate the test data

    # DEPRECATED ALPACA EVAL CONFIG
    # TODO: remove these after v0.3.0
    create_other_annotator_tmp_configs: bool = True
    constitution: Optional[Union[str, bool]] = None
    is_single_annotator: bool = False
    base_constitutional_annotator_configs: list[str] = field(default_factory=lambda: [])
    other_annotator_configs: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        """Check if deprecated config values are being used."""
        self._check_deprecated_config()

    def _check_deprecated_config(self):
        """Check if deprecated configuration options are set to non-default values."""
        # Map of deprecated fields to their default values and replacements
        deprecated_fields = {
            "create_other_annotator_tmp_configs": (
                True,
                "alpaca_eval.create_other_annotator_tmp_configs",
            ),
            "constitution": (None, "alpaca_eval.constitution"),
            "is_single_annotator": (False, "alpaca_eval.is_single_annotator"),
            "base_constitutional_annotator_configs": (
                [],
                "alpaca_eval.base_constitutional_annotator_configs",
            ),
            "other_annotator_configs": ([], "alpaca_eval.other_annotator_configs"),
        }

        warnings = []

        # Check each deprecated field
        for field, (default_value, replacement) in deprecated_fields.items():
            current_value = getattr(self, field)

            # Check if this is a list that should be empty
            if isinstance(default_value, list) and current_value != default_value:
                warnings.append(f"{field} is deprecated, use {replacement} instead")
            # Check other types of values
            elif not isinstance(default_value, list) and current_value != default_value:
                warnings.append(
                    f"annotator.{field} is deprecated, use annotator.{replacement} instead."
                )

        # Log warnings if any deprecated config options are being used
        if warnings:
            for warning in warnings:
                logger.error(f"DEPRECATED CONFIG: {warning}")
            logger.error(
                "These deprecated config options do not have any effect anymore and will be removed in the future."
            )
            raise ValueError(
                "These deprecated config options found, please update your config file as suggested in Error log above."
            )


@dataclass
class ExpConfig:
    """Main inverse constitutional AI experiment configuration."""

    # General config
    secrets_path: str = "./secrets.toml"  # Path to the secrets file
    parallel_workers: int = (
        -1
    )  # Number of parallel workers to use, -1 for all avaliable

    # logging config via wandb (DEPRECATED, only here for compatibility)
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_silent: bool = True

    # Data config
    data_path: str = "./data/processed/example/example.csv"  # Path to the data file
    data_len: Optional[int] = None  # Number of samples to use for the experiment
    data_start_index: int = 0  # Index of the first sample to use
    data_invert_labels: bool = False  # Whether to invert the labels of the data

    # Test data config
    test_data_path: Any = None
    test_data_len: Any = None
    test_data_start_index: Any = 0
    test_data_invert_labels: Any = False

    # Cache config
    prior_cache_path: str | None = (
        None  # Path to the prior experiment results (path to main run directory)
    )

    ## Algorithm config
    # general
    alg_model: str = "openai/gpt-4o-mini-2024-07-18"  # model to use for the algorithm
    alg_model_cache: bool = False  # whether to use the cache for the model
    generate_constitution: bool = (
        True  # whether to generate a constitution using the algorithm
    )

    # Optional: set principles to test (skip generation)
    s0_added_principles_to_test: list[str] | None = None
    s0_added_standard_principles_to_test: list[str] | None = (
        None  # version number of standard principles to test, e.g. "v1" or "v2"
    )
    s0_skip_principle_generation: bool = False

    # Stage 1: principle generation
    s1_num_principles_per_sampling_step: int = 3
    s1_num_principles_per_instance: int | None = None
    s1_num_rankings_per_sampling_step: int = 1

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

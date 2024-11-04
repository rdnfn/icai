"""Module with configurations for built-in datasets."""

from dataclasses import dataclass, field
import gradio as gr
from loguru import logger

from inverse_cai.app.constants import NONE_SELECTED_VALUE
from inverse_cai.app.data_loader import load_icai_data, DATA_DIR

load_icai_data()

DATA_DIR = DATA_DIR / "icai-data"


@dataclass
class BuiltinDataset:
    """Class to represent a built-in dataset."""

    name: str
    path: str | None = None
    description: str | None = None
    options: list | None = None
    filterable_columns: list[str] | None = None


@dataclass
class Config:
    """Class to represent a configuration."""

    name: str
    show_individual_prefs: bool = False
    pref_order: str = "By reconstruction success"
    filter_col: str = NONE_SELECTED_VALUE
    filter_value: str = NONE_SELECTED_VALUE
    filter_col_2: str = NONE_SELECTED_VALUE
    filter_value_2: str = NONE_SELECTED_VALUE
    metrics: list = field(default_factory=lambda: ["perf", "relevance", "acc"])


# Builtin datasets

SYNTHETIC = BuiltinDataset(
    name="🧪 Synthetic",
    path=DATA_DIR / "synthetic_v1",
    description="Synthetic dataset generated according to three different rules.",
    options=None,
)

CHATBOT_ARENA = BuiltinDataset(
    name="🏟️ Chatbot Arena",
    path=DATA_DIR / "chatbot_arena_v1",
    description="LMSYS Chatbot Arena data.",
    options=[
        Config(
            name="GPT-4-1106-preview winning (against all other models)",
            filter_col="winner_model",
            filter_value="gpt-4-1106-preview",
        ),
        Config(
            name="GPT-4-1106-preview winning against GPT-4-0314",
            filter_col="winner_model",
            filter_value="gpt-4-1106-preview",
            filter_col_2="loser_model",
            filter_value_2="gpt-4-0314",
        ),
        Config(
            name="GPT-4-1106-preview losing to GPT-4-0314",
            filter_col="loser_model",
            filter_value="gpt-4-1106-preview",
            filter_col_2="winner_model",
            filter_value_2="gpt-4-0314",
        ),
    ],
)

PRISM = BuiltinDataset(
    name="💎 PRISM",
    path=DATA_DIR / "prism_1k_v1",
    description="PRISM dataset.",
    filterable_columns=["chosen_model", "location_birth_region", "english_proficiency"],
    options=[
        Config(
            name="GPT-4-1106-preview winning (against all other models)",
            filter_col="chosen_model",
            filter_value="gpt-4-1106-preview",
        ),
        Config(
            name="Location (by birth region): Americas",
            filter_col="location_birth_region",
            filter_value="Americas",
            metrics=["perf", "relevance", "acc", "perf_diff", "perf_base"],
        ),
        Config(
            name="Location (by birth region): Europe",
            filter_col="location_birth_region",
            filter_value="Europe",
            metrics=["perf", "relevance", "acc", "perf_diff", "perf_base"],
        ),
        Config(
            name="English proficiency: intermediate",
            filter_col="english_proficiency",
            filter_value="Intermediate",
            metrics=["perf", "relevance", "acc", "perf_diff", "perf_base"],
        ),
        Config(
            name="English proficiency: native speaker",
            filter_col="english_proficiency",
            filter_value="Native speaker",
            metrics=["perf", "relevance", "acc", "perf_diff", "perf_base"],
        ),
    ],
)

ALPACA_EVAL = BuiltinDataset(
    name="🦙 AlpacaEval",
    path=DATA_DIR / "alpacaeval_v1",
    description="AlpacaEval cross-annotated dataset of 648 pairwise comparisons. Each comparison is rated by 4 human annotators. We use the majority vote as the ground truth, breaking ties randomly.",
)

# List of all built-in datasets
BUILTIN_DATASETS = [SYNTHETIC, CHATBOT_ARENA, PRISM, ALPACA_EVAL]

# make sure entire dataset is an option for all built-in datasets
for dataset in BUILTIN_DATASETS:
    if dataset.options is not None:
        dataset.options = dataset.options + [Config("Entire dataset")]
    else:
        dataset.options = [Config("Entire dataset")]


# utility functions
def get_config_from_name(name: str, config_options: list) -> Config:
    """Get a configuration from its name."""
    if name == NONE_SELECTED_VALUE or name is None:  # default config
        return Config(name=name)

    for config in config_options:
        if config.name == name:
            return config

    raise ValueError(f"Configuration with name '{name}' not found.")


def get_dataset_from_name(name: str) -> BuiltinDataset:
    """Get a dataset from its name."""
    for dataset in BUILTIN_DATASETS:
        if dataset.name == name:
            logger.info(f"Loading dataset '{name}'", duration=5)
            return dataset

    raise ValueError(f"Dataset with name '{name}' not found.")

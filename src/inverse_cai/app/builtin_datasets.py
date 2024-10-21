"""Module with configurations for built-in datasets."""

from dataclasses import dataclass

from inverse_cai.app.constants import NONE_SELECTED_VALUE


@dataclass
class BuiltinDataset:
    """Class to represent a built-in dataset."""

    name: str
    path: str | None = None
    description: str | None = None
    options: list | None = None


@dataclass
class Config:
    """Class to represent a configuration."""

    name: str
    efficent_mode: bool = True
    pref_order: str = "By reconstruction success"
    filter_col: str = NONE_SELECTED_VALUE
    filter_value: str = NONE_SELECTED_VALUE
    filter_col_2: str = NONE_SELECTED_VALUE
    filter_value_2: str = NONE_SELECTED_VALUE


# Builtin datasets

SYNTHETIC = BuiltinDataset(
    name="🧪 Synthetic",
    path="exp/outputs/2024-10-12_15-58-26",
    description="Synthetic dataset generated according to three different rules.",
    options=[
        Config(
            name="rule1",
        ),
    ],
)

CHATBOT_ARENA = BuiltinDataset(name="🏟️ Chatbot Arena")

PRISM = BuiltinDataset(name="💎 PRISM")

# List of all built-in datasets
BUILTIN_DATASETS = [SYNTHETIC, CHATBOT_ARENA, PRISM]


# utility functions
def get_config_from_name(name: str, config_options: list) -> Config:
    """Get a configuration from its name."""
    for config in config_options:
        if config.name == name:
            return config

    raise ValueError(f"Configuration with name '{name}' not found.")


def get_dataset_from_name(name: str) -> BuiltinDataset:
    """Get a dataset from its name."""
    for dataset in BUILTIN_DATASETS:
        if dataset.name == name:
            return dataset

    raise ValueError(f"Dataset with name '{name}' not found.")

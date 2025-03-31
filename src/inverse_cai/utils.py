"""Utility functions for inverse_cai module."""

import pathlib
import sys
import json
from loguru import logger

EXPERIMENT_DIR = pathlib.Path("./icai_exp")


def setup():
    """Create the experiment directory."""
    EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)


def setup_logger(include_time: bool = False, log_level: str = "INFO") -> None:
    """
    Set up the logger with the specified log level and format.
    Args:
        include_time (bool, optional): Whether to include the timestamp in the log messages. Defaults to False.
        log_level (str, optional): The log level to set for the logger. Defaults to "INFO".
    Returns:
        None
    """

    logger.remove()

    if include_time:
        time_str = "<light-black>[{time:YYYY-MM-DD, HH:mm:ss.SSSS}]</light-black> "
    else:
        time_str = ""

    log_format = f"📜 {time_str} | " "{level: <4} | <level>{message}</level>"

    logger.level("INFO", color="")
    logger.add(
        sys.stdout,
        colorize=True,
        format=log_format,
        level=log_level,
    )
    logger.level("LM_API_CALL", no=15, color="<yellow>", icon="📢")


def save_to_json(data, path):
    """
    Save data to JSON.
    """
    json.dump(data, open(path, "w", encoding="utf-8"), indent=4)


setup_logger(include_time=False, log_level="INFO")

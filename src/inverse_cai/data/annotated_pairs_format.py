"""Functionality for working with the annotated pairs format.

This module provides tools for converting ICAI experiment results to the
annotated pairs format, a standardized JSON format for representing model
comparisons with annotations from both human evaluators and principles.
"""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd
from loguru import logger

from inverse_cai.data.loader import icai

# Constants
DEFAULT_ANNOTATOR_DESCRIPTION = (
    "Default annotator from original dataset (from column `preferred_text`)"
)
FORMAT_VERSION = "1.0"
DEFAULT_ANNOTATOR_TYPE = "unknown"
DEFAULT_PREFERENCE_KEY = "pref"
DEFAULT_PREFERENCE_COLUMN = "preferred_text"


def hash_string(s: str) -> str:
    """Create a shortened hash of a string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def hash_comparison(text_a: str, text_b: str, prompt: Optional[str]) -> str:
    """Create a hash ID for a comparison based on its content."""
    combined = f"{text_a}|{text_b}|"
    if prompt is not None:
        combined = f"{prompt}|{combined}"
    return hash_string(combined)


def votes_to_annotations(
    votes: Mapping[int, Optional[bool]],
    principle_index_to_text: Mapping[int, str],
    active_principles: Sequence[str],
    reference_preference: str,
) -> Dict[str, Dict[str, str]]:
    """Convert principle votes to annotations in the standardized format.

    Args:
        votes: Dictionary mapping principle IDs to their votes (True/False/None)
        principle_index_to_text: Dictionary mapping principle IDs to their text
        active_principles: List of active principles to include
        reference_preference: The reference preference (text_a or text_b)

    Returns:
        Dictionary mapping principle IDs (hashed) to dictionaries with preferences
    """
    annotations = {}

    for principle_idx, vote in votes.items():
        principle_text = principle_index_to_text[principle_idx]

        # Only include principles that are active
        if principle_text in active_principles:
            principle_id = hash_string(principle_text)

            # Convert vote to text_a, text_b or not_applicable
            if vote is None:
                annotations[principle_id] = {DEFAULT_PREFERENCE_KEY: "not_applicable"}
            elif vote is True:
                # Principle agrees with reference preference
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: reference_preference
                }
            else:  # vote is False
                # Principle disagrees with reference preference
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: (
                        "text_b" if reference_preference == "text_a" else "text_a"
                    )
                }

    return annotations


def add_annotators(
    output: Dict,
    principles: Mapping[int, str],
    filtered_principles: Sequence[str],
    filter_to_constitution: bool = True,
    additional_columns: List[str] = None,
) -> None:
    """Add all annotators to the output structure.

    This function modifies the output dictionary in-place by adding annotator
    information to output["annotators"] and setting the default annotator.

    Args:
        output: The output dataset dictionary to modify in-place
        principles: Dictionary of principles where keys are principle IDs
        filtered_principles: List of filtered principles
        filter_to_constitution: Only include principles that made it to the constitution
        additional_columns: List of additional columns from the training data to include as annotations
    """
    # Create default annotator
    default_annotator_id = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)
    output["annotators"][default_annotator_id] = {
        "name": "Default",
        "description": DEFAULT_ANNOTATOR_DESCRIPTION,
        "type": DEFAULT_ANNOTATOR_TYPE,
    }
    output["metadata"]["default_annotator"] = default_annotator_id

    # Determine active principles
    active_principles = (
        filtered_principles if filter_to_constitution else list(principles.values())
    )

    # Create principle annotators
    for principle in active_principles:
        annotator_id = hash_string(principle)
        output["annotators"][annotator_id] = {
            "description": principle,
            "type": "principle",
        }

    # Create column annotators if additional columns are specified
    if additional_columns:
        for col in additional_columns:
            column_annotator_id = hash_string(f"column_{col}")
            # Set type to "human" if "human" is in the column name, otherwise use DEFAULT_ANNOTATOR_TYPE
            annotator_type = "human" if "human" in col.lower() else "unknown"
            output["annotators"][column_annotator_id] = {
                "name": col,
                "description": f"Column from original dataset: {col}",
                "type": annotator_type,
            }


def detect_annotator_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that appear to be annotator columns in the DataFrame.

    This function looks for columns that:
    1. Contain boolean values or values that can be converted to text_a/text_b
    2. Have a reasonable number of non-null values
    3. Are not standard columns (text_a, text_b, input, etc.)

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names that appear to be annotator columns
    """
    standard_columns = {
        "text_a",
        "text_b",
        "model_a",
        "model_b",
        DEFAULT_PREFERENCE_COLUMN,
    }
    potential_annotators = []

    for col in df.columns:
        if col in standard_columns:
            continue

        # Check if column contains boolean values or values that can be converted to text_a/text_b
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= 10:  # Allow for None/NA values
            # Check if values can be interpreted as preferences
            values_set = set(str(v).lower() for v in unique_values if pd.notna(v))

            # Simply check if text_a and text_b are present
            if "text_a" in values_set and "text_b" in values_set:
                potential_annotators.append(col)

    return potential_annotators


def create_annotated_pairs(
    train_df: pd.DataFrame,
    principles: Mapping[int, str],
    filtered_principles: Sequence[str],
    comparison_votes: Mapping[int, Dict[int, Optional[bool]]],
    dataset_name: str,
    filter_to_constitution: bool = True,
    additional_columns: List[str] = None,
    auto_detect_annotators: bool = True,
) -> Dict:
    """Convert ICAI results to annotated pairs format using direct data inputs.

    Args:
        train_df: DataFrame with training data. Must have mandatory "text_a", "text_b", and DEFAULT_PREFERENCE_COLUMN rows, and an optional "input" (prompt).
        principles: Dictionary of principles where keys are principle IDs
        filtered_principles: List of filtered principles (those that made it to the constitution)
        comparison_votes: Dictionary of comparison votes
        dataset_name: Name for the dataset
        filter_to_constitution: Only include principles that made it to the constitution
        additional_columns: List of additional columns from the training data to include as annotations
        auto_detect_annotators: Whether to automatically detect annotator columns in the DataFrame

    Returns:
        The annotated pairs format as a dictionary
    """
    # Initialize the output structure
    output = {
        "metadata": {
            "version": FORMAT_VERSION,
            "description": "Annotated pairs dataset with annotations from ICAI",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_name": dataset_name,
        },
        "annotators": {},
        "comparisons": [],
    }

    # Detect annotator columns if enabled
    detected_columns = []
    if auto_detect_annotators:
        detected_columns = detect_annotator_columns(train_df)
        if detected_columns:
            logger.info(f"Automatically detected annotator columns: {detected_columns}")

    # Combine detected columns with manually specified ones
    all_additional_columns = list(set((additional_columns or []) + detected_columns))

    # Add all annotators to the output
    add_annotators(
        output,
        principles,
        filtered_principles,
        filter_to_constitution,
        all_additional_columns,
    )

    # Identify metadata columns (columns that are not standard columns, not annotator columns, and not used for comparison content)
    standard_columns = {"text_a", "text_b", DEFAULT_PREFERENCE_COLUMN}
    metadata_columns = [
        col
        for col in train_df.columns
        if col not in standard_columns and col not in all_additional_columns
    ]

    # Add metadata columns to the overall metadata
    if metadata_columns:
        output["metadata"]["available_metadata_keys_per_comparison"] = metadata_columns
        logger.info(f"Available metadata columns: {metadata_columns}")

    # Prepare data needed for annotations
    default_annotator_id = output["metadata"]["default_annotator"]
    active_principles = (
        filtered_principles if filter_to_constitution else list(principles.values())
    )

    # Process each comparison
    for idx, row in train_df.iterrows():
        # Create unique ID for this comparison
        comparison_id = hash_comparison(row["text_a"], row["text_b"], row.get("input"))

        # Initialize annotations dict with default annotator annotation
        annotations = {}
        reference_preference = row[DEFAULT_PREFERENCE_COLUMN]
        annotations[default_annotator_id] = {
            DEFAULT_PREFERENCE_KEY: reference_preference
        }

        # Add principle annotations based on votes
        if idx in comparison_votes:
            votes = comparison_votes[idx]
            principle_annotations = votes_to_annotations(
                votes, principles, active_principles, reference_preference
            )
            annotations.update(principle_annotations)
        else:
            logger.warning(
                f"Missing votes for comparison with index {idx}, skipping principle annotations"
            )

        # Add additional columns as annotations if specified
        if all_additional_columns:
            for col in all_additional_columns:
                if col in row and pd.notna(row[col]):
                    # Create a unique ID for this column annotator
                    column_annotator_id = hash_string(f"column_{col}")

                    # Add the annotation
                    annotations[column_annotator_id] = {"value": str(row[col])}

        # Create the comparison entry
        comparison = {
            "id": comparison_id,
            "prompt": row.get("input"),
            "text_a": row["text_a"],
            "text_b": row["text_b"],
            "annotations": annotations,
        }

        # Add all metadata columns to the comparison metadata
        if metadata_columns:
            comparison["metadata"] = {}
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    comparison["metadata"][col] = str(row[col])

        output["comparisons"].append(comparison)

    return output


def results_to_annotated_pairs(
    results_dir: str,
    dataset_name: str,
    filter_to_constitution: bool = True,
    additional_columns: List[str] = None,
    auto_detect_annotators: bool = True,
) -> Dict[str, object]:
    """Convert ICAI results to annotated pairs format from files.

    Args:
        results_dir: Path to ICAI results directory
        dataset_name: Name for the dataset
        filter_to_constitution: Only include principles that made it to the constitution
        additional_columns: List of additional columns from the training data to include as annotations
        auto_detect_annotators: Whether to automatically detect annotator columns in the DataFrame

    Returns:
        The annotated pairs format as a dictionary
    """
    results_path = Path(results_dir)

    # Load all required data using the icai loader module
    train_df = icai.load_train_data(results_path)
    principles = icai.load_principles(results_path)
    filtered_principles = icai.load_filtered_principles(results_path)
    comparison_votes = icai.load_votes_per_comparison(results_path)

    # Call the core implementation with loaded data
    result = create_annotated_pairs(
        train_df=train_df,
        principles=principles,
        filtered_principles=filtered_principles,
        comparison_votes=comparison_votes,
        dataset_name=dataset_name,
        filter_to_constitution=filter_to_constitution,
        additional_columns=additional_columns,
        auto_detect_annotators=auto_detect_annotators,
    )

    return result


def save_annotated_pairs_to_file(annotated_pairs: Dict, output_file: str) -> None:
    """Save the annotated pairs to a JSON file.

    Args:
        annotated_pairs: The annotated pairs dataset to save
        output_file: Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotated_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Created annotated pairs format dataset: {output_file}")
    logger.info(f"- Dataset contains {len(annotated_pairs['comparisons'])} comparisons")
    logger.info(f"- Dataset contains {len(annotated_pairs['annotators'])} annotators")

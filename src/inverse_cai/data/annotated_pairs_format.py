"""Functionality for working with the annotated pairs format.

This module provides tools for converting ICAI experiment results to the
annotated pairs format, a standardized JSON format for representing model
comparisons with annotations from both human evaluators and principles.
"""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union, Any

import pandas as pd
from loguru import logger

from inverse_cai.data.loader import icai

# Constants
DEFAULT_ANNOTATOR_DESCRIPTION = (
    "Default annotator from original dataset (from column `preferred_text`)"
)
FORMAT_VERSION = "2.0"
DEFAULT_ANNOTATOR_TYPE = "unknown"
DEFAULT_PREFERENCE_KEY = "pref"
DEFAULT_PREFERENCE_COLUMN = "preferred_text"

PREFERENCE_ALIAS_MAPPING = {
    "a": "a",
    "b": "b",
    "text_a": "a",
    "text_b": "b",
}


def hash_string(s: str) -> str:
    """Create a shortened hash of a string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def hash_comparison(
    response_a: Dict[str, Any], response_b: Dict[str, Any], prompt: Optional[str]
) -> str:
    """Create a hash ID for a comparison based on its content.

    Args:
        response_a: A dictionary with at least a 'text' key
        response_b: A dictionary with at least a 'text' key
        prompt: An optional prompt string
    """
    # Serialize each response dictionary to JSON to include all fields
    response_a_str = json.dumps(response_a, sort_keys=True)
    response_b_str = json.dumps(response_b, sort_keys=True)

    combined = f"{response_a_str}|{response_b_str}"
    if prompt is not None:
        combined = f"{prompt}|{combined}"
    return hash_string(combined)


def votes_to_annotations(
    votes: Mapping[int, Union[bool, str, None]],
    principle_index_to_text: Mapping[int, str],
    active_principles: Sequence[str],
    reference_preference: str,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Convert principle votes to annotations in the standardized format.

    Args:
        votes: Dictionary mapping principle IDs to their votes:
            - True: Principle agrees with reference preference
            - False: Principle disagrees with reference preference
            - None: Principle not applicable to this comparison
            - "invalid": Vote is invalid for technical reasons
        principle_index_to_text: Dictionary mapping principle IDs to their text
        active_principles: List of active principles to include
        reference_preference: The reference preference ("a" or "b")

    Returns:
        Dictionary mapping principle IDs (hashed) to dictionaries with preference information:
        - For True/False votes: {"pref": "a" or "b"}
        - For None votes: {"pref": None, "no_pref_reason": "not_applicable"}
        - For "invalid" votes: {"pref": None, "no_pref_reason": "invalid"}
    """
    annotations = {}

    for principle_idx, vote in votes.items():
        principle_text = principle_index_to_text[principle_idx]

        # Only include principles that are active
        if principle_text in active_principles:
            principle_id = hash_string(principle_text)

            # Convert vote to a, b, or None with reason
            if vote is None:
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: None,
                    "no_pref_reason": "not_applicable",
                }
            elif vote is True:
                # Principle agrees with reference preference
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: reference_preference
                }
            elif vote is False:
                # Principle disagrees with reference preference
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: (
                        "b" if reference_preference == "a" else "a"
                    )
                }
            elif vote == "invalid":
                # Special case for invalid votes
                annotations[principle_id] = {
                    DEFAULT_PREFERENCE_KEY: None,
                    "no_pref_reason": "invalid",
                }
            else:
                raise ValueError(
                    f"Unexpected vote value: {vote} (type: {type(vote)}) for principle {principle_text}.\n\nTaken from votes: {votes} (type: {type(votes)})"
                )

    return annotations


def add_annotators(
    output: Dict,
    principles: Mapping[int, str] | None = None,
    additional_columns: List[str] = None,
) -> None:
    """Add all annotators to the output structure.

    This function modifies the output dictionary in-place by adding annotator
    information to output["annotators"] and setting the default annotator.

    Args:
        output (Dict): The output dataset dictionary to modify in-place
        principles (Mapping[int, str] | None): Dictionary of principles where keys are principle IDs
        additional_columns (List[str] | None): List of additional columns from the training data to include as annotations
    """
    # Create default annotator
    default_annotator_id = hash_string(DEFAULT_ANNOTATOR_DESCRIPTION)
    output["annotators"][default_annotator_id] = {
        "name": DEFAULT_PREFERENCE_COLUMN,
        "description": DEFAULT_ANNOTATOR_DESCRIPTION,
        "type": DEFAULT_ANNOTATOR_TYPE,
    }
    output["metadata"]["default_annotator"] = default_annotator_id

    # Determine active principles
    if principles is not None:
        active_principles = list(principles.values())
    else:
        active_principles = []

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
    1. Contain boolean values or values that can be converted to a/b
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

        # Check if column contains boolean values or values that can be converted to a/b
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= 10:  # Allow for None/NA values
            # Check if values can be interpreted as preferences
            values_set = set(str(v).lower() for v in unique_values if pd.notna(v))

            # Simply check if text_a and text_b are present
            if "text_a" in values_set or "text_b" in values_set:
                potential_annotators.append(col)

    return potential_annotators


def create_annotated_pairs(
    df: pd.DataFrame,
    dataset_name: str,
    principles: Mapping[int, str] | None = None,
    comparison_votes: (
        Mapping[int, Dict[int, Union[bool, str, None]]] | pd.Series | None
    ) = None,
    additional_columns: List[str] = None,
    auto_detect_annotators: bool = True,
) -> Dict:
    """Convert ICAI results to annotated pairs format using direct data inputs.

    Args:
        df: DataFrame with preference data pairs. Must have mandatory "text_a", "text_b", and DEFAULT_PREFERENCE_COLUMN rows, and an optional "input" (prompt).
        dataset_name: Name for the dataset
        additional_columns: List of additional columns from the training data to include as annotations
        principles: Dictionary of principles where keys are principle IDs
        comparison_votes: Dictionary of comparison votes
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

    df = df.copy()
    df = df.applymap(str)  # ensure all columns are hashable

    # Detect annotator columns if enabled
    detected_columns = []
    if auto_detect_annotators:
        detected_columns = detect_annotator_columns(df)
        if detected_columns:
            logger.info(f"Automatically detected annotator columns: {detected_columns}")

    # Combine detected columns with manually specified ones
    all_additional_columns = list(set((additional_columns or []) + detected_columns))

    # Add all annotators to the output
    add_annotators(
        output,
        principles,
        all_additional_columns,
    )

    # Identify metadata columns (columns that are not standard columns, not annotator columns, and not used for comparison content)
    standard_columns = {
        "text_a",
        "text_b",
        "model_a",
        "model_b",
        "input",
        DEFAULT_PREFERENCE_COLUMN,
    }
    metadata_columns = [
        col
        for col in df.columns
        if col not in standard_columns and col not in all_additional_columns
    ]

    # Add metadata columns to the overall metadata
    if metadata_columns:
        output["metadata"]["available_metadata_keys_per_comparison"] = metadata_columns
        logger.info(f"Available metadata columns: {metadata_columns}")

    # Prepare data needed for annotations
    default_annotator_id = output["metadata"]["default_annotator"]

    if principles is None or comparison_votes is None:
        if not (principles is None and comparison_votes is None):
            raise ValueError(
                "Got values for principles or comparison_votes, but not both. "
                "Please provide both or neither."
            )
        active_principles = []
    else:
        active_principles = list(principles.values())

    if principles is not None:
        active_principles = list(principles.values())
        assert (
            comparison_votes is not None
        ), "Comparison votes are required when principles are provided"
    else:
        active_principles = []

    # Process each comparison
    for idx, row in df.iterrows():
        # Extract model information for response_a and response_b
        response_a = {"text": row["text_a"]}
        response_b = {"text": row["text_b"]}

        # Add model information if available
        if "model_a" in row and pd.notna(row["model_a"]):
            response_a["model"] = row["model_a"]
        if "model_b" in row and pd.notna(row["model_b"]):
            response_b["model"] = row["model_b"]

        # Create unique ID for this comparison
        comparison_id = hash_comparison(response_a, response_b, row.get("input"))

        # Initialize annotations dict with default annotator annotation
        annotations = {}
        reference_preference = row[DEFAULT_PREFERENCE_COLUMN]
        # Map aliases to standard format
        assert (
            reference_preference in PREFERENCE_ALIAS_MAPPING.keys()
        ), f"Invalid reference preference: {reference_preference}"
        reference_preference = PREFERENCE_ALIAS_MAPPING[reference_preference]

        # Validate that the preference is in the correct format
        assert reference_preference in [
            "a",
            "b",
        ], f"Invalid preference value: {reference_preference}. Should be 'a' or 'b'."

        annotations[default_annotator_id] = {
            DEFAULT_PREFERENCE_KEY: reference_preference
        }

        if comparison_votes is not None:
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
                    annotations[column_annotator_id] = {
                        DEFAULT_PREFERENCE_KEY: str(row[col])
                    }

        # Create the comparison entry
        comparison = {
            "id": comparison_id,
            "prompt": row.get("input"),
            "response_a": response_a,
            "response_b": response_b,
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
    comparison_votes = icai.load_votes_per_comparison(results_path)

    # Call the core implementation with loaded data
    result = create_annotated_pairs(
        df=train_df,
        principles=principles,
        comparison_votes=comparison_votes,
        dataset_name=dataset_name,
        additional_columns=additional_columns,
        auto_detect_annotators=auto_detect_annotators,
    )

    return result


def save_annotated_pairs_to_file(
    annotated_pairs: Dict, output_file: str | Path
) -> None:
    """Save the annotated pairs to a JSON file.

    Args:
        annotated_pairs: The annotated pairs dataset to save
        output_file: Path to the output file
    """
    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotated_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Created annotated pairs format dataset: {output_file}")
    logger.info(f"- Dataset contains {len(annotated_pairs['comparisons'])} comparisons")
    logger.info(f"- Dataset contains {len(annotated_pairs['annotators'])} annotators")


def load_annotated_pairs_from_file(annotated_pairs_file: str | Path) -> Dict:
    """Load an annotated pairs dataset from a JSON file.

    Args:
        annotated_pairs_file: Path to the annotated pairs dataset file
    """
    return json.load(open(annotated_pairs_file, "r", encoding="utf-8"))


def merge_annotated_pairs(
    annotated_pairs_list: List[Dict], merged_metadata: Dict | None = None
) -> Dict:
    """Merge a list of annotated pairs datasets into a single annotated pairs dataset.

    Args:
        annotated_pairs_list: List of annotated pairs datasets to merge

    Returns:
        The merged annotated pairs dataset
    """
    if not annotated_pairs_list:
        raise ValueError("No annotated pairs datasets provided for merging")

    # Check that all datasets have the same format version
    format_versions = [
        dataset.get("metadata", {}).get("version") for dataset in annotated_pairs_list
    ]
    if not all(version == format_versions[0] for version in format_versions):
        raise ValueError(
            "All annotated pairs datasets must have the same format version"
        )

    original_descriptions = [
        dataset.get("metadata", {}).get("description")
        for dataset in annotated_pairs_list
    ]
    # Filter out None values
    original_descriptions = [desc for desc in original_descriptions if desc is not None]

    # Merge dataset names if available
    dataset_names = []
    for dataset in annotated_pairs_list:
        if dataset.get("metadata", {}).get("dataset_name"):
            dataset_names.append(dataset["metadata"]["dataset_name"])

    # Initialize merged dataset with metadata from the first dataset
    merged = {
        "metadata": {},
        "annotators": {},
        "comparisons": [],
    }

    if merged_metadata is not None:
        merged["metadata"].update(merged_metadata)
    else:
        merged["metadata"] = {
            "version": FORMAT_VERSION,
            "description": f"Merged annotated pairs dataset from multiple datasets ({', '.join(dataset_names)}). Original descriptions: {', '.join(original_descriptions)}",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_name": "merged_dataset",
        }

    if dataset_names:
        merged["metadata"]["dataset_name"] = "_".join(dataset_names)

    # Track comparison IDs to avoid duplicates
    comparison_ids = set()
    # Dictionary to store comparisons by ID for easy access
    comparison_map = {}

    # Merge annotators and comparisons
    for i, dataset in enumerate(annotated_pairs_list):
        # Merge annotators
        for annotator_id, annotator in dataset.get("annotators", {}).items():
            if annotator_id not in merged["annotators"]:
                merged["annotators"][annotator_id] = annotator

        # Copy default annotator if this is the first dataset and it has one
        if not merged["metadata"].get("default_annotator") and dataset.get(
            "metadata", {}
        ).get("default_annotator"):
            merged["metadata"]["default_annotator"] = dataset["metadata"][
                "default_annotator"
            ]

        # check that comparison ids are unique within each dataset
        dataset_comparison_ids = [
            comparison["id"] for comparison in dataset.get("comparisons", [])
        ]
        if len(dataset_comparison_ids) != len(set(dataset_comparison_ids)):
            raise ValueError(
                f"Comparison IDs are not unique in dataset {dataset_names[i]}. Could not merge datasets."
            )

        # Merge comparisons and their annotations
        for comparison in dataset.get("comparisons", []):
            comparison_id = comparison["id"]

            if comparison_id not in comparison_map:
                # First time seeing this comparison, add it to our map
                comparison_map[comparison_id] = comparison.copy()
                comparison_ids.add(comparison_id)
            else:
                # We've seen this comparison before, merge annotations
                existing_comparison = comparison_map[comparison_id]

                # Merge annotations
                for annotator_id, annotation in comparison.get(
                    "annotations", {}
                ).items():
                    if annotator_id not in existing_comparison["annotations"]:
                        existing_comparison["annotations"][annotator_id] = annotation
                    else:
                        assert (
                            existing_comparison["annotations"][annotator_id]
                            == annotation
                        ), f"Annotations for {annotator_id} on comparison {comparison_id} are not the same ({existing_comparison['annotations'][annotator_id]} != {annotation}). Could not merge datasets."

                # Merge metadata if present
                if "metadata" in comparison and "metadata" in existing_comparison:
                    existing_comparison["metadata"].update(
                        comparison.get("metadata", {})
                    )
                elif "metadata" in comparison:
                    existing_comparison["metadata"] = comparison["metadata"].copy()

    # Use the merged comparisons from our map
    merged["comparisons"] = list(comparison_map.values())

    logger.info(
        f"Merged {len(annotated_pairs_list)} datasets with {len(merged['comparisons'])} unique comparisons"
    )

    return merged

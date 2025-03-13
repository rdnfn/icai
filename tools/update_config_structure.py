#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

# Fields that need to be moved from annotator to annotator.alpaca_eval
FIELDS_TO_MOVE = [
    "create_other_annotator_tmp_configs",
    "constitution",
    "is_single_annotator",
    "base_constitutional_annotator_configs",
    "other_annotator_configs",
]


def update_config_file(file_path):
    """
    Update a single YAML config file to move deprecated annotator fields to the
    new alpaca_eval nested structure, while preserving the rest of the file.
    """
    print(f"Processing {file_path}")

    # Read the YAML file
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

    # Check if file has annotator section
    if not config or "annotator" not in config:
        print(f"  No annotator section found, skipping")
        return False

    # Check if any fields need to be moved
    fields_to_update = []
    for field in FIELDS_TO_MOVE:
        if field in config["annotator"]:
            fields_to_update.append(field)

    if not fields_to_update:
        print(f"  No deprecated fields found, skipping")
        return False

    # Create alpaca_eval section if it doesn't exist
    if "alpaca_eval" not in config["annotator"]:
        config["annotator"]["alpaca_eval"] = {}

    # Move fields - do this in two steps to avoid modifying while iterating
    for field in fields_to_update:
        # Copy value to alpaca_eval
        config["annotator"]["alpaca_eval"][field] = config["annotator"][field]

    # Now remove the fields from the top level
    for field in fields_to_update:
        del config["annotator"][field]

    # Write updated config back to file
    try:
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"  Error writing file: {e}")
        return False

    print(f"  Updated {len(fields_to_update)} fields: {', '.join(fields_to_update)}")
    return True


def update_all_configs():
    """
    Find and update all YAML config files in the exp/configs directory.
    """
    base_dir = "exp/configs"
    updated_count = 0
    processed_count = 0

    # Find all YAML config files recursively
    yaml_files = list(Path(base_dir).glob("**/*.yaml")) + list(
        Path(base_dir).glob("**/*.yml")
    )

    print(f"Found {len(yaml_files)} YAML config files")

    for yaml_file in yaml_files:
        processed_count += 1
        try:
            if update_config_file(str(yaml_file)):
                updated_count += 1
        except Exception as e:
            print(f"  Error processing {yaml_file}: {e}")

    print(f"Updated {updated_count} out of {processed_count} files")


if __name__ == "__main__":
    update_all_configs()

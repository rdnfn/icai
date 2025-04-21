#!/usr/bin/env python3
"""
Generate an annotated pairs dataset from legacy ICAI results (version <= 0.2.1). Note that more recent
versions of ICAI natively generate this format.

Usage:
  # Generate annotated pairs format from ICAI results directory
  python convert_to_annotated_pairs.py path/to/results ap_dataset.json
"""

import argparse
import sys

from loguru import logger

from inverse_cai.data.annotated_pairs_format import (
    results_to_annotated_pairs,
    save_annotated_pairs_to_file,
)


def main():
    """Main function to parse arguments and run the converter."""
    parser = argparse.ArgumentParser(
        description="Convert ICAI results to annotated pairs format."
    )
    parser.add_argument("results_dir", help="Path to ICAI results directory")
    parser.add_argument("output", help="Path to output annotated pairs JSON")
    parser.add_argument(
        "--dataset-name",
        "-n",
        default="ICAI Generated Dataset",
        help="Name for the dataset (default: ICAI Generated Dataset)",
    )
    parser.add_argument(
        "--additional-columns",
        "-a",
        nargs="+",
        default=[],
        help="Additional columns from the training data to include as annotations (default: none, example: -a 'column1 column2')",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        default=False,
        help="Disable automatic detection of annotator columns (default: auto detection is enabled)",
    )

    args = parser.parse_args()

    try:
        logger.info(f"Converting ICAI results from {args.results_dir}")
        annotated_pairs = results_to_annotated_pairs(
            results_dir=args.results_dir,
            dataset_name=args.dataset_name,
            additional_columns=args.additional_columns,
            auto_detect_annotators=not args.no_auto_detect,
        )
        save_annotated_pairs_to_file(annotated_pairs, args.output)
        logger.success(
            f"Successfully saved {len(annotated_pairs['comparisons'])} pairs to {args.output}"
        )
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise e

    return 0


if __name__ == "__main__":
    sys.exit(main())

import pandas as pd
import time
from pathlib import Path
from filelock import FileLock
from loguru import logger
import csv
import json
import ast


class VoteCache:
    def __init__(self, cache_path: Path, lock_timeout: int = 10):
        """Initialize vote cache with path to csv file.

        Args:
            cache_path: Path to csv file storing votes
            lock_timeout: Maximum time to wait for file lock in seconds
        """
        self.cache_path = Path(cache_path)
        self.index_path = self.cache_path.with_suffix(".index.json")
        self.lock_path = self.cache_path.with_suffix(".lock")
        self.lock_timeout = lock_timeout
        self.lock = FileLock(self.lock_path, timeout=lock_timeout)

        # Initialize empty cache files if they don't exist
        if not self.cache_path.exists():
            with open(self.cache_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "votes"])

        if not self.index_path.exists():
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(set(), f, default=list)

    def _get_processed_indices(self) -> set:
        """Load set of processed indices."""
        with open(self.index_path, "r", encoding="utf-8") as f:
            return set(json.load(f))

    def _add_processed_index(self, index: int):
        """Add index to processed set."""
        processed = self._get_processed_indices()
        processed.add(index)
        with open(self.index_path, "w") as f:
            json.dump(list(processed), f)

    def get_cached_votes(self) -> dict:
        """Get all cached votes as dictionary of index -> vote."""
        with self.lock:
            votes_dict = {}
            with open(self.cache_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["votes"] and pd.notna(row["votes"]):
                        votes_dict[int(row["index"])] = ast.literal_eval(row["votes"])
        return votes_dict

    def update_cache(self, index: int, vote: dict):
        """Update cache with new vote result by appending to file.
        Skips if index was already processed.

        Will retry if lock is not available.
        """
        max_retries = 10
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with self.lock:
                    # Check if already processed
                    processed = self._get_processed_indices()
                    if index in processed:
                        return

                    # Append new vote and update index
                    with open(self.cache_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([index, str(vote)])

                    self._add_processed_index(index)
                return
            except TimeoutError as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Lock acquisition failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise TimeoutError(
                        f"Failed to acquire lock after {max_retries} retries"
                    ) from exc

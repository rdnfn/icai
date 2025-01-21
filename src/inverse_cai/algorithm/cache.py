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

        # Initialize empty cache file if it doesn't exist
        if not self.cache_path.exists():
            with open(self.cache_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "votes"])

        if not self.index_path.exists():
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(set(), f, default=list)

    def get_processed_indices(self) -> set:
        """Load set of processed indices."""
        with open(self.index_path, "r", encoding="utf-8") as f:
            return set(json.load(f))

    def get_full_index(self, index: int, vote: dict) -> str:
        """Get full index string for a vote.

        Starting with comparison index, then lowest and highest principle id voted on.
        Note that the votes_per_comparison csv can contain multiple votes per
        comparison, so we need to be able to handle this.
        """
        min_principle_id_voted_on = min(vote.keys())
        max_principle_id_voted_on = max(vote.keys())
        return f"{index}_{min_principle_id_voted_on}_{max_principle_id_voted_on}"

    def check_if_index_processed(self, index: int, vote: dict) -> bool:
        """Check if index has been processed."""
        processed = self.get_processed_indices()
        return self.get_full_index(index, vote) in processed

    def _add_processed_index(self, index: int, vote: dict):
        """Add index to processed set."""
        processed = self.get_processed_indices()
        processed.add(self.get_full_index(index, vote))
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(list(processed), f)

    def get_cached_votes(self) -> dict:
        """Get all cached votes as dictionary of index -> dictionary of votes."""
        with self.lock:
            votes_dict = {}
            with open(self.cache_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["votes"] and pd.notna(row["votes"]):
                        index = int(row["index"])
                        vote = ast.literal_eval(row["votes"])
                        full_index = self.get_full_index(index, vote)
                        if full_index not in votes_dict:
                            votes_dict[full_index] = {}
                        votes_dict[full_index] = vote
        return votes_dict

    def update_cache(self, index: int, vote: dict):
        """Update cache with new vote result by appending to file.
        Multiple votes per index are allowed.

        Will retry if lock is not available.
        """
        max_retries = 10
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with self.lock:
                    if self.check_if_index_processed(index, vote):
                        print(
                            f"Cache warning: Index {index} already processed. Cache not updated."
                        )
                        return
                    # Append new vote
                    with open(self.cache_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([index, str(vote)])

                    self._add_processed_index(index, vote)
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

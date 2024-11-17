import pandas as pd
import time
from pathlib import Path
from filelock import FileLock
from loguru import logger


class VoteCache:
    def __init__(self, cache_path: Path, lock_timeout: int = 10):
        """Initialize vote cache with path to csv file.

        Args:
            cache_path: Path to csv file storing votes
            lock_timeout: Maximum time to wait for file lock in seconds
        """
        self.cache_path = Path(cache_path)
        self.lock_path = self.cache_path.with_suffix(".lock")
        self.lock_timeout = lock_timeout
        self.lock = FileLock(self.lock_path, timeout=lock_timeout)

        # Initialize empty cache file if it doesn't exist
        if not self.cache_path.exists():
            self._save_df(pd.DataFrame(columns=["index", "votes"]))

    def _save_df(self, df: pd.DataFrame):
        """Save dataframe to csv with index."""
        df.to_csv(self.cache_path, index=False)

    def _load_df(self) -> pd.DataFrame:
        """Load dataframe from csv."""
        return pd.read_csv(self.cache_path)

    def get_cached_votes(self) -> dict:
        """Get all cached votes as dictionary of index -> vote."""
        with self.lock:
            df = self._load_df()
            # Convert string representation of dict back to dict
            votes_dict = {
                row["index"]: eval(row["votes"])
                for _, row in df.iterrows()
                if pd.notna(row["votes"])
            }
        return votes_dict

    def update_cache(self, index: int, vote: dict):
        """Update cache with new vote result.

        Will retry if lock is not available.
        """
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with self.lock:
                    df = self._load_df()

                    # Update or append new row
                    new_row = pd.DataFrame({"index": [index], "votes": [str(vote)]})
                    df = pd.concat(
                        [df[df["index"] != index], new_row], ignore_index=True
                    )

                    self._save_df(df)
                return
            except TimeoutError as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Lock acquisition failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise TimeoutError(
                        "Failed to acquire lock after multiple retries"
                    ) from exc
